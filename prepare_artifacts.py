import os, pickle, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.pairwise import pairwise_distances
from math import sqrt
import nltk; nltk.download('stopwords'); nltk.download('wordnet')

from config import RANDOM_STATE, DATA_PATH, AUG_PATH, OUTPUT_DIR, TOPK_CF
from utils import clean_text, synonym_replacement

# 1) Load & detect columns
df = pd.read_csv(DATA_PATH)
col_map = {'rating':None,'username':None,'product':None,'review_text':None}
for c in df.columns:
    lc = c.lower()
    if col_map['rating']    is None and 'rating' in lc: col_map['rating'] = c
    if col_map['username']      is None and any(k in lc for k in ('username')): col_map['username'] = c
    if col_map['product']   is None and any(k in lc for k in ('product','item','asin','name')): col_map['product'] = c
    if col_map['review_text'] is None and ('review' in lc and 'text' in lc): col_map['review_text'] = c
rating_col, user_col, product_col, text_col = col_map['rating'], col_map['username'], col_map['product'], col_map['review_text']
user_col = "reviews_username"  

# 2) Clean & label
df = df.dropna(subset=[user_col, text_col]).drop_duplicates().reset_index(drop=True)
df['sentiment'] = df[rating_col].apply(lambda x: 'positive' if x >= 4 else 'negative')
df['clean'] = df[text_col].apply(clean_text)

# 3) Augment negatives
neg_df = df[df['sentiment']=='negative']
aug_texts = [synonym_replacement(t, 2) for t in neg_df['clean']]
augment_data = pd.DataFrame({'clean':aug_texts,'sentiment':'negative'})
os.makedirs(os.path.dirname(AUG_PATH), exist_ok=True)
augment_data.to_csv(AUG_PATH, index=False)
train_sent_df = pd.concat([df[['clean','sentiment']], augment_data], ignore_index=True)

# 4) Vectorize & balance
vectorizer = TfidfVectorizer(max_features=5000)
X_all = vectorizer.fit_transform(train_sent_df['clean'])
y_all = (train_sent_df['sentiment']=='positive').astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE)
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_tr, y_tr = ros.fit_resample(X_tr, y_tr)

# 5) Train 4 models (tiny/no-grid to save RAM)
models = {}

# Logistic Regression (single config)
lr = LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_STATE)
lr.fit(X_tr, y_tr); models['LR'] = lr

# Random Forest (your chosen best)  ← we’ll fix as best
rf = RandomForestClassifier(n_estimators=150, max_depth=None,
                            random_state=RANDOM_STATE, n_jobs=1)
rf.fit(X_tr, y_tr); models['RF'] = rf

# XGBoost (optional; skip if RAM issues)
# comment out if it still crashes
try:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        random_state=RANDOM_STATE, tree_method='hist',
                        n_estimators=150, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, n_jobs=1)
    xgb.fit(X_tr, y_tr); models['XGB'] = xgb
except Exception as e:
    print("XGB skipped:", e)

# Naive Bayes
nb = MultinomialNB(alpha=1.0)
nb.fit(X_tr, y_tr); models['NB'] = nb

# ---- DIRECTLY SELECT RF as best ----
best_name = 'RF'
best_sent_model = models[best_name]
print('Best sentiment model (fixed):', best_name)


# 6) CF split (leave‑one‑out)
ratings      = df.pivot_table(index=user_col, columns=product_col, values=rating_col)
interactions = ratings.stack().reset_index().rename(columns={0:rating_col})
sel = interactions.groupby(user_col, group_keys=False).sample(1, random_state=RANDOM_STATE).index

test_i  = interactions.loc[sel].reset_index(drop=True)
train_i = interactions.drop(sel).reset_index(drop=True)
train_r = train_i.pivot(index=user_col, columns=product_col, values=rating_col)

# 7) Adjusted-cosine UBCF & IBCF
user_means    = train_r.mean(axis=1)
user_demeaned = train_r.sub(user_means, axis=0).fillna(0)
user_sim      = 1 - pairwise_distances(user_demeaned, metric='correlation')

item_means    = train_r.mean(axis=0)
item_demeaned = train_r.sub(item_means, axis=1).fillna(0)
item_sim      = 1 - pairwise_distances(item_demeaned.T, metric='correlation')

user_sim_df = pd.DataFrame(user_sim, index=train_r.index, columns=train_r.index)
item_sim_df = pd.DataFrame(item_sim, index=train_r.columns, columns=train_r.columns)

# ---- Vectorized UBCF (kept for completeness) ----
R = train_r.fillna(0).values
S = user_sim_df.values
raw_u = S.dot(R)
raw_u /= np.where(np.abs(S).sum(axis=1, keepdims=True)==0, 1, np.abs(S).sum(axis=1, keepdims=True))
pred_df = pd.DataFrame(raw_u, index=train_r.index, columns=train_r.columns)

# ---- Vectorized IBCF (this creates pred_df_item) ----
T = item_sim_df.values
raw_i = R.dot(T.T)
raw_i /= np.where(np.abs(T).sum(axis=0, keepdims=True)==0, 1, np.abs(T).sum(axis=0, keepdims=True))
pred_df_item = pd.DataFrame(raw_i, index=train_r.index, columns=train_r.columns)

# vectorized preds
R = train_r.fillna(0).values
# UBCF
S = user_sim_df.values
ub_raw = S.dot(R)
ub_raw /= np.where(np.abs(S).sum(axis=1, keepdims=True)==0, 1, np.abs(S).sum(axis=1, keepdims=True))
ub_df = pd.DataFrame(ub_raw, index=train_r.index, columns=train_r.columns)
# IBCF
T = item_sim_df.values
ib_raw = R.dot(T.T)
ib_raw /= np.where(np.abs(T).sum(axis=0, keepdims=True)==0, 1, np.abs(T).sum(axis=0, keepdims=True))
ib_df = pd.DataFrame(ib_raw, index=train_r.index, columns=train_r.columns)

# RMSE quick check
def rmse(pred_df_local):
    errs=[]
    for _,r in test_i.iterrows():
        u,i,t = r[user_col], r[product_col], r[rating_col]
        if u in pred_df_local.index and i in pred_df_local.columns:
            p = pred_df_local.loc[u,i]
            if not np.isnan(p): errs.append((p-t)**2)
    return sqrt(np.mean(errs)) if errs else np.nan

rmse_ub = rmse(ub_df)
rmse_ib = rmse(ib_df)
print(f"UBCF RMSE: {rmse_ub:.4f}, IBCF RMSE: {rmse_ib:.4f}")

best_cf_name = 'IBCF'
cf_matrix = pred_df_item.copy()
cf_matrix[~train_r.isna()] = -np.inf
print('Best CF (fixed):', best_cf_name)

# mask seen items
cf_matrix = cf_matrix.copy()
cf_matrix[~train_r.isna()] = -np.inf

# 8) Save artifacts
pickle.dump(train_r,      open(os.path.join(OUTPUT_DIR,'train_r.pkl'),'wb'))
pickle.dump(cf_matrix,    open(os.path.join(OUTPUT_DIR,'cf_matrix.pkl'),'wb'))
pickle.dump(vectorizer,   open(os.path.join(OUTPUT_DIR,'vectorizer.pkl'),'wb'))
pickle.dump(best_sent_model, open(os.path.join(OUTPUT_DIR,'sentiment_model.pkl'),'wb'))
pickle.dump({'rating_col':rating_col,'user_col':user_col,
             'product_col':product_col,'text_col':text_col,
             'best_sent_model':best_name,'best_cf':best_cf_name},
            open(os.path.join(OUTPUT_DIR,'meta.pkl'),'wb'))

print('Artifacts saved in', OUTPUT_DIR)