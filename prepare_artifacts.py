import os, pickle
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.pairwise import pairwise_distances
import nltk

# download NLTK data if needed
nltk.download('stopwords')
nltk.download('wordnet')

from config import RANDOM_STATE, DATA_PATH, AUG_PATH, OUTPUT_DIR, TOPK_CF, ALPHA_BETA_GAMMA
from utils import clean_text, synonym_replacement

# 1) Load & detect columns
df = pd.read_csv(DATA_PATH)
col_map = {'rating':None,'user':None,'product':None,'review_text':None}
for c in df.columns:
    lc = c.lower()
    if col_map['rating'] is None and 'rating' in lc:      col_map['rating'] = c
    if col_map['user']   is None and any(k in lc for k in ('user','username','userid')): col_map['user'] = c
    if col_map['product'] is None and any(k in lc for k in ('product','item','asin','name')):  col_map['product'] = c
    if col_map['review_text'] is None and ('review' in lc and 'text' in lc): col_map['review_text'] = c
rating_col, user_col, product_col, text_col = (
    col_map['rating'], col_map['user'], col_map['product'], col_map['review_text']
)

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
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
)
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_tr, y_tr = ros.fit_resample(X_tr, y_tr)

# 5) Train 4 models with small grids
models = {}
# Logistic Regression
gr_lr = GridSearchCV(
    LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    {'C':[0.1,1]}, cv=3, scoring='f1', n_jobs=-1
)
gr_lr.fit(X_tr, y_tr); models['LR'] = gr_lr.best_estimator_
# Random Forest
gr_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    {'n_estimators':[100,150]}, cv=3, scoring='f1', n_jobs=-1
)
gr_rf.fit(X_tr, y_tr); models['RF'] = gr_rf.best_estimator_
# XGBoost
gr_xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
    {'n_estimators':[100], 'max_depth':[3]}, cv=3, scoring='f1', n_jobs=-1
)
gr_xgb.fit(X_tr, y_tr); models['XGB'] = gr_xgb.best_estimator_
# Naive Bayes
gr_nb = GridSearchCV(
    MultinomialNB(), {'alpha':[1.0]}, cv=3, scoring='f1', n_jobs=-1
)
gr_nb.fit(X_tr, y_tr); models['NB'] = gr_nb.best_estimator_

# Select best sentiment model
y_pred = {name: m.predict(X_te) for name,m in models.items()}
f1s = {name: f1_score(y_te, pred, zero_division=0) for name,pred in y_pred.items()}
best_name = max(f1s, key=f1s.get)
best_sent_model = models[best_name]
print('Best sentiment model (fixed):', best_name)

# 6) CF leave-one-out split
df_ratings    = df.pivot_table(index=user_col, columns=product_col, values=rating_col)
interactions   = df_ratings.stack().reset_index().rename(columns={0:rating_col})
sel            = interactions.groupby(user_col, group_keys=False).sample(1, random_state=RANDOM_STATE).index
test_i         = interactions.loc[sel].reset_index(drop=True)
train_i        = interactions.drop(sel).reset_index(drop=True)
train_r        = train_i.pivot(index=user_col, columns=product_col, values=rating_col)

# 7) Adjusted-cosine UBCF & IBCF
user_means    = train_r.mean(axis=1)
user_demeaned = train_r.sub(user_means, axis=0).fillna(0)
user_sim      = 1 - pairwise_distances(user_demeaned, metric='correlation')
user_sim_df   = pd.DataFrame(user_sim, index=train_r.index, columns=train_r.index)

item_means    = train_r.mean(axis=0)
item_demeaned = train_r.sub(item_means, axis=1).fillna(0)
item_sim      = 1 - pairwise_distances(item_demeaned.T, metric='correlation')
item_sim_df   = pd.DataFrame(item_sim, index=train_r.columns, columns=train_r.columns)

R_mat = train_r.fillna(0).values

# UBCF pred
ub_raw  = user_sim_df.values.dot(R_mat)
ub_norm = np.abs(user_sim_df.values).sum(axis=1, keepdims=True)
ub_norm[ub_norm==0] = 1
ub_df    = pd.DataFrame(ub_raw/ub_norm, index=train_r.index, columns=train_r.columns)
# IBCF pred
ib_raw  = R_mat.dot(item_sim_df.values.T)
ib_norm = np.abs(item_sim_df.values).sum(axis=0, keepdims=True)
ib_norm[ib_norm==0] = 1
ib_df    = pd.DataFrame(ib_raw/ib_norm, index=train_r.index, columns=train_r.columns)

# Compute RMSE by iterating through held-out test set
errs_ub, errs_ib = [], []
for _, row in test_i.iterrows():
    u, i, t = row[user_col], row[product_col], row[rating_col]
    if u in ub_df.index and i in ub_df.columns:
        p = ub_df.loc[u, i]
        if not np.isnan(p): errs_ub.append((p - t)**2)
    if u in ib_df.index and i in ib_df.columns:
        p = ib_df.loc[u, i]
        if not np.isnan(p): errs_ib.append((p - t)**2)
rmse_ub = sqrt(np.mean(errs_ub)) if errs_ub else np.nan
rmse_ib = sqrt(np.mean(errs_ib)) if errs_ib else np.nan

# Choose best CF
best_cf_name = 'UBCF' if (not np.isnan(rmse_ub) and rmse_ub < rmse_ib) else 'IBCF'
print('Best CF:', best_cf_name)

# 8) Blend CF + Sentiment + Popularity
alpha, beta, gamma = ALPHA_BETA_GAMMA
cf_blend = alpha * ub_df + (1 - alpha) * ib_df

# sentiment score by item
sent_preds = best_sent_model.predict(vectorizer.transform(df['clean']))
df['sent_score'] = (sent_preds == 1).astype(int)
sentiment_score_df = df.groupby(product_col)['sent_score'].mean()
sentiment_score_df = sentiment_score_df.reindex(cf_blend.columns, fill_value=sentiment_score_df.mean())

# popularity score by item
pop_df = df[product_col].value_counts(normalize=True)
pop_df = pop_df.reindex(cf_blend.columns, fill_value=0)

# final hybrid
hybrid_df = cf_blend.copy()
hybrid_df = hybrid_df.add(beta * sentiment_score_df.values, axis=1)
hybrid_df = hybrid_df.add(gamma * pop_df.values, axis=1)
hybrid_df[~train_r.isna()] = -np.inf

# 9) Save artifacts
os.makedirs(OUTPUT_DIR, exist_ok=True)
pickle.dump(train_r,        open(f"{OUTPUT_DIR}/train_r.pkl",        'wb'))
pickle.dump(cf_blend,       open(f"{OUTPUT_DIR}/cf_matrix.pkl",      'wb'))
pickle.dump(hybrid_df,      open(f"{OUTPUT_DIR}/hybrid_df.pkl",      'wb'))
pickle.dump(vectorizer,     open(f"{OUTPUT_DIR}/vectorizer.pkl",     'wb'))
pickle.dump(best_sent_model,open(f"{OUTPUT_DIR}/sentiment_model.pkl","wb"))
pickle.dump({
    'rating_col': rating_col,
    'user_col':   user_col,
    'product_col':product_col,
    'best_sent':  best_name,
    'best_cf':    best_cf_name
}, open(f"{OUTPUT_DIR}/meta.pkl","wb"))

print('Artifacts saved in', OUTPUT_DIR)
