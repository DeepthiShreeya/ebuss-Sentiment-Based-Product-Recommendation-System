# prepare_artifacts.py
import os
import pickle
import numpy as np
import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model         import LogisticRegression
from sklearn.ensemble              import RandomForestClassifier
from xgboost                       import XGBClassifier
from sklearn.naive_bayes           import MultinomialNB
from sklearn.metrics.pairwise      import pairwise_distances
from imblearn.over_sampling        import RandomOverSampler

from config import RANDOM_STATE, DATA_PATH, AUG_PATH, OUTPUT_DIR, TOPK_CF
from utils  import clean_text, synonym_replacement

# Step 1: NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load & preprocess raw reviews
print("1) Loading data…")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['reviews_username','reviews_text']).reset_index(drop=True)
df['clean']     = df['reviews_text'].astype(str).apply(clean_text)
df['sentiment'] = (df['reviews_rating'] >= 4).astype(int)

# Step 3: Augment negative class
print("2) Augmenting negatives…")
neg = df[df['sentiment']==0].copy()
neg['clean'] = neg['clean'].apply(lambda x: synonym_replacement(x, n_sr=2))
os.makedirs(os.path.dirname(AUG_PATH), exist_ok=True)
neg[['clean','sentiment']].to_csv(AUG_PATH, index=False)

train_sent = pd.concat([df[['clean','sentiment']], neg[['clean','sentiment']]], ignore_index=True)

# Step 4: TF‑IDF + resample
print("3) Vectorizing & balancing…")
vec = TfidfVectorizer(max_features=5000)
X   = vec.fit_transform(train_sent['clean'])
y   = train_sent['sentiment']
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_res, y_res = ros.fit_resample(X, y)

# Step 5: Train & select best sentiment model
print("4) Training sentiment models…")
models = {
    'LR': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'RF': RandomForestClassifier(random_state=RANDOM_STATE),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
    'NB':  MultinomialNB()
}
best_score, best_model = 0, None
for name, m in models.items():
    m.fit(X_res, y_res)
    sc = m.score(X_res, y_res)
    print(f"   {name} score: {sc:.4f}")
    if sc > best_score:
        best_score, best_model = sc, m
sent_model = best_model
print(f"   → Best sentiment model: {type(sent_model).__name__}")

# Step 6: Build item‑based CF candidates only
print("5) Building item‑based CF candidates…")
R = df.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating',
    aggfunc='mean'
).fillna(0)

# Compute item–item cosine similarity (only ~200×200 matrix)
item_sim = 1 - pairwise_distances(R.T, metric='cosine')  # shape = (n_items, n_items)

# Raw CF score = R × sim_item
raw = R.values.dot(item_sim)

# Denominator = sum of abs(sim) per item (axis=0) → shape (1, n_items)
den = np.abs(item_sim).sum(axis=0, keepdims=True)
den[den == 0] = 1

cf_full = pd.DataFrame(raw/den, index=R.index, columns=R.columns)

# Top‑20 candidates per user
cf_candidates = {
    u: cf_full.loc[u].nlargest(TOPK_CF).index.tolist()
    for u in cf_full.index
}

# Step 7: Save pickles for deployment
print("6) Saving artifacts…")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "cf_candidates.pkl"),   "wb") as f: pickle.dump(cf_candidates, f)
with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"),      "wb") as f: pickle.dump(vec, f)
with open(os.path.join(OUTPUT_DIR, "sentiment_model.pkl"), "wb") as f: pickle.dump(sent_model, f)
with open(os.path.join(OUTPUT_DIR, "meta.pkl"),            "wb") as f:
    pickle.dump({'topk_cf': TOPK_CF, 'topk_final': 5}, f)

print("✅ Artifacts saved to", OUTPUT_DIR)
