# model.py
import os
import pickle
import pandas as pd

from config import DATA_PATH, OUTPUT_DIR
from utils  import clean_text

# Load artifacts
with open(f"{OUTPUT_DIR}/cf_candidates.pkl","rb")   as f: cf_candidates   = pickle.load(f)
with open(f"{OUTPUT_DIR}/vectorizer.pkl","rb")      as f: vectorizer      = pickle.load(f)
with open(f"{OUTPUT_DIR}/sentiment_model.pkl","rb") as f: sentiment_model = pickle.load(f)
with open(f"{OUTPUT_DIR}/meta.pkl","rb")            as f: meta            = pickle.load(f)

TOPK_FINAL = meta['topk_final']

# Load full reviews for sentiment re‑scoring
df = pd.read_csv(DATA_PATH)
df['clean'] = df['reviews_text'].astype(str).apply(clean_text)

def recommend_top5(user: str):
    # 1) get up to 20 CF candidates
    cands = cf_candidates.get(user, [])
    pct   = {}
    # 2) for each product, predict all its review sentiments
    for p in cands:
        texts = df[df['name']==p]['clean']
        if texts.empty:
            pct[p] = 0.0
        else:
            preds = sentiment_model.predict(vectorizer.transform(texts))
            pct[p] = preds.sum()/len(preds)
    # 3) return top‑5 by % positive
    return [prod for prod,_ in sorted(pct.items(), key=lambda x: x[1], reverse=True)[:TOPK_FINAL]]
