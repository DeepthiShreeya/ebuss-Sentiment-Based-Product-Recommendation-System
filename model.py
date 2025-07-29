import os
import pickle
import numpy as np
import pandas as pd
from config import Config

# --- load artifacts once at import time ---
with open(Config.VECTORIZER_FILE, 'rb') as f:
    vectorizer = pickle.load(f)

with open(Config.SENTIMENT_MODEL_FILE, 'rb') as f:
    sentiment_model = pickle.load(f)

with open(Config.HYBRID_DF_FILE, 'rb') as f:
    hybrid_df = pickle.load(f)   # expects columns ['reviews_username','name','reviews_text',...]

with open(Config.TRAIN_R_FILE, 'rb') as f:
    train_r = pickle.load(f)     # expects a DataFrame indexed by username, cols=name

def recommend_top20(username, topn=20):
    """Collaborative/hybrid rating lookup."""
    if username not in train_r.index:
        return []
    user_series = train_r.loc[username].sort_values(ascending=False)
    return user_series.head(topn).index.tolist()

def filter_top5_by_sentiment(products, topk=5):
    """From 20 recs, pick 5 with highest % positive reviews."""
    scores = []
    for p in products:
        texts = hybrid_df.loc[hybrid_df['name']==p, 'reviews_text']
        if texts.empty:
            continue
        X = vectorizer.transform(texts)
        preds = sentiment_model.predict(X)
        pos_pct = np.mean(preds == 1)
        scores.append((p, pos_pct))
    # sort descending by pos_pct
    scores.sort(key=lambda x: x[1], reverse=True)
    return [p for p,_ in scores[:topk]]

def get_top5(username):
    top20 = recommend_top20(username)
    return filter_top5_by_sentiment(top20)
