import pickle
import numpy as np
import pandas as pd
import os, pickle

BASE = os.path.dirname(__file__)
PICKLE_DIR = os.path.join(BASE, "pickles")

def _load_pickle(name: str):
    path = os.path.join(PICKLE_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


# Load precomputed artifacts
cf_candidates = _load_pickle("cf_candidates.pkl")  # this is a dict
vectorizer   = _load_pickle("vectorizer.pkl")
sentiment_model = _load_pickle("sentiment_model.pkl")
meta_data       = _load_pickle('meta.pkl')

def recommend_top5(username: str) -> list:
    # If user not in our dict, no recommendations
    if username not in cf_candidates:
        return []

    # each value is a pandas.Series of {product_id: score}
    user_scores = cf_candidates[username]
    return user_scores.nlargest(5).index.tolist()