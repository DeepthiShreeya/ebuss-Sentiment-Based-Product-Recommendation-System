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
        raise KeyError(username)

    # get that user’s stored scores (might be Series, dict, or list)
    user_scores = cf_candidates.get(username, [])

    # if it’s already a Series, use nlargest
    if isinstance(user_scores, pd.Series):
        return user_scores.nlargest(5).index.tolist()

    # if it’s a dict (prod_id -> score), turn into a Series
    if isinstance(user_scores, dict):
        s = pd.Series(user_scores)
        return s.nlargest(5).index.tolist()

    # if it’s a list:
    #  • maybe it’s a list of (prod_id, score) tuples
    #  • or maybe just a list of prod_ids already sorted
    if isinstance(user_scores, list):
        if user_scores and isinstance(user_scores[0], tuple):
            # sort by score descending, take prod_ids
            sorted_by_score = sorted(user_scores, key=lambda x: x[1], reverse=True)
            return [pid for pid, _ in sorted_by_score[:5]]
        # otherwise assume it’s already a list of IDs
        return user_scores[:5]

    # fallback: coerce to Series
    try:
        s = pd.Series(user_scores)
        return s.nlargest(5).index.tolist()
    except Exception:
        return []