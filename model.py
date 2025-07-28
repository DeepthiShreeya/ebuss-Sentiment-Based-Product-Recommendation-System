import pickle
import numpy as np
import pandas as pd


def _load_pickle(path: str):
    """
    Utility to load a pickle artifact from disk.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# Load precomputed artifacts
cf_candidates = _load_pickle("pickles/cf_candidates.pkl")  # this is a dict
vectorizer   = _load_pickle("pickles/vectorizer.pkl")
sentiment_model = _load_pickle("pickles/sentiment_model.pkl")
meta_data       = _load_pickle('meta.pkl')

def recommend_top5(username: str) -> list:
    # If user not in our dict, no recommendations
    if username not in cf_candidates:
        return []

    # each value is a pandas.Series of {product_id: score}
    user_scores = cf_candidates[username]
    return user_scores.nlargest(5).index.tolist()