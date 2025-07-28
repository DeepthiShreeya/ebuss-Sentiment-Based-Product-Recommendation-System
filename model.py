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
cf_candidates = _load_pickle("pickles/cf_candidates.pkl")
vectorizer   = _load_pickle("pickles/vectorizer.pkl")
sentiment_model = _load_pickle("pickles/sentiment_model.pkl")


def recommend_top5(username: str) -> list:
    """
    Given a username, return top-5 product recommendations.
    If the user is unknown, returns an empty list.
    """
    # If user not in CF index, no recommendations
    if username not in cf_candidates.index:
        return []

    # Get similarity scores for this user
    user_scores = cf_candidates.loc[username]

    # Select top-5 highest-scoring product IDs
    top_items = user_scores.nlargest(5).index.tolist()
    return top_items
