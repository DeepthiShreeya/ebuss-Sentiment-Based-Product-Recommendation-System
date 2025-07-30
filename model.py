import sys
import numpy as _np

# module‐alias hack so pickle can find the old numpy._core
sys.modules["numpy._core"] = _np.core

import pickle
from config import Config

# helper to load a pickle
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# load artifacts
sentiment_model = load_pickle(Config.SENTIMENT_MODEL_PATH)
vectorizer      = load_pickle(Config.VECTORIZER_PATH)
hybrid_df       = load_pickle(Config.HYBRID_DF_PATH)
train_r         = load_pickle(Config.TRAIN_R_PATH)


def predict_sentiment(texts):
    """
    texts: list of review strings
    returns: numpy array of 0/1 predictions
    """
    X = vectorizer.transform(texts)
    return sentiment_model.predict(X)
