import os
import pickle
import pandas as pd
from utils import clean_text

# Helper to load pickle from pickles directory
def _load_pickle(filename):
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, 'pickles', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find pickle: {filename}")
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load raw CF candidates (could be dict or DataFrame)
_raw_cf = _load_pickle('cf_candidates.pkl')
if isinstance(_raw_cf, dict):
    # Convert dict[user] -> dict[item:score] into DataFrame
    cf_candidates = pd.DataFrame.from_dict(_raw_cf, orient='index')
else:
    cf_candidates = _raw_cf

# Load other artifacts
tf_vectorizer  = _load_pickle('vectorizer.pkl')
sentiment_model = _load_pickle('sentiment_model.pkl')
meta_data       = _load_pickle('meta.pkl')


def recommend_top5(username: str) -> list[dict]:
    """
    Return top-5 product recommendations for a user:
      1. Sort CF candidate scores descending
      2. Filter first 50 by positive sentiment on reviews
      3. Fill to 5 via fallback pure CF
    """
    # Validate user exists
    if username not in cf_candidates.index:
        raise KeyError(f"User '{username}' not in CF data")

    # Extract user's score series
    user_scores = cf_candidates.loc[username]
    if isinstance(user_scores, pd.DataFrame):
        # If multi-column, pick first col
        user_scores = user_scores.iloc[:, 0]

    # Sort products by score
    sorted_pids = user_scores.sort_values(ascending=False).index.tolist()

    recommendations = []
    checked = 0

    # Step 1: positive sentiment filter
    for pid in sorted_pids:
        if checked >= 50:
            break
        checked += 1

        # Clean and vectorize review text
        text_raw   = meta_data.at[pid, 'reviews_text'] if 'reviews_text' in meta_data.columns else ''
        text_clean = clean_text(text_raw)
        X          = tf_vectorizer.transform([text_clean])

        # Predict sentiment
        if sentiment_model.predict(X)[0] == 1:
            title = meta_data.at[pid, 'product_title'] if 'product_title' in meta_data.columns else str(pid)
            recommendations.append({'product_id': pid, 'product_title': title})
            if len(recommendations) >= 5:
                break

    # Step 2: fallback to pure CF
    idx = 0
    while len(recommendations) < 5 and idx < len(sorted_pids):
        pid = sorted_pids[idx]
        if pid not in {r['product_id'] for r in recommendations}:
            title = meta_data.at[pid, 'product_title'] if 'product_title' in meta_data.columns else str(pid)
            recommendations.append({'product_id': pid, 'product_title': title})
        idx += 1

    return recommendations
