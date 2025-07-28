import os
import pickle
from utils import clean_text

# Helper to load pickle files from the pickles directory
def _load_pickle(fn):
    base = os.path.dirname(__file__)
    path = os.path.join(base, 'pickles', fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find pickle: {fn}")
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load your artifacts
cf_candidates   = _load_pickle('cf_candidates.pkl')
vectorizer      = _load_pickle('vectorizer.pkl')
sentiment_model = _load_pickle('sentiment_model.pkl')
meta_data       = _load_pickle('meta.pkl')

def recommend_top5(username: str) -> list[dict]:
    """
    Generate top-5 recommendations for `username` by
    1) pulling CF candidates,
    2) filtering for positive sentiment on each item's reviews,
    3) falling back to pure CF if fewer than 5 positives found.
    Returns a list of {'product_id': id, 'product_title': title}.
    """
    if username not in cf_candidates.index:
        raise KeyError(f"User '{username}' not found")

    # Sort items by CF score
    scores = cf_candidates.loc[username].sort_values(ascending=False).index.tolist()

    recs = []
    checked = 0

    # First pass: grab up to 5 items with positive sentiment
    for pid in scores:
        if checked >= 50:  # only check top‑50
            break
        checked += 1

        raw = meta_data.at[pid, 'reviews_text'] if 'reviews_text' in meta_data.columns else ''
        clean = clean_text(raw)
        vec = vectorizer.transform([clean])
        if sentiment_model.predict(vec)[0] == 1:  # positive
            title = meta_data.at[pid, 'product_title'] if 'product_title' in meta_data.columns else str(pid)
            recs.append({'product_id': pid, 'product_title': title})
            if len(recs) == 5:
                break

    # Fallback: fill to 5 with pure CF
    idx = 0
    while len(recs) < 5 and idx < len(scores):
        pid = scores[idx]
        if pid not in {r['product_id'] for r in recs}:
            title = meta_data.at[pid, 'product_title'] if 'product_title' in meta_data.columns else str(pid)
            recs.append({'product_id': pid, 'product_title': title})
        idx += 1

    return recs
