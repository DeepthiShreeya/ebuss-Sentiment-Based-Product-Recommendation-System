import os
import pickle
from flask import Flask, request, render_template

# Base directory for locating pickles
BASE = os.path.dirname(__file__)

def _load_pickle(name):
    """
    Attempt to load a pickle first from the project root,
    then from the artifacts/ subfolder.
    """
    for path in (
        os.path.join(BASE, name),
        os.path.join(BASE, 'artifacts', name)
    ):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not find pickle: {name}")

# Load the hybrid recommendation matrix (falls back to CF-only if needed)
try:
    hybrid_df = _load_pickle('hybrid_df.pkl')
except FileNotFoundError:
    hybrid_df = _load_pickle('cf_matrix.pkl')

# Load the training ratings matrix for masking seen items
train_r = _load_pickle('train_r.pkl')

# Precompute a global popularity order to backfill recommendations
_pop_order = (
    train_r.notna()
           .sum(axis=0)
           .sort_values(ascending=False)
           .index
           .tolist()
)

def _raw_cf_row(user_id):
    """
    Return the raw hybrid CF scores for a given user. If the user
    is unknown, return a zero‑vector Series with the same index.
    """
    if user_id in hybrid_df.index:
        return hybrid_df.loc[user_id]
    # return an all‑zero Series matching the columns
    return hybrid_df.iloc[0] * 0

def recommend_top20(user_id, n=20):
    """
    Generate up to n recommendations for user_id, excluding already
    seen items and backfilling with global popularity if needed.
    """
    # 1) get raw hybrid scores
    row = _raw_cf_row(user_id)

    # 2) mask out items the user has already rated
    if user_id in train_r.index:
        seen = set(train_r.loc[user_id].dropna().index)
        row = row.drop(labels=seen, errors='ignore')

    # 3) take the top‑n highest‑scoring items
    recs = list(row.nlargest(n).index)

    # 4) if fewer than n, backfill with unseen popular items
    if len(recs) < n:
        # find popular items the user hasn't seen or already been recommended
        extras = [
            item for item in _pop_order
            if item not in recs and (user_id not in train_r.index or item not in train_r.loc[user_id].dropna().index)
        ]
        recs.extend(extras[: n - len(recs)])

    return recs

def recommend_top5(user_id):
    """
    Simply wrap recommend_top20 to produce top‑5 recommendations.
    """
    return recommend_top20(user_id, 5)

# Initialize Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_id = request.form.get('username', '').strip()
        recs = recommend_top5(user_id)
        return render_template(
            'results.html',
            username=user_id,
            recommendations=recs
        )
    return render_template('index.html')

if __name__ == "__main__":
    # Use PORT in env if provided (Render sets $PORT)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
