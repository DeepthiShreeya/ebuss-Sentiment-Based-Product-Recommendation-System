import os, pickle
from flask import Flask, request, render_template

BASE = os.path.dirname(__file__)

def _load_pickle(name):
    for p in (os.path.join(BASE, name),
              os.path.join(BASE, 'artifacts', name)):
        if os.path.exists(p):
            return pickle.load(open(p, 'rb'))
    raise FileNotFoundError(name)

# try hybrid first, else cf_matrix
try:
    hybrid_df = _load_pickle('hybrid_df.pkl')
except FileNotFoundError:
    hybrid_df = _load_pickle('cf_matrix.pkl')

train_r = _load_pickle('train_r.pkl')

# Precompute a simple popularity order once (items most rated)
_pop_order = train_r.notna().sum(axis=0).sort_values(ascending=False).index.tolist()

def _raw_cf_row(u):
    """Return the raw CF score row for user u (or empty Series)."""
    if u in hybrid_df.index:
        return hybrid_df.loc[u]
    return hybrid_df.iloc[0]*0  # empty-like

def recommend_top20(u, n=20):
    if u not in hybrid_df.index:
        return []
    # 1) get raw scores
    row = _raw_cf_row(u)

    # 2) drop seen
    seen = set(train_r.loc[u].dropna().index) if u in train_r.index else set()
    row = row.drop(seen, errors='ignore')

    # 3) take top-N
    recs = list(row.nlargest(n).index)

    # 4) backfill if not enough
    if len(recs) < n:
        # unseen popularity list
        extras = [it for it in _pop_order if it not in seen and it not in recs]
        recs.extend(extras[: n - len(recs)])
    return recs

def recommend_top5(u):
    return recommend_top20(u, 5)

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        user = request.form.get('username')
        recs = recommend_top5(user)
        return render_template('results.html', username=user, recommendations=recs)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
