import os
import pickle
from flask import Flask, render_template, request, flash, redirect, url_for
from model import recommend_top5

# Utility to load pickles from the pickles directory
def _load_pickle(filename):
    base = os.path.dirname(__file__)
    path = os.path.join(base, 'pickles', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find pickle: {filename}")
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load serialized artifacts (must match files in pickles/)
cf_candidates    = _load_pickle('cf_candidates.pkl')
vectorizer       = _load_pickle('vectorizer.pkl')
sentiment_model  = _load_pickle('sentiment_model.pkl')
meta_data        = _load_pickle('meta.pkl')

app = Flask(__name__)
# Use an env var for secret key in production
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handle home page rendering and recommendation form submission.
    - GET: Render the input form without recommendations.
    - POST: Process submitted username and display recommendations.
    """
    username = None
    recommendations = []

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        if username:
            try:
                recommendations = recommend_top5(username)
            except KeyError:
                flash(f"User '{username}' not found.", 'error')
                return redirect(url_for('home'))

    return render_template(
        'index.html',
        username=username,
        recommendations=recommendations
    )

if __name__ == '__main__':
    # Pick up $PORT on Render, default to 5000 locally
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
