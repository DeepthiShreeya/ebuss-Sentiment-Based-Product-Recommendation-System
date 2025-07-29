import os, pickle
import pandas as pd

BASE = os.path.dirname(__file__)
PICKLES_DIR = os.path.join(BASE, 'pickles')

def load_pickle(name):
    path = os.path.join(PICKLES_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

# preload your artifacts
vectorizer      = load_pickle('vectorizer.pkl')
sentiment_model = load_pickle('sentiment_model.pkl')
cf_candidates   = load_pickle('hybrid_df.pkl')   # or whatever your CF df is
train_r         = load_pickle('train_r.pkl')


def get_recommendations(username):
    """
    1. Get top-20 from our recommender (e.g. ALS/hybrid_df).
    2. For each, compute percent positive reviews via sent_model+vectorizer.
    3. Return top_n sorted by positivity.
    """
    # 1) base recommendations:
    user_ratings = hybrid_df.loc[username].sort_values(ascending=False).head(20)
    candidates = user_ratings.index.tolist()
    
    # 2) fetch all reviews for these products, predict sentiment, aggregate:
    # (assuming you have a df `reviews_df` with columns ['product_id','review_text'])
    # load here if needed:
    # reviews_df = pd.read_csv("path/to/reviews.csv")
    # For brevity, let's assume hybrid_df also stores a list of reviews:
    pos_rates = {}
    for pid in candidates:
        texts = hybrid_df.at[username, f"{pid}_reviews"]  # adjust to your storage
        X = vectorizer.transform(texts)
        preds = sent_model.predict(X)
        pos_rates[pid] = preds.mean()
    
    # 3) pick top_n by pos_rate
    top5 = sorted(pos_rates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [pid for pid, rate in top5]
