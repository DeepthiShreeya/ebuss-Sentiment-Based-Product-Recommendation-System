import pickle
import pandas as pd

# load artifacts
with open("pickles/vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)
with open("pickles/sentiment_model.pkl","rb") as f:
    sent_model = pickle.load(f)
with open("pickles/train_r.pkl","rb") as f:
    train_r = pickle.load(f)            # user×item ratings matrix
with open("pickles/hybrid_df.pkl","rb") as f:
    hybrid_df = pickle.load(f)          # precomputed hybrid scores

def recommend(username, top_n=5):
    """
    1. Get top-20 from your recommender (e.g. ALS/hybrid_df).
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
