import pandas as pd
from config import Config
from model import hybrid_df, predict_sentiment

def get_user_recs(username, n=20):
    """
    Return the top‐n products for a user from hybrid_df.
    """
    df = hybrid_df
    if username in df.index:
        row = df.loc[username]
        return row.nlargest(n).index.tolist()
    return []

def filter_top5_by_sentiment(username, reviews_df):
    """
    reviews_df must have Config.PRODUCT_COL, Config.REVIEW_TEXT
    """
    top20 = get_user_recs(username, 20)
    scores = {}
    for prod in top20:
        texts = reviews_df.loc[
            reviews_df[Config.PRODUCT_COL] == prod,
            Config.REVIEW_TEXT
        ].dropna().tolist()
        if not texts:
            continue
        preds = predict_sentiment(texts)
        scores[prod] = preds.mean()
    # pick top 5 by highest positive ratio
    top5 = sorted(scores, key=lambda k: scores[k], reverse=True)[:5]
    return top5
