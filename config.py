import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "pickles")

class Config:
    USER_COL       = "reviews_username"
    PRODUCT_COL    = "name"
    REVIEW_TEXT    = "reviews_text"
    RATING_COL     = "reviews_rating"
    SENTIMENT_COL  = "sentiment"

    # artifact filepaths
    SENTIMENT_MODEL_PATH = os.path.join(PICKLE_DIR, "sentiment_model.pkl")
    VECTORIZER_PATH      = os.path.join(PICKLE_DIR, "vectorizer.pkl")
    HYBRID_DF_PATH       = os.path.join(PICKLE_DIR, "hybrid_df.pkl")
    TRAIN_R_PATH         = os.path.join(PICKLE_DIR, "train_r.pkl")




