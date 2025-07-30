import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "pickles")

# Deployment‐time constants
RANDOM_STATE     = 42
DATA_PATH        = os.path.join(BASE_DIR, "data", "sample30.csv")      
AUG_PATH         = os.path.join(PICKLE_DIR, "augment_data.csv")
OUTPUT_DIR       = PICKLE_DIR  
TOPK_CF          = 20
ALPHA_BETA_GAMMA = (0.6, 0.3, 0.1)

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




