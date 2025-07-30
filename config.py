import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "pickles")

# Required by prepare_artifacts.py
RANDOM_STATE     = 42
DATA_PATH        = '/content/drive/MyDrive/EPGP in ML and AI/sample30.csv'
AUG_PATH         = '/content/drive/MyDrive/EPGP in ML and AI/augment_data.csv'
OUTPUT_DIR       = '/content/drive/MyDrive/EPGP in ML and AI/OUTPUT'
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




