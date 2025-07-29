import os

# base directory of the app
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# where your pickles live
PICKLES_DIR = os.path.join(BASE_DIR, "pickles")

VECTORIZER_PATH     = os.path.join(PICKLES_DIR, "vectorizer.pkl")
SENTIMENT_MODEL_PATH= os.path.join(PICKLES_DIR, "sentiment_model.pkl")
TRAIN_R_PATH        = os.path.join(PICKLES_DIR, "train_r.pkl")
HYBRID_DF_PATH      = os.path.join(PICKLES_DIR, "hybrid_df.pkl")

# port to listen on (Render sets PORT env var)
PORT = int(os.environ.get("PORT", 5000))
