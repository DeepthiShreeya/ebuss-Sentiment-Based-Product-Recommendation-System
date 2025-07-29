import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # SECURITY
    SECRET_KEY = os.environ.get(
        'SECRET_KEY',
        'abcd'  
    )

    # FLASK
    DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1','true','yes')

    # Where your pickled artifacts live
    PICKLES_DIR = os.environ.get(
        'PICKLES_DIR',
        os.path.join(BASE_DIR, 'pickles')
    )

    # Individual artifact paths
    VECTORIZER_FILE      = os.path.join(PICKLES_DIR, 'vectorizer.pkl')
    SENTIMENT_MODEL_FILE = os.path.join(PICKLES_DIR, 'sentiment_model.pkl')
    HYBRID_DF_FILE       = os.path.join(PICKLES_DIR, 'hybrid_df.pkl')
    TRAIN_R_FILE         = os.path.join(PICKLES_DIR, 'train_r.pkl')

