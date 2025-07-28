import pickle
from utils import clean_text

# Load sentiment artifacts 
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
sentiment_model = pickle.load(open('sentiment_model.pkl','rb'))

def predict_sentiment(texts):
    X = vectorizer.transform([clean_text(t) for t in texts])
    return sentiment_model.predict(X)
