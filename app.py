import os
import pickle
import pandas as pd
from flask import Flask, render_template, request
import config

app = Flask(__name__)

# Load artifacts at startup
with open(config.VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
with open(config.SENTIMENT_MODEL_PATH, "rb") as f:
    sentiment_model = pickle.load(f)
with open(config.TRAIN_R_PATH, "rb") as f:
    train_r = pickle.load(f)           # e.g. user-item ratings matrix or similar
hybrid_df = pd.read_pickle(config.HYBRID_DF_PATH)  # full DF of recommendations

from utils import get_top20_recs, filter_top5_by_sentiment

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    if request.method == "POST":
        username = request.form["username"].strip()
        top20 = get_top20_recs(username, train_r, hybrid_df)
        recommendations = filter_top5_by_sentiment(top20, sentiment_model, vectorizer)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT)

