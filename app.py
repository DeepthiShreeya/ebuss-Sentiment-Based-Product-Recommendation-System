import os, pickle
from flask import Flask, render_template, request

app = Flask(__name__)

BASEDIR = os.path.abspath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(BASEDIR, "pickles")

with open(os.path.join(PICKLE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
# … same for sentiment_model.pkl, hybrid_df.pkl, train_r.pkl …

@app.route("/", methods=["GET","POST"])
def index():
    recommendation = None
    if request.method=="POST":
        username = request.form["username"]
        # call your recommendation + sentiment filter logic:
        recommendation = make_recommendation(username)
    return render_template("index.html", recommendations=recommendation)

