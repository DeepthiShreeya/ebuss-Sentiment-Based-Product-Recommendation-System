import os, pickle
from flask import Flask, render_template, request

app = Flask(__name__)

BASEDIR = os.path.abspath(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(BASEDIR, "pickles")

with open(os.path.join(PICKLE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(PICKLE_DIR, "sentiment_model.pkl"), "rb") as f:
    sentiment_model = pickle.load(f)
with open(os.path.join(PICKLE_DIR, "hybrid_df.pkl"), "rb") as f:
    hybrid_df = pickle.load(f)
with open(os.path.join(PICKLE_DIR, "train_r.pkl"), "rb") as f:
    train_r = pickle.load(f)

@app.route("/", methods=["GET","POST"])
def index():
    recommendation = None
    if request.method=="POST":
        username = request.form["username"]
        # call your recommendation + sentiment filter logic:
        recommendation = make_recommendation(username)
    return render_template("index.html", recommendations=recommendation)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
