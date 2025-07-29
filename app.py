from flask import Flask, render_template, request
import pandas as pd

from config import Config
from model import predict_sentiment, hybrid_df
from utils import filter_top5_by_sentiment

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.config.from_object(Config)

# load the full reviews dataset once
REVIEWS_DF = pd.read_csv("data/sample30.csv")  

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    username = request.form.get("username")
    if username not in hybrid_df[Config.USER_COL].values:
        return render_template(
            "index.html",
            error=f"User '{username}' not found."
        )
    top5 = filter_top5_by_sentiment(username, REVIEWS_DF)
    return render_template(
        "index.html",
        username=username,
        recommendations=top5
    )

if __name__ == "__main__":
    app.run(debug=True)
