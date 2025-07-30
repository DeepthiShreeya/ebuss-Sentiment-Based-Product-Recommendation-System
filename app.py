from flask import Flask, request, jsonify, render_template
from model import predict_sentiment, hybrid_df

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text_vector"]
    pred = predict_sentiment(data)
    return jsonify({"sentiment": int(pred[0])})

@app.route("/recommend", methods=["POST"])
def recommend_form():
    username = request.form["username"]
    if username in hybrid_df.index:
        recs = hybrid_df.loc[username].nlargest(5).index.tolist()
    else:
        recs = []
    return render_template("results.html", username=username, recommendations=recs)

@app.route("/recommend/<username>")
def recommend_api(username):
    if username in hybrid_df.index:
        recs = hybrid_df.loc[username].nlargest(5).index.tolist()
    else:
        recs = []
    return jsonify({"user": username, "recs": recs})

if __name__ == "__main__":
    app.run()

