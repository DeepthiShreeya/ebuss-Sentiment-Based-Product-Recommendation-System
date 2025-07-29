from flask import Flask, request, jsonify
from model import predict_sentiment, hybrid_df

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text_vector"]
    pred = predict_sentiment(data)
    return jsonify({"sentiment": int(pred[0])})

@app.route("/recommend/<username>")
def recommend(username):
    # use hybrid_df to fetch recommendations…
    recs = hybrid_df.get(username, [])  # whatever your logic is
    return jsonify({"user": username, "recs": recs})

if __name__ == "__main__":
    app.run()
