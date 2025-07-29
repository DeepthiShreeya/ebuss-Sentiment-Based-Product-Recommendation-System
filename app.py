# app.py
from flask import Flask, request, render_template
from model import recommend

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recs = None
    if request.method == "POST":
        user = request.form.get("username").strip()
        try:
            recs = recommend(user)
        except Exception as e:
            recs = f"Error: {e}"
    return render_template("index.html", recommendations=recs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
