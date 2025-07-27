# app.py
from flask import Flask, render_template, request
from model   import recommend_top5

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():
    recs = []
    user = None
    if request.method=="POST":
        user = request.form.get("username","").strip()
        if user:
            recs = recommend_top5(user)
    return render_template("index.html", username=user, recommendations=recs)

if __name__=="__main__":
    app.run(debug=True)
