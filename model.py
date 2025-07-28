from flask import Flask, render_template, request
from model import recommend_top5

app = Flask(__name__)

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handle home page rendering and recommendation form submission.
    GET: render form.
    POST: fetch recommendations for given username.
    """
    username = None
    recommendations = []

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        if username:
            try:
                recommendations = recommend_top5(username)
            except KeyError:
                recommendations = []

    return render_template(
        'index.html',
        username=username,
        recommendations=recommendations
    )

if __name__ == '__main__':
    # Development server
    app.run(debug=True)
