import os
import logging
from flask import Flask, render_template, request
from model import get_recommendations  # or however you named your function

app = Flask(__name__)

# Send debug & error logs to stdout so Render will capture them
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# Where our pickles actually live
PICKLES_DIR = os.path.join(os.path.dirname(__file__), 'pickles')
if not os.path.isdir(PICKLES_DIR):
    app.logger.error(f"🎯  Cannot find pickles directory at {PICKLES_DIR}")

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    recommendations = []
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        if not username:
            error = "Please enter a username."
        else:
            try:
                recommendations = get_recommendations(username)
            except Exception as e:
                app.logger.error("Error generating recommendations", exc_info=True)
                error = str(e)
    return render_template('index.html',
                           recommendations=recommendations,
                           error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
