from flask import Flask, render_template, request
from config import Config
import model

app = Flask(__name__, static_folder='static')
app.config.from_object(Config)

@app.route('/', methods=['GET','POST'])
def index():
    recs = None
    if request.method == 'POST':
        user = request.form.get('username','').strip()
        if user:
            recs = model.get_top5(user)
    return render_template('index.html', recommendations=recs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=app.config['DEBUG'])
