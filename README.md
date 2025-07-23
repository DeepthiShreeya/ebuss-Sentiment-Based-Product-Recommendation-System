# Sentiment-Based Product Recommender (eBuss)

## Quick Start (local)
```bash
python -m venv venv
# Windows PowerShell
your-repo> .\venv\Scripts\Activate.ps1
# Linux/Mac
$ source venv/bin/activate
pip install -r requirements.txt
python prepare_artifacts.py
cp artifacts/*.pkl .
python app.py  # or: gunicorn app:app
```

## Deploy to Render (free)
1. Push this folder to GitHub.
2. Go to render.com → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --preload --bind 0.0.0.0:$PORT`
5. Click Create. Wait. Open the given URL.

## Notebook link
Insert your deployed URL inside your Jupyter Notebook under a "Deployment Link" section.
"""