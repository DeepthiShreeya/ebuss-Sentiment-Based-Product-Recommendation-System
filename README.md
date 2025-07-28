# eBuss Sentiment‑Based Product Recommendation System
> A Flask web application that delivers personalized product recommendations re‑ranked by review sentiment, powered by collaborative filtering and sentiment analysis.

## Table of Contents
* [General Information](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- **Background & Business Problem**  
  Traditional collaborative filtering can surface popular items, but may still recommend products with mixed or negative feedback. This project enhances an item‑based CF engine by re‑ranking the top‑20 candidates according to positive sentiment in product reviews, yielding more satisfying recommendations.
- **Dataset**  
  We use a subset of the Amazon Reviews dataset (sample30.csv), containing user IDs, product names, ratings, and review text.
- **Approach**  
  1. **Preprocessing & Sentiment Modeling**  
     - Clean and lemmatize review text.  
     - Label ratings ≥ 4 as positive, < 4 as negative.  
     - Augment negative examples via synonym replacement.  
     - Train a RandomForest sentiment classifier on TF‑IDF features.  
  2. **Item‑Based Collaborative Filtering**  
     - Build an item–item cosine similarity matrix from user–item ratings.  
     - Generate raw CF scores and select the top‑20 candidates per user.  
  3. **Sentiment Re‑Ranking**  
     - For each candidate, predict sentiment on all its reviews and compute the % positive.  
     - Return the top‑5 products with the highest positive‑sentiment ratio.  
  4. **Deployment**  
     - Flask backend (`app.py`, `model.py`) loads precomputed artifacts from `pickles/*.pkl`.  
     - Frontend template (`templates/index.html`) accepts a username and displays recommendations.  
     - Deployed live on Render: _Your Live App URL Here_.

## Conclusions
- **Improved Relevance:** Integrating sentiment filtering reduces recommendations with high negative feedback.  
- **Scalable Pipeline:** Precomputing CF candidates and sentiment scores allows sub‑second response times in production.  
- **Reproducibility:** All artifacts (`cf_candidates.pkl`, `vectorizer.pkl`, `sentiment_model.pkl`, `meta.pkl`) and raw data are included for end‑to‑end reruns.  
- **Seamless Deployment:** A single build command (`python prepare_artifacts.py`) regenerates models and artifacts; the Flask app can run locally or on Render with a standard `gunicorn app:app` start.

## Technologies Used
- **Python 3.11.9**  
- **Flask** – web framework  
- **pandas**, **NumPy** – data manipulation  
- **scikit‑learn**, **XGBoost**, **imbalanced‑learn** – modeling & resampling  
- **NLTK** – text cleaning & synonym replacement  
- **implicit** – (optional) for future ALS-based extensions  
- **Gunicorn** – production WSGI server  
- **Render** – cloud deployment  

## Acknowledgements
- Based on the Amazon Reviews dataset.  
- Inspired by collaborative filtering and sentiment analysis tutorials from the scikit‑learn documentation.  
- Thank you to the Render team for an easy‑to‑use deployment platform.

## Contact
Created by [@DeepthiShreeya](https://github.com/DeepthiShreeya) – feel free to reach out with questions or feedback!
