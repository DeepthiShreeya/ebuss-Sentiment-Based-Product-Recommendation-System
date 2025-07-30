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
1. **Preprocessing & Sentiment Modeling**  
   - Clean & lemmatize review text  
   - Label ratings ≥ 4 as positive, < 4 as negative  
   - Augment negatives via synonym replacement  
   - Train four sentiment models (Logistic Regression, RandomForest, XGBoost, Multinomial NB) on TF‑IDF; select best by F1.

2. **Collaborative Filtering**  
   - Build both user‑based and item‑based adjusted‑cosine similarity matrices  
   - Choose the lower‑RMSE model on held‑out ratings  
   - Generate the top‑20 CF candidate products per user.

3. **Sentiment Re‑Ranking**  
   - For each CF candidate, predict sentiment across its reviews and compute the positive ratio  
   - Return the top‑5 products by highest positive‑sentiment ratio.

4. **Deployment**  
   - Build step: `pip install -r requirements.txt && python prepare_artifacts.py`  
   - Flask app (`app.py`, `model.py`) loads fresh pickles from `pickles/` at startup  
   - Frontend (`templates/index.html`) accepts a username and displays recommendations  
   - Hosted on Render at  <https://ebuss-sentiment-based-product-5eus.onrender.com/>
   
## Conclusions
- **Improved Relevance:** Integrating sentiment filtering reduces recommendations with high negative feedback.  
- **Scalable Pipeline:** Precomputing CF candidates and sentiment scores allows sub‑second response times in production.  
- **Reproducibility:** All artifacts (`hybrid_df.pkl`, `vectorizer.pkl`, `sentiment_model.pkl`, `train_r.pkl`) and raw data are included for end‑to‑end reruns.  
- **Seamless Deployment:** A single build command (`python prepare_artifacts.py`) regenerates models and artifacts; the Flask app can run locally or on Render with a standard `gunicorn app:app` start.

## Technologies Used
- **Python 3.9.13**  
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
