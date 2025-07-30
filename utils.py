import re
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from config import Config

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Lowercase, remove non-alphabetic characters, tokenize, remove stopwords, and lemmatize.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS
    ]
    return " ".join(cleaned)

def synonym_replacement(text: str, n: int) -> str:
    """
    Replace up to n non-stopword tokens in the text with a random synonym from WordNet.
    """
    tokens = text.split()
    new_tokens = tokens.copy()
    candidate_idxs = [i for i, t in enumerate(tokens) if t not in STOP_WORDS]
    random.shuffle(candidate_idxs)

    replaced = 0
    for idx in candidate_idxs:
        word = tokens[idx]
        synonyms = set(
            lemma.name().replace('_', ' ').lower()
            for syn in wordnet.synsets(word)
            for lemma in syn.lemmas()
            if lemma.name().lower() != word
        )
        if synonyms:
            new_tokens[idx] = random.choice(list(synonyms))
            replaced += 1
        if replaced >= n:
            break

    return " ".join(new_tokens)

def get_user_recs(username: str, n: int = 20) -> list[str]:
    """
    Return the top‑n products for a user from hybrid_df.
    """
    from model import hybrid_df
    if username in hybrid_df.index:
        row = hybrid_df.loc[username]
        return row.nlargest(n).index.tolist()
    return []

def filter_top5_by_sentiment(username: str, reviews_df: pd.DataFrame) -> list[str]:
    """
    Given a DataFrame of reviews (with Config.PRODUCT_COL and Config.REVIEW_TEXT),
    pick the top‑20 collaborative recommendations, score each by its mean
    predicted sentiment, and return the top‑5 products.
    """
    from model import predict_sentiment
    top20 = get_user_recs(username, 20)
    scores: dict[str, float] = {}
    for prod in top20:
        texts = reviews_df.loc[
            reviews_df[Config.PRODUCT_COL] == prod,
            Config.REVIEW_TEXT
        ].dropna().tolist()
        if not texts:
            continue
        preds = predict_sentiment(texts)
        scores[prod] = preds.mean()

    # Return the 5 products with highest positive‑sentiment ratio
    return sorted(scores, key=lambda p: scores[p], reverse=True)[:5]
