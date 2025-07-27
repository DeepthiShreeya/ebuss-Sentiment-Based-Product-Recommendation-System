# utils.py  (unchanged)
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem     import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_stopwords  = set(stopwords.words('english'))

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [w for w in s.split() if w not in _stopwords]
    return ' '.join(_lemmatizer.lemmatize(w) for w in tokens)

def synonym_replacement(sent: str, n_sr: int = 2) -> str:
    import random
    words = clean_text(sent).split()
    if not words:
        return sent
    new = words.copy()
    idxs = list(range(len(words)))
    random.shuffle(idxs)
    rep = 0
    for i in idxs:
        syns = {
            lemma.name().replace('_',' ')
            for syn in wordnet.synsets(words[i])
            for lemma in syn.lemmas()
        }
        syns.discard(words[i])
        if syns:
            new[i] = random.choice(list(syns))
            rep += 1
        if rep >= n_sr:
            break
    return ' '.join(new)
