import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
for pkg in ('stopwords', 'wordnet'):
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

_stop = set(stopwords.words('english'))
_lemm = WordNetLemmatizer()

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    toks = [w for w in s.split() if w not in _stop]
    return ' '.join(_lemm.lemmatize(w) for w in toks)

def synonym_replacement(sent: str, n_sr: int = 2) -> str:
    import random
    words = sent.split()
    if not words:
        return sent
    new = words.copy()
    idxs = list(range(len(words)))
    random.shuffle(idxs)
    rep = 0
    for i in idxs:
        syns = set(l.name().replace('_',' ') for syn in wordnet.synsets(words[i]) for l in syn.lemmas())
        syns.discard(words[i])
        if syns:
            new[i] = random.choice(list(syns))
            rep += 1
        if rep >= n_sr:
            break
    return ' '.join(new)