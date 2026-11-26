"""Small text-mining helpers shared by the Lesson 03 notebooks (SMS spam and tweet sentiment).

Everything here is deliberately short and readable: open the file and read it alongside the notebooks.

* :func:`clean_tweet` / :func:`tokenize` turn a raw tweet into a list of tokens (keeping ``USER``, ``URL``,
  hashtags and emoticons, because they carry signal).
* :func:`load_lexicon` and :func:`lexicon_score` score a text word by word with a sentiment lexicon,
  with a simple negation rule ("not good" → the sign of "good" flips until the next punctuation mark).
* :func:`evaluate` returns the three numbers we compare models with (accuracy, macro-F1, recall of a class).
* :func:`top_coefficients` lists the words that push a linear model most towards each class.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

# --------------------------------------------------------------------------------------------------
# Cleaning and tokenisation
# --------------------------------------------------------------------------------------------------

#: Emoticons we keep as single tokens (checked before splitting the rest of the text).
EMOTICON_RE = r"(?:[:;=8][\-o\*']?[\)\]\(\[dDpP/\\|@3]+|[\)\]\(\[dDpP/\\|]+[\-o\*']?[:;=8]|<3|\^_?\^)"
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")  # "soooo" -> "soo"
_SPACES_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b(?!(?:USER|URL)\b)\w+")
_TOKEN_RE = re.compile(
    rf"{EMOTICON_RE}"              # emoticons  :)  :-(  <3
    r"|#\w+"                       # hashtags   #fail
    r"|\w+(?:'\w+)?"               # words, incl. don't / it's
    r"|[!?.,;]"                    # punctuation that matters (ends a negation, shows emotion)
)


def clean_tweet(text: object, lower: bool = True) -> str:
    """Normalise one tweet (or SMS) so that similar texts look the same.

    Steps: missing → ``""``; HTML entities (``&amp;`` → ``&``); links → ``URL``; ``@name`` → ``USER``;
    letters repeated 3+ times shortened to 2 (``"sooooo"`` → ``"soo"``); spaces collapsed; optional lower case
    (the placeholders ``USER``/``URL`` are kept in upper case so they stand out).
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = html.unescape(str(text))
    text = _URL_RE.sub(" URL ", text)
    text = _MENTION_RE.sub(" USER ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _SPACES_RE.sub(" ", text).strip()
    if lower:  # lower-case every word except the two placeholders
        text = _WORD_RE.sub(lambda m: m.group().lower(), text)
    return text


def tokenize(text: str) -> list[str]:
    """Split a (cleaned) text into tokens: words, hashtags, emoticons and ``! ? . , ;``.

    >>> tokenize("i don't like it :( #fail")
    ['i', "don't", 'like', 'it', ':(', '#fail']
    """
    return _TOKEN_RE.findall(text or "")


# --------------------------------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------------------------------

#: Words that flip the sentiment of the words that follow them (until the next punctuation mark).
NEGATIONS = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "cannot", "without",
    "dont", "cant", "wont", "isnt", "aint", "didnt", "doesnt", "wasnt", "hardly", "barely",
    "don't", "can't", "won't", "isn't", "ain't", "didn't", "doesn't", "wasn't", "aren't", "weren't",
    "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't", "couldn't", "mustn't",
})
#: Tokens that end the scope of a negation.
SCOPE_END = frozenset({".", ",", "!", "?", ";", "but"})


def load_lexicon(path: str | Path) -> dict[str, float]:
    """Read a tab-separated lexicon (first column = token, second = score) into a ``dict``.

    Works for both ``vader_lexicon.txt`` (scores −4..+4) and ``afinn_lexicon.txt`` (−5..+5).
    """
    table = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["token", "score"],
                        quoting=3, keep_default_na=False, encoding="utf-8")
    return dict(zip(table["token"].astype(str).str.lower(), table["score"].astype(float)))


def lexicon_score(tokens: Sequence[str], lexicon: dict[str, float], negation: bool = True,
                  explain: bool = False) -> float | tuple[float, pd.DataFrame]:
    """Add up the lexicon score of every token; with ``negation=True`` flip signs after a negation word.

    The negation scope starts after a word in :data:`NEGATIONS` and ends at the next token in
    :data:`SCOPE_END` (punctuation or "but"). With ``explain=True`` also return a word-by-word table.
    """
    total, negated, rows = 0.0, False, []
    for token in tokens:
        t = token.lower()
        value = float(lexicon.get(t, 0.0))
        used = (-value if (negation and negated) else value) + 0.0  # + 0.0 avoids printing "-0.0"
        total += used
        rows.append({"token": token, "lexicon": value, "negated": negation and negated, "counts as": used})
        if t in SCOPE_END:
            negated = False
        elif t in NEGATIONS or t.endswith("n't"):
            negated = True
    total = round(total, 4)
    if explain:
        return total, pd.DataFrame(rows)
    return total


def lexicon_predict(texts: Iterable[str], lexicon: dict[str, float], negation: bool = True,
                    threshold: float = 0.0, pos: str = "pos", neg: str = "neg") -> np.ndarray:
    """Label each text ``pos`` if its lexicon score is ≥ ``threshold`` (texts are cleaned and tokenised here)."""
    scores = np.array([lexicon_score(tokenize(clean_tweet(t)), lexicon, negation=negation) for t in texts])
    return np.where(scores >= threshold, pos, neg)


# --------------------------------------------------------------------------------------------------
# Evaluation and interpretation
# --------------------------------------------------------------------------------------------------

def evaluate(y_true: Sequence, y_pred: Sequence, focus: str = "neg") -> dict[str, float]:
    """Return ``accuracy``, ``macro_f1`` and ``recall_<focus>`` (the recall of the class we care about).

    Macro-F1 averages the F1 of each class with the same weight, so a model that always predicts the
    majority class gets a low score even when its accuracy looks high.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"recall_{focus}": float(recall_score(y_true, y_pred, labels=[focus], average="macro",
                                              zero_division=0)),
    }


def top_coefficients(vectorizer, model, n: int = 15) -> pd.DataFrame:
    """Words with the most negative and most positive coefficients of a fitted linear model.

    Works for ``LogisticRegression``/``LinearSVC`` (``coef_``) and ``MultinomialNB`` (log-probability
    difference between the two classes). Positive weights push towards ``model.classes_[1]``.
    Returns columns ``word``, ``weight`` and ``towards`` (the class the word points to).
    """
    words = np.asarray(vectorizer.get_feature_names_out())
    if hasattr(model, "coef_"):
        weights = np.ravel(model.coef_)
    else:  # Naive Bayes: how much more likely the word is in class 1 than in class 0
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    order = np.argsort(weights)
    low, high = order[:n], order[::-1][:n]
    classes = list(model.classes_)
    frame = pd.DataFrame({
        "word": np.concatenate([words[high], words[low]]),
        "weight": np.concatenate([weights[high], weights[low]]),
        "towards": [classes[1]] * len(high) + [classes[0]] * len(low),
    })
    return frame.reset_index(drop=True)
