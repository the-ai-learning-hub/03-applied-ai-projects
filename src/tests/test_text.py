"""Fast tests for src/text.py (run with ``uv run pytest``)."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.paths import data_dir
from src.text import (clean_tweet, evaluate, lexicon_predict, lexicon_score, load_lexicon, tokenize,
                      top_coefficients)


def test_clean_tweet_placeholders_and_case():
    raw = "@Anna LOOOOVE this &amp; that http://t.co/xyz USER"
    assert clean_tweet(raw) == "USER loove this & that URL USER"
    assert clean_tweet("My username is ok") == "my username is ok"
    assert clean_tweet(None) == "" and clean_tweet(float("nan")) == ""
    assert clean_tweet("Hi  THERE", lower=False) == "Hi THERE"


def test_tokenize_keeps_emoticons_hashtags_and_negations():
    assert tokenize("i don't like it :( #fail!") == ["i", "don't", "like", "it", ":(", "#fail", "!"]
    assert ":)" in tokenize("great day :)") and "<3" in tokenize("love u <3")
    assert tokenize("") == []


TOY = {"good": 2.0, "bad": -2.0, "cute": 1.5}


def test_lexicon_score_with_and_without_negation():
    tokens = tokenize("not good , but cute")
    assert lexicon_score(tokens, TOY, negation=False) == pytest.approx(3.5)
    assert lexicon_score(tokens, TOY, negation=True) == pytest.approx(-0.5)  # only "good" is flipped
    score, table = lexicon_score(tokenize("it isn't bad"), TOY, explain=True)
    assert score == pytest.approx(2.0)
    assert list(table.columns) == ["token", "lexicon", "negated", "counts as"]
    assert table["negated"].tolist() == [False, False, True]


def test_lexicon_predict_threshold():
    pred = lexicon_predict(["so good", "so bad", "nothing here"], TOY)
    assert pred.tolist() == ["pos", "neg", "pos"]  # score 0 counts as positive


def test_real_lexicons_load():
    vader = load_lexicon(data_dir() / "lexicons" / "vader_lexicon.txt")
    afinn = load_lexicon(data_dir() / "lexicons" / "afinn_lexicon.txt")
    assert len(vader) > 7000 and len(afinn) > 3000
    assert vader["good"] > 0 > vader["bad"] and afinn["love"] > 0 > afinn["hate"]
    assert ":)" in vader


def test_evaluate_majority_baseline():
    y_true = ["pos"] * 8 + ["neg"] * 2
    scores = evaluate(y_true, ["pos"] * 10)
    assert scores["accuracy"] == pytest.approx(0.8)
    assert scores["recall_neg"] == 0.0
    assert scores["macro_f1"] == pytest.approx((0 + 2 * 0.8 / 1.8) / 2)
    assert evaluate(["spam", "ham"], ["spam", "ham"], focus="spam")["recall_spam"] == 1.0


@pytest.mark.parametrize("model", [LogisticRegression(), MultinomialNB()])
def test_top_coefficients(model):
    texts = ["free prize now", "win free cash", "see you at lunch", "lunch at noon ok"]
    labels = ["spam", "spam", "ham", "ham"]
    vec = TfidfVectorizer()
    model.fit(vec.fit_transform(texts), labels)
    top = top_coefficients(vec, model, n=2)
    assert len(top) == 4 and set(top["towards"]) == {"ham", "spam"}
    assert top.iloc[0]["towards"] == "spam" and top.iloc[0]["word"] == "free"
    assert np.all(np.diff(top["weight"].iloc[:2]) <= 0)
