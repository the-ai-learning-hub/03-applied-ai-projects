# 🎮 Challenge 1 · Fool the Sentiment Model

**Can you write a sentence that tricks an AI?** You attack the models of notebooks 01 and 02: make two sentiment judges say the wrong thing, then sneak a spam SMS past the spam filter. ⏱️ **≈ 15 min** · you play alone, on your own laptop (or in Colab).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/03_challenge_fool_the_model/Challenge_1_Fool_The_Sentiment_Model.ipynb) · 🏠 [Text Mining](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#challenge-1)

## The judges

| Judge | Model | Trained on |
|---|---|---|
| A | TF-IDF (words + bigrams) + Logistic Regression | the 39,000 tweets of notebook 02, trained in the notebook in ≈ 3 s (cached in `data/_cache/`) |
| B | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), pretrained | 124 M tweets; `neg` if *negative* is its top class, else `pos` (same rule as notebook 02) |
| C | TF-IDF + Logistic Regression spam filter | the SMS Spam Collection (2011), same split as notebook 01 |

## Rules

**Round 1 · Sarcasm attack:** write sentences with their **true** sentiment (`"pos"` / `"neg"`).
**Round 2 · Sneak past the spam filter:** write SMS that are **still spam** (honesty rule: `still_spam=True` only if it still asks for money, a click, a call or data).

| Rule | Points |
|---|---|
| Round 1: each judge (A, B) that predicts the opposite of your true label | +2 |
| Round 1: both judges fooled by the same sentence | +3 bonus |
| Round 2: a real spam with P(spam) < 0.5 (gets through) | +3 |
| Round 2: stealth, P(spam) < 0.10 | +2 bonus |
| Limits | first 8 different sentences, first 6 different messages |

**Medals:** 🥉 Bronze ≥ 10 · 🥈 Silver ≥ 25 · 🥇 Gold ≥ 40. The pre-filled examples score **33 (Silver)**.

## How to run

- **Locally:** `uv sync` once, open the notebook with the project's `.venv` kernel, *Run All*, then edit the two ✏️ **YOUR TURN** cells (`ROUND_1`, `ROUND_2`) and re-run from there.
- **Google Colab** (if your setup fails): click the badge above, *Runtime → Run all*. The first cell clones the repository.
- **Runtime:** ≈ 20 s on a laptop CPU (first run: + download of Twitter-RoBERTa, ≈ 500 MB, already there if you ran notebook 02).

The scoring rules live in [`src/challenges.py`](../../src/challenges.py) (`CH1_RULES`).
