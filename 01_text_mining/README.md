# 01 · Text Mining: from words to sentiment

How does text become numbers? We start with six sentences and a pencil (bag of words, TF-IDF, cosine similarity), build a real SMS spam filter, and then race lexicons, six classic models and two transformers on 50,000 tweets. 🏠 [Lesson 03](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#text-mining)

| Notebook | Question | Time | Open |
|---|---|---|---|
| [01 · Text Mining basics: SMS spam](01_mini_project_sms_spam/01_Text_Mining_Basics_SMS_Spam.ipynb) | How does text become numbers? | 60 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/01_mini_project_sms_spam/01_Text_Mining_Basics_SMS_Spam.ipynb) |
| [02 · Tweet sentiment analysis](02_project_tweets/02_Tweet_Sentiment_Analysis.ipynb) | Can we beat the 83% of "always say positive"? | 90 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/02_project_tweets/02_Tweet_Sentiment_Analysis.ipynb) |
| [🎮 Challenge 1 · Fool the sentiment model](03_challenge_fool_the_model/Challenge_1_Fool_The_Sentiment_Model.ipynb) | Can you write a sentence that tricks both judges? | 15 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/03_challenge_fool_the_model/Challenge_1_Fool_The_Sentiment_Model.ipynb) |

**Order:** 01 → 02 (02 assumes bag of words, TF-IDF and text pipelines from 01).

## 🎯 Objectives

- **Text → numbers:** tokens, vocabulary, bag of words and TF-IDF computed by hand on a toy corpus, then with scikit-learn; cosine similarity as a tiny search engine.
- **A real spam filter:** `Pipeline(TfidfVectorizer + Naive Bayes / Logistic Regression)`, spam as the positive class, the model's "spam words", its mistakes, and a cell to test **your own messages**.
- **Clean text data honestly:** empty texts, duplicates and **train/test leakage**.
- **Sentiment, three ways:** lexicons (VADER, AFINN) with a negation rule → classic ML on TF-IDF (Naive Bayes, Logistic Regression, linear SVM, KNN, Random Forest, XGBoost) and sentence embeddings → transformers (pretrained Twitter-RoBERTa, fine-tuned DistilBERT), all on one **leaderboard** (macro-F1, recall of negatives).
- **Think like a practitioner:** class weights and thresholds, error analysis of the tweets every model gets wrong, and **score vs cost**.

## 📋 Prerequisites

Lesson 02 (train/test split, `Pipeline`, leakage, precision/recall/F1, confusion matrix, threshold tuning, PyTorch training loop) and the slides of the Text Mining block. Lesson 01's word counts and word clouds help.

## 📂 Data

| Folder | What | Size |
|---|---|---|
| `data/sms_spam/` | SMS Spam Collection (UCI, CC BY 4.0): 5,574 messages, `ham` / `spam` | 0.5 MB |
| `data/tweets/` | English tweets labelled `pos` / `neg` (train 39,936 · test 9,985) | 4.3 MB |
| `data/lexicons/` | VADER (MIT) and AFINN (ODbL) sentiment lexicons | 0.5 MB |

Details, sources, licences and known issues: [`data/DATA_DICTIONARY.md`](../data/DATA_DICTIONARY.md).

## ⏱️ Runtime on a laptop CPU

Measured on the authors' laptop (8 CPU threads), downloads excluded.

| Notebook | First run | Later runs |
|---|---|---|
| 01 · SMS spam | ≈ 25 s | ≈ 25 s |
| 02 · Tweets | ≈ 2.5 min (sentence embeddings are computed once) | ≈ 1.2 min (embeddings cached in `data/_cache/`) |

Fine-tuning DistilBERT takes ≈ 2 min on an RTX 4060 Laptop GPU but ≈ 30+ min on a CPU, so it is behind `RUN_HEAVY = False`: the notebook loads the precomputed run from [`02_project_tweets/results/`](02_project_tweets/results/) (`finetune_results.json` with metrics, timing and training history; `distilbert_test_probs.csv` and `roberta_test_probs.csv` with test-set probabilities). Set `RUN_HEAVY = True` on a GPU (Colab: *Runtime → Change runtime type → T4 GPU*) to train it live.

## 📦 Pretrained models (downloaded on first run, notebook 02)

| Model | Used for | Size |
|---|---|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | sentence embeddings (384 numbers per tweet) | ≈ 90 MB |
| [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | pretrained tweet sentiment (negative / neutral / positive) | ≈ 500 MB |
| [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) | fine-tuning (only with `RUN_HEAVY = True`) | ≈ 270 MB |

Weights go to the Hugging Face cache (`~/.cache/huggingface`). Fine-tuned models are saved in `models/` (git-ignored).

## ✍️ Your own text

Notebook 01 ends with `check_sms("...")` and a text box to type your own SMS; notebook 02's Exercise 4 compares your own tweets across a lexicon, Logistic Regression and RoBERTa.

## 🧰 Helpers

Shared code lives in [`src/text.py`](../src/text.py) (`clean_tweet`, `tokenize`, `load_lexicon`, `lexicon_score` with negation, `evaluate`, `top_coefficients`), tested in `src/tests/test_text.py` (`uv run pytest`).
