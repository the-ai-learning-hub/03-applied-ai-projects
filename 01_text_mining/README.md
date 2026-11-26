# 01 · Text Mining: from words to sentiment

How does text become numbers? We start with six sentences and a pencil (bag of words, TF-IDF, cosine similarity), build a real SMS spam filter, and then follow a full sentiment-analysis project on 50,000 tweets: exploration, ready-made tools, lexicons, machine learning and transformers. 🏠 [Lesson 03](../README.md) · 📖 [Slides](https://the-ai-learning-hub.github.io/03-applied-ai-projects/#text-mining)

| Notebook | Question | Time | Open |
|---|---|---|---|
| [01 · Text Mining basics: SMS spam](01_mini_project_sms_spam/01_Text_Mining_Basics_SMS_Spam.ipynb) | How does text become numbers? | 90 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/01_mini_project_sms_spam/01_Text_Mining_Basics_SMS_Spam.ipynb) |
| [02.0 · Tweets: exploratory data analysis](02_project_tweets/02.0_Tweets_EDA.ipynb) | What do positive and negative tweets look like? | 30 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/02_project_tweets/02.0_Tweets_EDA.ipynb) |
| [02.1 · Tweets: baseline tools](02_project_tweets/02.1_Tweets_Baseline_Tools.ipynb) | How good are ready-made tools (TextBlob, VADER, SentiWordNet, AFINN)? | 30 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/02_project_tweets/02.1_Tweets_Baseline_Tools.ipynb) |
| [02.2 · Tweets: lexicons](02_project_tweets/02.2_Tweets_Lexicons.ipynb) | Do sentiment dictionaries (NRC, VADER, AFINN) and negation handling help? | 30 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/02_project_tweets/02.2_Tweets_Lexicons.ipynb) |
| [02.3 · Tweets: machine learning](02_project_tweets/02.3_Tweets_Machine_Learning.ipynb) | Which models learn best from TF-IDF and GloVe? | 30 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/02_project_tweets/02.3_Tweets_Machine_Learning.ipynb) |
| [02.4 · Tweets: transformers](02_project_tweets/02.4_Tweets_Transformers.ipynb) | How much do pretrained and fine-tuned transformers win? | 30 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/02_project_tweets/02.4_Tweets_Transformers.ipynb) |
| [🎮 Challenge 1 · 🎚️ Flip It!](03_challenge_flip_it/Challenge_1_Flip_It.ipynb) | How few words must you change to make a sentiment AI change its mind? | 15 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/the-ai-learning-hub/03-applied-ai-projects/blob/main/01_text_mining/03_challenge_flip_it/Challenge_1_Flip_It.ipynb) |

**Order:** 01 → 02.0 → … → 02.4. Notebook 01 is the tutorial: it practises every concept of the Text Mining slides. The 02 notebooks are a real project, based on the *Text Mining para Ciência de Dados* project by Diogo Freitas, João Francisco Botas and Ana Sofia Trabulo (ISCTE, 2024/25; [project slides](https://github.com/the-ai-learning-hub/03-applied-ai-projects/releases/tag/tmcd-project)): they keep the outputs of the original runs (some steps needed a GPU), so you can read the results without re-running everything.

## 🎯 Objectives

- **Text → numbers:** tokens, vocabulary, bag of words and TF-IDF computed by hand on a toy corpus, then with scikit-learn; cosine similarity as a tiny search engine.
- **A real spam filter:** `Pipeline(TfidfVectorizer + Naive Bayes / Logistic Regression)`, spam as the positive class, the model's "spam words", its mistakes, and a cell to test **your own messages**.
- **Clean text data honestly:** empty texts, duplicates and **train/test leakage**.
- **A real project, step by step:** exploratory analysis → ready-made tools (TextBlob, VADER, SentiWordNet, AFINN) → lexicons (NRC EmoLex, VADER, AFINN) with and without negation handling → machine learning on TF-IDF and GloVe (Logistic Regression, SVM, Random Forest, KNN, XGBoost, RNN/LSTM/GRU/CNN) → transformers (zero-shot pipelines and fine-tuning).
- **Read results critically:** every notebook ends with a comparison chart, and ⚠️ *Limitation* boxes point out the mistakes in the original project (accuracy on an unbalanced dataset, train/test duplicates, "recurrent" networks that never see a sequence).

## 📋 Prerequisites

Lesson 02 (train/test split, `Pipeline`, leakage, precision/recall/F1, confusion matrix, threshold tuning, PyTorch training loop) and the slides of the Text Mining block. Lesson 01's word counts and word clouds help.

## 📂 Data

| Folder | What | Size |
|---|---|---|
| `data/sms_spam/` | SMS Spam Collection (UCI, CC BY 4.0): 5,574 messages, `ham` / `spam` | 0.5 MB |
| `data/tweets/` | English tweets labelled `pos` / `neg` (train 39,936 · test 9,985) | 4.3 MB |
| `data/lexicons/` | VADER (MIT), AFINN (ODbL) and NRC EmoLex (research and education use) sentiment lexicons | 1 MB |

Details, sources, licences and known issues: [`data/DATA_DICTIONARY.md`](../data/DATA_DICTIONARY.md).

## ⏱️ Runtime on a laptop CPU

Measured on the authors' laptop (8 CPU threads), downloads excluded.

| Notebook | First run | Later runs |
|---|---|---|
| 01 · SMS spam | ≈ 20 s (+ ≈ 90 MB MiniLM download) | ≈ 20 s |
| 02.0 – 02.2 · Tweets | seconds to a minute each | same |
| 02.3 – 02.4 · Tweets | outputs are saved: the heavy cells (⏳) took minutes to hours on a GPU | read the saved outputs |
| 🎮 Challenge 1 · Flip It! | ≈ 5 s (the model trains in ≈ 1.5 s) | ≈ 5 s |

## 📦 Pretrained models (notebook 02.4)

Notebook 02.4 uses Hugging Face transformers (DistilBERT, RoBERTa, `cardiffnlp/twitter-roberta-base-sentiment` and others; fine-tuning needs a GPU). Without a GPU the notebook still opens and shows the saved results.

## ✍️ Your own text

Notebook 01 ends with `check_sms("...")` and a text box to type your own SMS; Challenge 1 lets you type any sentence and watch the model react word by word.

## 🧰 Helpers

Shared code lives in [`src/tweet_tools.py`](../src/tweet_tools.py) (cleaning and evaluation helpers of the tweet project), [`src/text_cleaning_tuning.py`](../src/text_cleaning_tuning.py) (advanced cleaning + the grid search over cleaning choices) and [`src/text.py`](../src/text.py) (`clean_tweet`, `tokenize`, `load_lexicon`, `lexicon_score` with negation, `evaluate`, `top_coefficients`), tested in `src/tests/test_text.py` (`uv run pytest`). The Flip It! game (`FlipGame`: live word pushes, gauge, medals, scorecard) lives in [`src/challenges.py`](../src/challenges.py), tested in `src/tests/test_challenges.py`.
