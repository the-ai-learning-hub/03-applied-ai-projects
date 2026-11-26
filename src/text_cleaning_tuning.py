"""Advanced tweet cleaning and a grid search over the cleaning choices (tweet project, notebook 02.3).

``clean_text_ultimate`` switches every cleaning step on or off; ``run_all_experiments`` tries
all 1,024 combinations with TF-IDF + Logistic Regression and keeps the best F1-score.
"""

from __future__ import annotations

import itertools
import re

import emoji
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import unidecode
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

from src.paths import cache_dir

# Initial setup
for _pkg in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"):
    nltk.download(_pkg, quiet=True)
tqdm.pandas()

dataProcessed_dir = cache_dir() / "processed"
dataProcessed_dir.mkdir(parents=True, exist_ok=True)

# Stopwords, but keeping the negations
negation_words = {
    'no', 'not', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',
    "n't", "cannot", "can't", "dont", "don't", "doesn't", "isn't", "wasn't",
    "aren't", "hasn't", "haven't", "hadn't", "won't", "wouldn't", "shouldn't",
    "couldn't", "mightn't", "mustn't", "shan't", "without", "barely",
    "hardly", "scarcely", "rarely", "seldom", "nor", "ain't"
}
stop_words = set(stopwords.words('english')) - negation_words


def get_wordnet_pos(treebank_tag):
    """Map a Penn Treebank POS tag to the WordNet tag the lemmatizer expects."""
    if treebank_tag.startswith('J'):  # adjective
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):  # verb
        return wordnet.VERB
    elif treebank_tag.startswith('N'):  # noun
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):  # adverb
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text_ultimate(text, apply_lowercase=True, remove_emojis=True, do_stemming=True, use_pos=True,
                        remove_urls=True, remove_hashtags=True, remove_numbers=True,
                        handle_negation=True):
    if not isinstance(text, str):
        return ""
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    lemmatizer = WordNetLemmatizer()

    if apply_lowercase:
        text = text.lower()

    # Remove URLs, hashtags, numbers and emojis
    if remove_urls:
        text = re.sub(r"http\S+|www.\S+|\burl\b", "", text)
    if remove_hashtags:
        text = re.sub(r"#", "", text)
    if remove_numbers:
        text = re.sub(r"\d+", "", text)
    if remove_emojis:
        text = emoji.replace_emoji(text, replace='')

    text = re.sub(r'^(rt[:\s]*(user\s*)+)', '', text, flags=re.IGNORECASE).strip()
    text = unidecode.unidecode(text)

    tokens = tokenizer.tokenize(text)

    # Negation handling: every token after a negation (until punctuation) gets a NEG_ prefix
    if handle_negation:
        negation_scope = False
        punctuations = {'.', ',', ';', ':', '!', '?'}
        processed_tokens = []
        for token in tokens:
            if token in negation_words:
                negation_scope = True
                processed_tokens.append(token)
                continue
            if token in punctuations:
                negation_scope = False
            if negation_scope:
                token = f"NEG_{token}"
            processed_tokens.append(token)
        tokens = processed_tokens

    # POS-aware lemmatization, or stemming
    filtered = []
    if use_pos:
        tagged = pos_tag(tokens)
        for word, tag in tagged:
            if word not in stop_words:
                wn_tag = get_wordnet_pos(tag)
                word = lemmatizer.lemmatize(word, pos=wn_tag)
                filtered.append(word)
    else:
        if do_stemming:
            stemmer = SnowballStemmer("english")
            filtered = [stemmer.stem(w) for w in tokens if w not in stop_words]
        else:
            filtered = [w for w in tokens if w not in stop_words]

    filtered = [word for word in filtered if len(word) > 1]
    return " ".join(filtered)


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f"Confusion matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, model_name="Model"):
    print(f"\nClassification report - {model_name}")
    print(classification_report(y_true, y_pred))


def run_experiment(df_train, df_test, apply_lowercase=True, remove_emojis=True, do_stemming=True,
                   use_pos=True, remove_urls=True, remove_hashtags=True,
                   remove_numbers=True, handle_negation=True,
                   ngram_range=(1, 1), max_features=5000):
    """Clean with one combination of choices, then train and score TF-IDF + Logistic Regression."""
    print("\n--- Cleaning parameters ---")
    print(f"Lowercase: {apply_lowercase}")
    print(f"Remove emojis: {remove_emojis}")
    print(f"Stemming: {do_stemming}")
    print(f"POS lemmatization: {use_pos}")
    print(f"Remove URLs: {remove_urls}")
    print(f"Remove hashtags: {remove_hashtags}")
    print(f"Remove numbers: {remove_numbers}")
    print(f"Handle negation: {handle_negation}")
    print(f"N-gram range: {ngram_range}")
    print(f"Max features: {max_features}")

    # Make sure every text is a valid string
    df_train['text'] = df_train['text'].fillna('')
    df_test['text'] = df_test['text'].fillna('')

    options = dict(apply_lowercase=apply_lowercase, remove_emojis=remove_emojis, do_stemming=do_stemming,
                   use_pos=use_pos, remove_urls=remove_urls, remove_hashtags=remove_hashtags,
                   remove_numbers=remove_numbers, handle_negation=handle_negation)
    df_train['clean_text'] = df_train['text'].progress_apply(clean_text_ultimate, **options)
    df_test['clean_text'] = df_test['text'].progress_apply(clean_text_ultimate, **options)

    # Vectorization
    tfidf = TfidfVectorizer(lowercase=True, max_features=max_features, ngram_range=ngram_range)
    X_train = tfidf.fit_transform(df_train['clean_text'])
    X_test = tfidf.transform(df_test['clean_text'])
    y_train = df_train['class']
    y_test = df_test['class']

    # Model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    config_desc = (f"Lowercase={apply_lowercase}|Emojis={remove_emojis}|Stem={do_stemming}|POS={use_pos}|"
                   f"URLs={remove_urls}|hashtags={remove_hashtags}|numbers={remove_numbers}|"
                   f"NegHandl={handle_negation}|Ngram={ngram_range}|Feat={max_features}")

    return {
        "config": config_desc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "y_pred": y_pred
    }


def run_all_experiments(df_train, df_test):
    """Try every combination of cleaning choices (1,024 runs) and save the results to a CSV."""
    param_grid = {
        'apply_lowercase': [True, False],
        'remove_emojis': [True, False],
        'do_stemming': [True, False],
        'use_pos': [True, False],
        'remove_urls': [True, False],
        'remove_hashtags': [True, False],
        'remove_numbers': [True, False],
        'handle_negation': [True, False],
        'ngram_range': [(1, 1), (1, 2)],
        'max_features': [5000, 10000]
    }

    all_param_combinations = list(itertools.product(*param_grid.values()))
    total_combinations = len(all_param_combinations)
    results = []

    for idx, params in enumerate(all_param_combinations, start=1):
        print(f"\n### Iteration {idx}/{total_combinations} ###")
        params_dict = dict(zip(param_grid.keys(), params))
        result = run_experiment(df_train, df_test, **params_dict)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(dataProcessed_dir / "experiments_results.csv", index=False)

    # Best model by F1-score
    best_result = results_df.loc[results_df['f1_score'].idxmax()]
    print(f"\nBest result: {best_result['config']}, F1-score: {best_result['f1_score']}")

    return results_df
