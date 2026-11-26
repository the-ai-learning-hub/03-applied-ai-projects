"""Shared helpers for the tweet sentiment project (notebooks 02.0 to 02.4).

Basic cleaning, tokenization and the evaluation helpers (confusion matrix + report) used by
every notebook of the project.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)


def clean_text(text, stop_words=()):
    """Lowercase, drop a leading "RT USER" marker and links, and remove the given stopwords."""
    text = text.lower()
    # Remove RT/USER at the start of the text (e.g. "RT: USER USER")
    text = re.sub(r'^(rt[:\s]*(user\s*)+)', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r"http\S+|www.\S+|\burl\b", "", text)  # remove links and the URL placeholder
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text


def preprocess(text):
    """Tokenize a tweet with NLTK's TweetTokenizer (lowercase, no @handles, "soooo" -> "sooo")."""
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    return tokenizer.tokenize(text)


def plot_confusion_matrix(true_labels, pred_labels, model_name="Model", labels=("pos", "neg")):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(labels))
    sns.set(font_scale=1.2)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Positive", "Negative"],
                yticklabels=["Positive", "Negative"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix - {model_name}")
    plt.show()


def plot_confusion_matrix_glove(true_labels, pred_labels, label_encoder, model_name="GloVe"):
    """Confusion matrix when the labels were encoded as numbers with a LabelEncoder."""
    ordered_classes = label_encoder.transform(label_encoder.classes_)[::-1]  # reverse the label order
    label_text = label_encoder.classes_[::-1]

    cm = confusion_matrix(true_labels, pred_labels, labels=ordered_classes)

    sns.set(font_scale=1.2)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_text,
                yticklabels=label_text)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix - {model_name}")
    plt.show()


def print_classification_report(true_labels, pred_labels, model_name="Model", pos_label="pos"):
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, pos_label=pos_label)
    rec = recall_score(true_labels, pred_labels, pos_label=pos_label)
    f1 = f1_score(true_labels, pred_labels, pos_label=pos_label)

    print(f"\n=== Performance report: {model_name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\n--- Detailed report ---")
    print(classification_report(true_labels, pred_labels, target_names=["neg", "pos"]))


def print_classification_report_glove(true_labels, pred_labels, label_encoder, model_name="GloVe"):
    """Same report when the labels were encoded as numbers (weighted averages)."""
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    prec = precision_score(true_labels, pred_labels, average='weighted')
    rec = recall_score(true_labels, pred_labels, average='weighted')

    print(f"\n=== Performance report: {model_name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\n--- Detailed report ---")
    print(classification_report(true_labels, pred_labels, target_names=label_encoder.classes_))
