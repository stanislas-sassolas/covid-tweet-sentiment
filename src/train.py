"""Train and compare sentiment classifiers on the COVID-19 tweets dataset.

Models: a TF-IDF + logistic regression baseline and four Keras architectures
(LSTM, BiLSTM, GRU, CNN+LSTM). Early stopping uses a validation split carved
out of the training set, so the official test set is only used once, at the end.

Usage:
    python src/train.py                 # all models, 5 classes
    python src/train.py --classes 3     # merge "Extremely" classes into Positive/Negative
"""

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"

MAX_WORDS, MAX_LEN, EMBEDDING_DIM = 10_000, 64, 128
BATCH_SIZE, EPOCHS, SEED = 64, 15, 42

CLASS_ORDER_5 = ["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]
CLASS_ORDER_3 = ["Negative", "Neutral", "Positive"]


def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", str(text))
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def load(n_classes: int):
    train = pd.read_csv(DATA / "Corona_NLP_train.csv", encoding="latin-1")
    test = pd.read_csv(DATA / "Corona_NLP_test.csv", encoding="latin-1")
    classes = CLASS_ORDER_5
    if n_classes == 3:
        merge = {"Extremely Negative": "Negative", "Extremely Positive": "Positive"}
        for df in (train, test):
            df["Sentiment"] = df["Sentiment"].replace(merge)
        classes = CLASS_ORDER_3
    for df in (train, test):
        df["text"] = df["OriginalTweet"].map(clean_tweet)
        df["label"] = df["Sentiment"].map({c: i for i, c in enumerate(classes)})
    return train[train.text.str.len() > 0], test[test.text.str.len() > 0], classes


def rnn_model(kind: str, n_classes: int):
    if kind == "LSTM":
        core = [layers.SpatialDropout1D(0.2), layers.LSTM(64, dropout=0.2)]
    elif kind == "BiLSTM":
        core = [layers.SpatialDropout1D(0.2), layers.Bidirectional(layers.LSTM(64, dropout=0.2))]
    elif kind == "GRU":
        core = [layers.SpatialDropout1D(0.2), layers.GRU(64, dropout=0.2)]
    elif kind == "CNN+LSTM":
        core = [layers.Conv1D(128, 5, activation="relu"), layers.MaxPooling1D(2),
                layers.LSTM(64, dropout=0.2)]
    else:
        raise ValueError(kind)
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,), dtype="int32"),
        layers.Embedding(MAX_WORDS, EMBEDDING_DIM, mask_zero=kind != "CNN+LSTM"),
        *core,
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name=kind.replace("+", "_"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, choices=[3, 5], default=5)
    parser.add_argument("--models", nargs="*", default=["LSTM", "BiLSTM", "GRU", "CNN+LSTM"])
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    train, test, classes = load(args.classes)
    tr, val = train_test_split(train, test_size=0.1, stratify=train.label, random_state=SEED)
    print(f"{args.classes} classes | train {len(tr)} | val {len(val)} | test {len(test)}")

    results, best = {}, None

    # Baseline: TF-IDF (uni+bigrams) + logistic regression
    t0 = time.time()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=4)
    clf.fit(tfidf.fit_transform(tr.text), tr.label)
    pred = clf.predict(tfidf.transform(test.text))
    results["TF-IDF + LogReg"] = {**scores(test.label, pred), "train_s": round(time.time() - t0)}
    best = ("TF-IDF + LogReg", pred)
    print("TF-IDF + LogReg", results["TF-IDF + LogReg"])

    vectorizer = layers.TextVectorization(max_tokens=MAX_WORDS, output_sequence_length=MAX_LEN)
    vectorizer.adapt(tr.text.values)

    (RESULTS / f"vocabulary_{args.classes}cls.txt").write_text("\n".join(vectorizer.get_vocabulary()), encoding="utf-8")
    x_tr, x_val, x_te = (vectorizer(d.text.values).numpy() for d in (tr, val, test))
    histories = {}
    for kind in args.models:
        t0 = time.time()
        model = rnn_model(kind, len(classes))
        hist = model.fit(
            x_tr, tr.label.values, validation_data=(x_val, val.label.values),
            batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                        restore_best_weights=True)])
        pred = model.predict(x_te, verbose=0).argmax(axis=1)
        results[kind] = {**scores(test.label, pred), "epochs": len(hist.history["loss"]),
                         "params": model.count_params(), "train_s": round(time.time() - t0)}
        histories[kind] = hist.history
        print(kind, results[kind])
        if results[kind]["macro_f1"] > results[best[0]]["macro_f1"]:
            best = (kind, pred)
            model.save(RESULTS / f"best_model_{args.classes}cls.keras")

    suffix = f"{args.classes}cls"
    name, pred = best
    report = classification_report(test.label, pred, target_names=classes, digits=3)
    (RESULTS / f"metrics_{suffix}.json").write_text(json.dumps(
        {"results": results, "best": name, "n_test": len(test)}, indent=2))
    (RESULTS / f"classification_report_{suffix}.txt").write_text(f"Best model: {name}\n\n{report}")
    print(pd.DataFrame(results).T.to_string())
    print(f"\nBest: {name}\n{report}")

    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(test.label, pred, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion matrix (row-normalised) - {name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(FIGURES / f"confusion_matrix_{suffix}.png", dpi=130)

    if histories:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for kind, h in histories.items():
            ax[0].plot(h["val_accuracy"], marker="o", label=kind)
            ax[1].plot(h["val_loss"], marker="o", label=kind)
        ax[0].set_title("Validation accuracy"); ax[1].set_title("Validation loss")
        for a in ax:
            a.set_xlabel("Epoch"); a.grid(alpha=0.3); a.legend()
        plt.tight_layout(); plt.savefig(FIGURES / f"training_curves_{suffix}.png", dpi=130)


if __name__ == "__main__":
    main()
