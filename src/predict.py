"""Predict the sentiment of new tweets with the trained model.

Usage:
    python src/predict.py "Supermarket shelves are empty again, this is getting scary"
    python src/predict.py --classes 3 "Thank you to all the nurses working tonight!"
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import CLASS_ORDER_3, CLASS_ORDER_5, MAX_LEN, clean_tweet

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--classes", type=int, choices=[3, 5], default=5)
    args = parser.parse_args()

    vocab = (RESULTS / f"vocabulary_{args.classes}cls.txt").read_text(encoding="utf-8").split("\n")
    vectorizer = tf.keras.layers.TextVectorization(vocabulary=vocab[2:], output_sequence_length=MAX_LEN)
    model = tf.keras.models.load_model(RESULTS / f"best_model_{args.classes}cls.keras")
    classes = CLASS_ORDER_5 if args.classes == 5 else CLASS_ORDER_3

    probs = model.predict(vectorizer([clean_tweet(t) for t in args.texts]).numpy(), verbose=0)
    for text, p in zip(args.texts, probs):
        top = int(np.argmax(p))
        print(f"{classes[top]:<19} {p[top]:.0%}  {text}")


if __name__ == "__main__":
    main()
