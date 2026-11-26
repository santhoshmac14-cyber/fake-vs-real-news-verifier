"""Train and save the TF-IDF + logistic regression news classifier."""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "news_labeled.csv"
MODEL_FILE = PROJECT_DIR / "models" / "fake_real_model.joblib"


def read_articles() -> tuple[list[str], list[str]]:
    """Read the prepared CSV into separate text and label lists."""
    texts = []
    labels = []

    with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])

    return texts, labels


def main() -> None:
    texts, labels = read_articles()

    # Keep 20% aside for checking, with the same Fake/Real balance in both sets.
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # A pipeline first turns text into TF-IDF numbers, then classifies those numbers.
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    stop_words="english",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000)),
        ]
    )

    print(f"Training on {len(train_texts):,} articles...")
    model.fit(train_texts, train_labels)

    predictions = model.predict(test_texts)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test accuracy: {accuracy:.1%}")
    print("\nDetailed results:")
    print(classification_report(test_labels, predictions, zero_division=0))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved trained model to: {MODEL_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
