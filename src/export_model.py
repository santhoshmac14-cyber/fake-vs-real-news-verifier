"""Export the trained scikit-learn model for the browser-based app."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_DIR / "models" / "fake_real_model.joblib"
OUTPUT_FILE = PROJECT_DIR / "docs" / "model.json"


def main() -> None:
    model = joblib.load(MODEL_FILE)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    names = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]

    features = [
        [str(term), round(float(idf), 8), round(float(weight), 8)]
        for term, idf, weight in zip(names, vectorizer.idf_, weights)
    ]
    payload = {
        "classes": [str(label) for label in classifier.classes_],
        "intercept": float(classifier.intercept_[0]),
        "features": features,
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "settings": {
            "ngram_range": list(vectorizer.ngram_range),
            "token_pattern": vectorizer.token_pattern,
            "normalization": vectorizer.norm,
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(features):,} TF-IDF features to {OUTPUT_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
