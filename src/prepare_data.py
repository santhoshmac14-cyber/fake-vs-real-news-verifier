"""Combine the Kaggle Fake.csv and True.csv files into labeled text data."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "news_labeled.csv"


def write_labeled_rows(writer: csv.DictWriter, filename: str, label: str) -> int:
    source = RAW_DIR / filename
    written = 0

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"title", "text"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{filename} must contain the columns: "
                f"{', '.join(sorted(required_columns))}"
            )

        for row in reader:
            title = (row.get("title") or "").strip()
            body = (row.get("text") or "").strip()
            combined_text = " ".join(part for part in (title, body) if part)

            # Skip only rows with neither a title nor a body.
            if not combined_text:
                continue

            writer.writerow({"text": combined_text, "label": label})
            written += 1

    return written


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        counts["Fake"] = write_labeled_rows(writer, "Fake.csv", "Fake")
        counts["Real"] = write_labeled_rows(writer, "True.csv", "Real")

    print(f"Prepared {sum(counts.values()):,} articles: "
          f"{counts['Fake']:,} Fake, {counts['Real']:,} Real")
    print(f"Saved to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
