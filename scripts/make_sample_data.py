"""Create a small synthetic dataset so the pipeline can be run without the study data.

The text is generated from templates. It contains no real reviews, so the file can be
distributed freely; it exists to demonstrate the code, not to reproduce the findings.

    python -m scripts.make_sample_data --output data/sample_reviews.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LIBRARIES = [
    "National Library",
    "Connemara Public Library",
    "The Asiatic Society, Mumbai",
    "Delhi Public Library",
]
POSITIVE_TEMPLATES = [
    "A wonderful reading hall, very quiet and clean.",
    "The staff were helpful and polite throughout my visit.",
    "Excellent collection of books and newspapers.",
    "The rare books and manuscripts here are a treasure.",
    "Beautiful heritage building with a peaceful garden.",
    "Great place to study, comfortable seating and good lighting.",
    "The archives are well preserved and useful for research.",
]
NEGATIVE_TEMPLATES = [
    "The staff were rude and unhelpful when I asked for a book.",
    "Toilets were dirty and the reading room was noisy.",
    "Membership process is slow and the catalogue is out of date.",
    "Very poor maintenance, broken chairs everywhere.",
]
NEUTRAL_TEMPLATES = ["National Library, Kolkata.", "Open on weekdays.", "Library building."]
SHORT_TEMPLATES = ["Nice", "Good place", "Superb", "Ok"]
# Platform text the scrapers captured; included so the cleaning step can be demonstrated.
VISIT_PROMPTS = ["Weekday Wait time No wait Reservation recommended No", "Visited on Weekend Wait time 10-30 min"]
TRUNCATION = " … More"
TRANSLATION = " Translated by Google ・ See original (Hindi)"


def build(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        library = LIBRARIES[i % len(LIBRARIES)]
        draw = rng.random()
        if draw < 0.55:
            text, rating = rng.choice(POSITIVE_TEMPLATES), int(rng.choice([4, 5, 5]))
        elif draw < 0.70:
            text, rating = rng.choice(SHORT_TEMPLATES), 5
        elif draw < 0.82:
            text, rating = rng.choice(NEGATIVE_TEMPLATES), int(rng.choice([1, 2]))
        elif draw < 0.90:
            text, rating = rng.choice(NEUTRAL_TEMPLATES), 3
        elif draw < 0.95:
            text, rating = rng.choice(VISIT_PROMPTS), int(rng.choice([4, 5]))  # platform text only
        else:
            text, rating = str(rng.choice(POSITIVE_TEMPLATES)) + TRUNCATION, 5
        if rng.random() < 0.05:
            text = str(text) + TRANSLATION
        years_ago = int(rng.integers(0, 8))
        rows.append(
            {
                "review_id": f"sample_{i:04d}",
                "library": library,
                "rating": rating,
                "review_text": text,
                "date_relative": "a year ago" if years_ago == 1 else f"{max(years_ago, 1)} years ago",
                "date_absolute": pd.NaT,
                "review_text_translated": None,
                "original_language": None,
                "source": "synthetic",
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create the synthetic demo dataset")
    parser.add_argument("--output", default=Path("data/sample_reviews.csv"), type=Path)
    parser.add_argument("--n", default=200, type=int)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.n, args.seed).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
