"""Reproduce the published analysis from the shareable derived dataset (no review text needed).

    python -m scripts.run_from_derived --output outputs_derived
    python -m scripts.verify_paper --output outputs_derived

Reads data/derived/reviews_derived.csv and the human coding in data/derived/, writes the same
tables, figures, results.json and validation_human.json as scripts/run_analysis.py followed
by scripts/validate_human.py. The two classifier checks need review text and are skipped;
verify_paper reports them as not checked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from libperf import config
from libperf.derived import load_derived
from scripts.run_analysis import analyse
from scripts.validate_human import aspect_validation, sentiment_validation

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Reproduce the analysis from the derived dataset")
    parser.add_argument("--derived", default=DERIVED / "reviews_derived.csv", type=Path)
    parser.add_argument("--sentiment-codes", default=DERIVED / "sentiment_codes.csv", type=Path)
    parser.add_argument("--aspect-codes", default=DERIVED / "aspect_codes.csv", type=Path)
    parser.add_argument("--output", default=Path("outputs_derived"), type=Path)
    args = parser.parse_args(argv)

    prepared, scored = load_derived(args.derived)
    results = analyse(prepared, scored, args.output, classifier=False)
    human = {
        "sentiment": sentiment_validation(
            scored, pd.read_csv(args.sentiment_codes, dtype={"review_id": str}), config.CLASSIFIER_SEED
        ),
        "aspects": aspect_validation(scored, pd.read_csv(args.aspect_codes, dtype={"review_id": str})),
    }
    with open(args.output / "validation_human.json", "w", encoding="utf-8") as handle:
        json.dump(human, handle, indent=1)
    print(f"Analysed {results['sample']['n_in_scope']} text reviews from the derived dataset; "
          f"results written to {args.output}")
    return results


if __name__ == "__main__":
    main()
