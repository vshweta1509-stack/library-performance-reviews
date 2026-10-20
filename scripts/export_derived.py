"""Write the shareable derived dataset from the full review table (authors only).

    python -m scripts.export_derived --input four_libraries_combined.xlsx \
        --sentiment-codes sentiment_codes.csv --aspect-codes aspect_codes.csv --output data/derived

Writes to --output:
    reviews_derived.csv     one row per collected review, no text, names or platform ids
    sentiment_codes.csv     human sentiment coding keyed by study id
    aspect_codes.csv        human aspect coding keyed by study id
and, to --private-key (default: study_id_key_PRIVATE.csv in the current directory, never
to be published), the map from Google review id to study id.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from libperf import config, sentiment
from libperf.cleaning import prepare_reviews
from libperf.derived import build_derived, recode_human
from scripts.run_analysis import load_reviews


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the shareable derived dataset")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sentiment-codes", required=True, type=Path)
    parser.add_argument("--aspect-codes", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/derived"), type=Path)
    parser.add_argument("--private-key", default=Path("study_id_key_PRIVATE.csv"), type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    prepared = prepare_reviews(
        load_reviews(args.input), config.COLLECTION_DATE, config.LIBRARY_ORDER, config.LIBRARY_CODES,
        config.SHORT_REVIEW_WORDS,
    )
    scored = sentiment.score_reviews(prepared[prepared["in_scope"]], sentiment.load_dictionary())
    table, key = build_derived(prepared, scored)
    table.to_csv(args.output / "reviews_derived.csv", index=False)
    pd.Series(key, name="study_id").rename_axis("platform_review_id").to_csv(args.private_key)

    for name, path in (("sentiment_codes.csv", args.sentiment_codes), ("aspect_codes.csv", args.aspect_codes)):
        recode_human(pd.read_csv(path, dtype={"review_id": str}), key).to_csv(args.output / name, index=False)
    print(f"Wrote {len(table)} rows to {args.output / 'reviews_derived.csv'}; private key at {args.private_key}")


if __name__ == "__main__":
    main()
