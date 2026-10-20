"""Run the full analysis and write every table, figure and statistic reported in the paper.

Usage:
    python -m scripts.run_analysis --input data/sample_reviews.csv --output outputs
    python -m scripts.run_analysis --input path/to/four_libraries_combined.xlsx --output outputs

Outputs (in --output):
    results.json                all statistics reported in the text
    table1_sample_sentiment.csv Table 1
    table2_dimensions_ipa.csv   Table 2
    table3_asymmetry.csv        rating gaps quoted in Section 4.2
    table4_temporal.csv         before/after comparison, Section 4.4
    table5_sensitivity.csv      Section 4.5
    scored_reviews.csv          one row per analysed review with all labels
    figure1_ipa.png             Figure 1
    figure2_temporal.png        supporting figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from libperf import config, figures, indicators, sentiment
from libperf.cleaning import prepare_reviews
from libperf.derived import dropped_reason
from libperf.sentiment import NEGATIVE, NEUTRAL, POSITIVE


def load_reviews(path: Path) -> pd.DataFrame:
    """Read a review table from .xlsx or .csv and check the required columns."""
    reader = pd.read_excel if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"} else pd.read_csv
    data = reader(path)
    missing = [column for column in config.REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"input is missing required column(s): {', '.join(missing)}")
    unknown = set(data["library"].dropna().unique()) - set(config.LIBRARY_ORDER)
    if unknown:
        raise ValueError(f"unknown library name(s) in input: {sorted(unknown)}; edit libperf/config.py")
    return data


def describe_sample(prepared: pd.DataFrame, scored: pd.DataFrame) -> dict:
    """Counts reported in Section 3.2."""
    reasons = prepared["dropped_reason"] if "dropped_reason" in prepared.columns else dropped_reason(prepared)
    dropped = prepared["has_text"] & ~prepared["in_scope"]
    return {
        "n_reviews": int(len(prepared)),
        "n_rated": int(prepared["rating"].notna().sum()),
        "n_with_text": int(prepared["has_text"].sum()),
        "n_dropped_after_cleaning": int(dropped.sum()),
        "n_dropped_visit_prompt_only": int((reasons == "visit_prompt_only").sum()),
        "n_dropped_edit_notice_only": int((reasons == "edit_notice_only").sum()),
        "n_dropped_other": int((reasons == "other").sum()),
        "n_in_scope": int(len(scored)),
        "in_scope_by_library": scored["lib"].value_counts().reindex(config.LIBRARIES).to_dict(),
        "very_short_pct": round(float(scored["very_short"].mean() * 100), 3),
        "truncated_in_scope_by_library": scored.groupby("lib", observed=True)["flag_truncated"].sum()
        .reindex(config.LIBRARIES).to_dict(),
        "visit_prompt_in_scope": int(scored["flag_visit_prompt"].sum()),
        "edit_notice_in_scope": int(scored["flag_edited_notice"].sum()),
    }


def table1(prepared: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Table 1: reviews, ratings and review-level sentiment by library."""
    rated = prepared[prepared["rating"].notna()]
    counts, _ = indicators.sentiment_by_library(scored, config.LIBRARIES)
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    rows = []
    for library in config.LIBRARIES:
        library_rated = rated[rated["lib"] == library]
        rows.append(
            {
                "library": library,
                "reviews": int((prepared["lib"] == library).sum()),
                "text_reviews": int((scored["lib"] == library).sum()),
                "mean_rating": library_rated["rating"].mean(),
                "low_star_pct": (library_rated["rating"] <= 2).mean() * 100,
                "positive_pct": shares.loc[library, POSITIVE],
                "neutral_pct": shares.loc[library, NEUTRAL],
                "negative_pct": shares.loc[library, NEGATIVE],
            }
        )
    overall_shares = scored["tb"].value_counts(normalize=True) * 100
    rows.append(
        {
            "library": "All",
            "reviews": int(len(prepared)),
            "text_reviews": int(len(scored)),
            "mean_rating": rated["rating"].mean(),
            "low_star_pct": (rated["rating"] <= 2).mean() * 100,
            "positive_pct": overall_shares[POSITIVE],
            "neutral_pct": overall_shares[NEUTRAL],
            "negative_pct": overall_shares[NEGATIVE],
        }
    )
    return pd.DataFrame(rows).round(3)


def analyse(
    prepared: pd.DataFrame,
    scored: pd.DataFrame,
    output: Path,
    seed: int = config.BOOTSTRAP_SEED,
    bootstrap_draws: int = config.BOOTSTRAP_DRAWS,
    classifier: bool = True,
) -> dict:
    """Compute every table, figure and statistic from cleaned (prepared) and scored reviews.

    Shared by this script (which starts from review text) and scripts/run_from_derived.py
    (which starts from the shareable derived table and cannot run the text classifier).
    """
    output.mkdir(parents=True, exist_ok=True)
    rated = prepared[prepared["rating"].notna()]

    results: dict = {"sample": describe_sample(prepared, scored)}
    results["validation"] = {
        "lexicon_agreement": sentiment.lexicon_agreement(scored),
        "rating_convergence": sentiment.rating_convergence(scored),
        "spearman": indicators.spearman_with_rating(scored),
    }
    if classifier:
        results["validation"]["classifier"] = sentiment.classifier_check(scored, config.CLASSIFIER_SEED)

    counts, sentiment_test = indicators.sentiment_by_library(scored, config.LIBRARIES)
    results["sentiment_by_library_counts"] = counts.to_dict("index")
    results["sentiment_test"] = sentiment_test
    results["rating_tests"] = indicators.rating_tests(rated, config.LIBRARIES)

    dimensions = indicators.dimension_table(
        scored, config.LIBRARIES, config.DIMENSIONS, seed, bootstrap_draws, config.SMALL_SAMPLE
    )
    dimensions, cutoffs = indicators.add_ipa_quadrants(dimensions)
    results["ipa_cutoffs"] = cutoffs
    results["any_dimension_pct"] = round(float((scored["n_dimensions"] > 0).mean() * 100), 3)
    results["depository"] = indicators.depository_comparison(scored, config.LIBRARIES)

    asymmetry = indicators.asymmetry_table(scored, config.DIMENSIONS)
    temporal = indicators.temporal_comparison(rated, scored, config.LIBRARIES, config.SPLIT_YEAR, config.FIRST_YEAR)
    results["staff_shift_lowest_rated"] = {
        library: indicators.dimension_shift(scored, library, "AS", config.SPLIT_YEAR, config.FIRST_YEAR)
        for library in config.LIBRARIES
    }
    sensitivity = indicators.sensitivity(scored, config.LIBRARIES)

    table1(prepared, scored).to_csv(output / "table1_sample_sentiment.csv", index=False)
    dimensions.round(3).to_csv(output / "table2_dimensions_ipa.csv", index=False)
    asymmetry.round(4).to_csv(output / "table3_asymmetry.csv", index=False)
    temporal.to_csv(output / "table4_temporal.csv", index=False)
    sensitivity.round(3).to_csv(output / "table5_sensitivity.csv", index=False)

    keep = [
        "review_id", "library", "lib", "rating", "date", "date_exact", "year", "text", "n_words",
        "flag_visit_prompt", "flag_truncated", "flag_translated", "very_short",
        "tb_polarity", "tb", "vader_compound", "vader", "n_dimensions",
    ] + [f"{prefix}_{d}" for d in config.DIMENSIONS for prefix in ("mentions", "sentiment", "vader")]
    scored[[column for column in keep if column in scored.columns]].to_csv(
        output / "scored_reviews.csv", index=False
    )

    figures.ipa_figure(dimensions, cutoffs, config.LIBRARY_NAMES, output / "figure1_ipa.png")
    figures.temporal_figure(rated, config.LIBRARIES, config.LIBRARY_NAMES, output / "figure2_temporal.png")

    with open(output / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, default=str)
    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Measuring library performance from Google Maps reviews")
    parser.add_argument("--input", required=True, type=Path, help="review table (.xlsx or .csv)")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="output directory")
    parser.add_argument("--bootstrap-draws", default=config.BOOTSTRAP_DRAWS, type=int)
    parser.add_argument("--seed", default=config.BOOTSTRAP_SEED, type=int)
    parser.add_argument("--skip-classifier", action="store_true", help="skip the supervised check")
    args = parser.parse_args(argv)

    reviews = load_reviews(args.input)
    prepared = prepare_reviews(
        reviews, config.COLLECTION_DATE, config.LIBRARY_ORDER, config.LIBRARY_CODES, config.SHORT_REVIEW_WORDS
    )
    dictionary = sentiment.load_dictionary()
    scored = sentiment.score_reviews(prepared[prepared["in_scope"]], dictionary)
    results = analyse(prepared, scored, args.output, args.seed, args.bootstrap_draws, not args.skip_classifier)
    print(f"Analysed {results['sample']['n_in_scope']} text reviews; results written to {args.output}")
    return results


if __name__ == "__main__":
    main()
