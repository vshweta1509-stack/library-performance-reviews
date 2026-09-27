"""Check that this code reproduces the values published in the paper.

Run after scripts/run_analysis.py on the study data:

    python -m scripts.verify_paper --output outputs

Every published number is listed below with the tolerance used to compare it. The script
exits with status 1 if any check fails, so it can be used in continuous integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# (description, published value, tolerance). Values are as printed in the manuscript.
PUBLISHED = {
    # Section 3.2, sample
    "reviews collected": (5411, 0),
    "rated reviews": (5410, 0),
    "reviews with text": (2591, 0),
    "dropped after cleaning": (101, 0),
    "dropped, visit prompts only": (94, 0),
    "dropped, other": (7, 0),
    "text reviews analysed": (2490, 0),
    "in scope NLI": (986, 0),
    "in scope CPL": (504, 0),
    "in scope ASM": (714, 0),
    "in scope DPL": (286, 0),
    "very short %": (17.8, 0.05),
    "truncated CPL+ASM+DPL": (160, 0),
    # Section 4.1, validation
    "TextBlob-VADER agreement %": (84.2, 0.05),
    "kappa": (0.585, 0.001),
    "lowest per-library agreement %": (83.6, 0.05),
    "highest per-library agreement %": (85.3, 0.05),
    "rated reviews with polar label": (2052, 0),
    "rating convergence C": (0.90, 0.005),
    "Spearman TextBlob": (0.19, 0.005),
    "Spearman VADER": (0.22, 0.005),
    "classifier accuracy %": (86.9, 0.05),
    "classifier negative F1": (0.41, 0.005),
    # Section 4.1, overall
    "positive % overall": (78.0, 0.05),
    "neutral % overall": (17.6, 0.05),
    "negative % overall": (4.4, 0.05),
    "sentiment chi2": (6.78, 0.01),
    "sentiment p": (0.34, 0.005),
    "Cramers V": (0.04, 0.005),
    "Kruskal-Wallis H": (55.18, 0.01),
    "low-star chi2": (34.96, 0.01),
    "mean rating all": (4.49, 0.005),
    # Section 4.2, dimensions
    "any dimension %": (67.9, 0.05),
    "importance LP %": (55.1, 0.05),
    "importance IC %": (31.4, 0.05),
    "importance DF %": (9.8, 0.05),
    "importance AS %": (7.6, 0.05),
    "NSS AS TextBlob": (48.4, 0.05),
    "NSS AS VADER": (46.3, 0.05),
    "mean rating AS praised": (4.19, 0.005),
    "mean rating AS criticised": (1.70, 0.005),
    "AS gap": (2.50, 0.005),
    "LP gap": (1.48, 0.005),
    "IC gap": (1.19, 0.005),
    "DF gap": (0.71, 0.005),
    "derived importance AS": (0.33, 0.005),
    "derived importance LP": (0.22, 0.005),
    "derived importance IC": (0.15, 0.005),
    "derived importance DF": (0.04, 0.005),
    "stated importance cut-off": (25.8, 0.05),
    "NSS cut-off": (65.4, 0.05),
    "derived importance cut-off": (0.196, 0.0005),
    "pairs used for derived cut-off": (15, 0),
    # Section 4.3, depository
    "depository mentions": (245, 0),
    "depository %": (9.8, 0.05),
    "depository % NLI": (12.7, 0.05),
    "depository % DPL": (2.8, 0.05),
    "depository positive %": (89.0, 0.05),
    "other positive %": (76.8, 0.05),
    "depository chi2": (19.97, 0.01),
    "depository mean rating": (4.65, 0.005),
    "other mean rating": (4.45, 0.005),
    "depository criticised": (13, 0),
    # Section 4.4, over time
    "NLI low-star % before": (5.0, 0.05),
    "NLI low-star % after": (8.9, 0.05),
    "NLI n before": (1497, 0),
    "NLI n after": (460, 0),
    "NLI chi2": (8.93, 0.01),
    "NLI p": (0.003, 0.0005),
    "NLI positive % before": (76.4, 0.05),
    "NLI positive % after": (76.4, 0.05),
    "NLI staff importance before %": (7.6, 0.05),
    "NLI staff importance after %": (13.8, 0.05),
    "NLI staff NSS before": (55.4, 0.05),
    "NLI staff NSS after": (29.0, 0.05),
    "NLI staff mentions before": (56, 0),
    "NLI staff mentions after": (31, 0),
    "CPL p": (0.026, 0.0005),
    "ASM p": (0.97, 0.005),
    "DPL p": (0.24, 0.005),
    # Section 4.5, sensitivity
    "max change excluding truncated": (1.2, 0.05),
    "max change excluding visit prompts": (0.2, 0.05),
    "max change excluding very short": (2.2, 0.05),
}


def collect(output: Path) -> dict:
    """Read the computed values that correspond to each published number."""
    results = json.loads((output / "results.json").read_text())
    dimensions = pd.read_csv(output / "table2_dimensions_ipa.csv")
    asymmetry = pd.read_csv(output / "table3_asymmetry.csv")
    temporal = pd.read_csv(output / "table4_temporal.csv").set_index("library")
    sensitivity = pd.read_csv(output / "table5_sensitivity.csv").set_index("subset")
    table1 = pd.read_csv(output / "table1_sample_sentiment.csv").set_index("library")
    sample = results["sample"]
    validation = results["validation"]
    agreement = validation["lexicon_agreement"]
    pooled = dimensions[dimensions["library"] == "ALL"].set_index("dimension")
    gaps = asymmetry.set_index("dimension")
    staff = results["staff_shift_lowest_rated"]["NLI"]
    truncated = sample["truncated_in_scope_by_library"]

    return {
        "reviews collected": sample["n_reviews"],
        "rated reviews": sample["n_rated"],
        "reviews with text": sample["n_with_text"],
        "dropped after cleaning": sample["n_dropped_after_cleaning"],
        "dropped, visit prompts only": sample["n_dropped_visit_prompt_only"],
        "dropped, other": sample["n_dropped_other"],
        "text reviews analysed": sample["n_in_scope"],
        "in scope NLI": sample["in_scope_by_library"]["NLI"],
        "in scope CPL": sample["in_scope_by_library"]["CPL"],
        "in scope ASM": sample["in_scope_by_library"]["ASM"],
        "in scope DPL": sample["in_scope_by_library"]["DPL"],
        "very short %": sample["very_short_pct"],
        "truncated CPL+ASM+DPL": truncated["CPL"] + truncated["ASM"] + truncated["DPL"],
        "TextBlob-VADER agreement %": agreement["overall"]["agreement_pct"],
        "kappa": agreement["overall"]["kappa"],
        "lowest per-library agreement %": min(v["agreement_pct"] for v in agreement["per_library"].values()),
        "highest per-library agreement %": max(v["agreement_pct"] for v in agreement["per_library"].values()),
        "rated reviews with polar label": validation["rating_convergence"]["n"],
        "rating convergence C": validation["rating_convergence"]["C"],
        "Spearman TextBlob": validation["spearman"]["textblob_rho"],
        "Spearman VADER": validation["spearman"]["vader_rho"],
        "classifier accuracy %": validation["classifier"]["accuracy_pct"],
        "classifier negative F1": validation["classifier"]["f1_negative"],
        "positive % overall": table1.loc["All", "positive_pct"],
        "neutral % overall": table1.loc["All", "neutral_pct"],
        "negative % overall": table1.loc["All", "negative_pct"],
        "sentiment chi2": results["sentiment_test"]["chi2"],
        "sentiment p": results["sentiment_test"]["p"],
        "Cramers V": results["sentiment_test"]["cramers_v"],
        "Kruskal-Wallis H": results["rating_tests"]["kruskal_h"],
        "low-star chi2": results["rating_tests"]["low_star_chi2"],
        "mean rating all": table1.loc["All", "mean_rating"],
        "any dimension %": results["any_dimension_pct"],
        "importance LP %": pooled.loc["LP", "stated_importance_pct"],
        "importance IC %": pooled.loc["IC", "stated_importance_pct"],
        "importance DF %": pooled.loc["DF", "stated_importance_pct"],
        "importance AS %": pooled.loc["AS", "stated_importance_pct"],
        "NSS AS TextBlob": pooled.loc["AS", "nss"],
        "NSS AS VADER": pooled.loc["AS", "nss_vader"],
        "mean rating AS praised": gaps.loc["AS", "mean_rating_praised"],
        "mean rating AS criticised": gaps.loc["AS", "mean_rating_criticised"],
        "AS gap": gaps.loc["AS", "gap_stars"],
        "LP gap": gaps.loc["LP", "gap_stars"],
        "IC gap": gaps.loc["IC", "gap_stars"],
        "DF gap": gaps.loc["DF", "gap_stars"],
        "derived importance AS": abs(pooled.loc["AS", "derived_importance"]),
        "derived importance LP": abs(pooled.loc["LP", "derived_importance"]),
        "derived importance IC": abs(pooled.loc["IC", "derived_importance"]),
        "derived importance DF": abs(pooled.loc["DF", "derived_importance"]),
        "stated importance cut-off": results["ipa_cutoffs"]["stated_importance_mean"],
        "NSS cut-off": results["ipa_cutoffs"]["nss_mean"],
        "derived importance cut-off": results["ipa_cutoffs"]["derived_importance_mean"],
        "pairs used for derived cut-off": results["ipa_cutoffs"]["n_pairs_derived"],
        "depository mentions": results["depository"]["n_mentions"],
        "depository %": results["depository"]["share_pct"],
        "depository % NLI": results["depository"]["share_by_library_pct"]["NLI"],
        "depository % DPL": results["depository"]["share_by_library_pct"]["DPL"],
        "depository positive %": results["depository"]["positive_pct_mentioning"],
        "other positive %": results["depository"]["positive_pct_other"],
        "depository chi2": results["depository"]["chi2"],
        "depository mean rating": results["depository"]["mean_rating_mentioning"],
        "other mean rating": results["depository"]["mean_rating_other"],
        "depository criticised": results["depository"]["n_criticised"],
        "NLI low-star % before": temporal.loc["NLI", "low_star_pct_before"],
        "NLI low-star % after": temporal.loc["NLI", "low_star_pct_after"],
        "NLI n before": temporal.loc["NLI", "n_before"],
        "NLI n after": temporal.loc["NLI", "n_after"],
        "NLI chi2": temporal.loc["NLI", "chi2_yates"],
        "NLI p": temporal.loc["NLI", "p_yates"],
        "NLI positive % before": temporal.loc["NLI", "positive_pct_before"],
        "NLI positive % after": temporal.loc["NLI", "positive_pct_after"],
        "NLI staff importance before %": staff["before"]["importance_pct"],
        "NLI staff importance after %": staff["after"]["importance_pct"],
        "NLI staff NSS before": staff["before"]["nss"],
        "NLI staff NSS after": staff["after"]["nss"],
        "NLI staff mentions before": staff["before"]["n"],
        "NLI staff mentions after": staff["after"]["n"],
        "CPL p": temporal.loc["CPL", "p_yates"],
        "ASM p": temporal.loc["ASM", "p_yates"],
        "DPL p": temporal.loc["DPL", "p_yates"],
        "max change excluding truncated": sensitivity.loc["excluding truncated", "max_change_vs_base"],
        "max change excluding visit prompts": sensitivity.loc["excluding visit prompts", "max_change_vs_base"],
        "max change excluding very short": sensitivity.loc["excluding very short", "max_change_vs_base"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that the code reproduces the published values")
    parser.add_argument("--output", default=Path("outputs"), type=Path)
    args = parser.parse_args(argv)

    computed = collect(args.output)
    failures = []
    for name, (published, tolerance) in PUBLISHED.items():
        value = float(computed[name])
        if abs(value - published) > tolerance:
            failures.append(f"{name}: paper {published}, computed {value}")
    print(f"{len(PUBLISHED) - len(failures)} of {len(PUBLISHED)} published values reproduced")
    for failure in failures:
        print("  MISMATCH", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
