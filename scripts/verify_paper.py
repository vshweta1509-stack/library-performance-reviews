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
    "dropped after cleaning": (146, 0),
    "dropped, visit prompts only": (116, 0),
    "dropped, edit notices only": (21, 0),
    "dropped, other": (9, 0),
    "text reviews analysed": (2445, 0),
    "in scope NLI": (986, 0),
    "in scope CPL": (490, 0),
    "in scope ASM": (687, 0),
    "in scope DPL": (282, 0),
    "very short %": (17.7, 0.05),
    "truncated CPL+ASM+DPL": (153, 0),
    # Section 4.1, validation
    "TextBlob-VADER agreement %": (84.2, 0.05),
    "kappa": (0.579, 0.001),
    "lowest per-library agreement %": (84.0, 0.05),
    "highest per-library agreement %": (85.1, 0.05),
    "rated reviews with polar label": (2040, 0),
    "rating convergence C": (0.90, 0.005),
    "Spearman TextBlob": (0.19, 0.005),
    "Spearman VADER": (0.22, 0.005),
    "classifier accuracy %": (86.5, 0.05),
    "classifier negative F1": (0.51, 0.005),
    # Section 4.1, overall
    "positive % overall": (78.9, 0.05),
    "neutral % overall": (16.6, 0.05),
    "negative % overall": (4.5, 0.05),
    "sentiment chi2": (8.92, 0.01),
    "sentiment p": (0.18, 0.005),
    "Cramers V": (0.04, 0.005),
    "Kruskal-Wallis H": (55.18, 0.01),
    "low-star chi2": (34.96, 0.01),
    "mean rating all": (4.49, 0.005),
    # Section 4.2, dimensions
    "any dimension %": (69.0, 0.05),
    "importance LP %": (55.9, 0.05),
    "importance IC %": (32.0, 0.05),
    "importance DF %": (10.0, 0.05),
    "importance AS %": (7.8, 0.05),
    "NSS AS TextBlob": (48.4, 0.05),
    "NSS AS VADER": (46.3, 0.05),
    "mean rating AS praised": (4.19, 0.005),
    "mean rating AS criticised": (1.70, 0.005),
    "AS gap": (2.50, 0.005),
    "LP gap": (1.48, 0.005),
    "IC gap": (1.19, 0.005),
    "DF gap": (0.71, 0.005),
    "derived importance AS": (0.34, 0.005),
    "derived importance LP": (0.22, 0.005),
    "derived importance IC": (0.15, 0.005),
    "derived importance DF": (0.04, 0.005),
    "stated importance cut-off": (26.2, 0.05),
    "NSS cut-off": (65.4, 0.05),
    "derived importance cut-off": (0.200, 0.0005),
    "pairs used for derived cut-off": (15, 0),
    # Section 4.3, depository
    "depository mentions": (245, 0),
    "depository %": (10.0, 0.05),
    "depository % NLI": (12.7, 0.05),
    "depository % DPL": (2.8, 0.05),
    "depository positive %": (89.0, 0.05),
    "other positive %": (77.8, 0.05),
    "depository chi2": (17.24, 0.01),
    "depository mean rating": (4.65, 0.005),
    "other mean rating": (4.46, 0.005),
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
    "max change excluding truncated": (1.3, 0.05),
    "max change excluding visit prompts": (0.3, 0.05),
    "max change excluding very short": (1.8, 0.05),
    "sensitivity p excluding truncated": (0.048, 0.0005),
    "sensitivity p excluding visit prompts": (0.16, 0.005),
    "sensitivity p excluding very short": (0.20, 0.005),
}


# Values from human coding (Sections 4.1 and 4.2); checked when validation_human.json exists,
# i.e. after scripts/validate_human.py has been run with the coding sheets.
PUBLISHED_HUMAN = {
    "sentiment validation reviews": (250, 0),
    "inter-coder agreement %": (82.4, 0.05),
    "inter-coder kappa": (0.40, 0.005),
    "gold negative reviews": (10, 0),
    "TextBlob accuracy %": (82.8, 0.05),
    "TextBlob kappa": (0.43, 0.005),
    "TextBlob macro-F1": (0.67, 0.005),
    "TextBlob positive F1": (0.90, 0.005),
    "TextBlob neutral F1": (0.40, 0.005),
    "TextBlob negatives found": (8, 0),
    "VADER accuracy %": (82.4, 0.05),
    "VADER kappa": (0.49, 0.005),
    "VADER macro-F1": (0.65, 0.005),
    "classifier vs gold accuracy %": (79.6, 0.05),
    "classifier vs gold kappa": (0.31, 0.005),
    "classifier vs gold macro-F1": (0.55, 0.005),
    "aspect validation reviews": (200, 0),
    "AS kappa": (0.86, 0.005),
    "AS precision": (0.82, 0.005),
    "AS recall": (0.97, 0.005),
    "IC kappa": (0.40, 0.005),
    "IC precision": (0.46, 0.005),
    "LP kappa": (0.31, 0.005),
    "LP recall": (0.68, 0.005),
    "DF kappa": (0.47, 0.005),
    "DF precision": (0.90, 0.005),
    "DF recall": (0.42, 0.005),
    "tone agreement lowest %": (67, 0.5),
    "tone agreement highest %": (80, 0.5),
}


def collect_human(output: Path) -> dict:
    """Read the human-validation values written by scripts/validate_human.py."""
    human = json.loads((output / "validation_human.json").read_text())
    s, a = human["sentiment"], human["aspects"]["dimensions"]
    tone = human["aspects"]["tone_agree_pct_range"]
    return {
        "sentiment validation reviews": s["n_in_scope"],
        "inter-coder agreement %": s["inter_coder"]["agreement_pct"],
        "inter-coder kappa": s["inter_coder"]["kappa"],
        "gold negative reviews": s["gold_counts"]["negative"],
        "TextBlob accuracy %": s["textblob"]["accuracy_pct"],
        "TextBlob kappa": s["textblob"]["kappa"],
        "TextBlob macro-F1": s["textblob"]["f1_macro"],
        "TextBlob positive F1": s["textblob"]["f1_by_class"]["positive"],
        "TextBlob neutral F1": s["textblob"]["f1_by_class"]["neutral"],
        "TextBlob negatives found": s["textblob"]["negative_found"],
        "VADER accuracy %": s["vader"]["accuracy_pct"],
        "VADER kappa": s["vader"]["kappa"],
        "VADER macro-F1": s["vader"]["f1_macro"],
        "classifier vs gold accuracy %": s["classifier"]["accuracy_pct"],
        "classifier vs gold kappa": s["classifier"]["kappa"],
        "classifier vs gold macro-F1": s["classifier"]["f1_macro"],
        "aspect validation reviews": human["aspects"]["n_in_scope"],
        "AS kappa": a["AS"]["kappa"],
        "AS precision": a["AS"]["precision"],
        "AS recall": a["AS"]["recall"],
        "IC kappa": a["IC"]["kappa"],
        "IC precision": a["IC"]["precision"],
        "LP kappa": a["LP"]["kappa"],
        "LP recall": a["LP"]["recall"],
        "DF kappa": a["DF"]["kappa"],
        "DF precision": a["DF"]["precision"],
        "DF recall": a["DF"]["recall"],
        "tone agreement lowest %": tone[0],
        "tone agreement highest %": tone[1],
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
        "dropped, edit notices only": sample["n_dropped_edit_notice_only"],
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
        "sensitivity p excluding truncated": sensitivity.loc["excluding truncated", "sentiment_chi2_p"],
        "sensitivity p excluding visit prompts": sensitivity.loc["excluding visit prompts", "sentiment_chi2_p"],
        "sensitivity p excluding very short": sensitivity.loc["excluding very short", "sentiment_chi2_p"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that the code reproduces the published values")
    parser.add_argument("--output", default=Path("outputs"), type=Path)
    args = parser.parse_args(argv)

    computed = collect(args.output)
    expected = dict(PUBLISHED)
    if (args.output / "validation_human.json").exists():
        computed.update(collect_human(args.output))
        expected.update(PUBLISHED_HUMAN)
    else:
        print("validation_human.json not found: human-validation values not checked")
    failures = []
    for name, (published, tolerance) in expected.items():
        value = float(computed[name])
        if abs(value - published) > tolerance:
            failures.append(f"{name}: paper {published}, computed {value}")
    print(f"{len(expected) - len(failures)} of {len(expected)} published values reproduced")
    for failure in failures:
        print("  MISMATCH", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
