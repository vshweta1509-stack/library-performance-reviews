"""Performance indicators, importance-performance analysis and statistical tests.

Reference: Sections 3.4 and 3.5 and equations (4)-(8) of the manuscript.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr

from .sentiment import NEGATIVE, NEUTRAL, POSITIVE

def safe_chi2(table) -> tuple[float, float, int]:
    """Chi-square test that returns NaN instead of raising when the table is too sparse.

    Small datasets can produce a row or column of zeros, for which the test is undefined.
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), float("nan"), 0
    keep_rows = table.sum(axis=1) > 0
    keep_cols = table.sum(axis=0) > 0
    table = table[np.ix_(keep_rows, keep_cols)]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), float("nan"), 0
    chi2, p_value, dof, _ = chi2_contingency(table)
    return float(chi2), float(p_value), int(dof)


QUADRANTS = {
    (True, True): "Keep up the good work",
    (True, False): "Concentrate here",
    (False, True): "Possible overkill",
    (False, False): "Low priority",
}


def net_sentiment_score(labels) -> float:
    """Equation (4): percentage positive minus percentage negative."""
    labels = pd.Series(list(labels))
    if labels.empty:
        return float("nan")
    return float(((labels == POSITIVE).mean() - (labels == NEGATIVE).mean()) * 100)


def stated_importance(n_mentions: int, n_reviews: int) -> float:
    """Equation (5): share of a library's reviews that mention the dimension."""
    return float(n_mentions / n_reviews * 100)


def derived_importance(negative_mention: pd.Series, rating: pd.Series) -> float:
    """Equation (6): point-biserial correlation between negative mention and star rating.

    Returns the signed correlation; the manuscript reports its absolute value. Undefined
    (NaN) when the dimension attracts at most one negative mention.
    """
    negative_mention = pd.Series(negative_mention).astype(float)
    rating = pd.Series(rating).astype(float)
    rating = rating.fillna(rating.mean())
    if negative_mention.sum() <= 1:
        return float("nan")
    share = negative_mention.mean()
    spread = rating.std(ddof=0)
    if spread == 0:
        return float("nan")
    difference = rating[negative_mention == 1].mean() - rating[negative_mention == 0].mean()
    return float(difference / spread * np.sqrt(share * (1 - share)))


def bootstrap_interval(labels, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap interval for the Net Sentiment Score."""
    values = np.asarray(list(labels), dtype=object)
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = [net_sentiment_score(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def dimension_table(
    scored: pd.DataFrame,
    libraries: list[str],
    dimensions: list[str],
    seed: int = 2026,
    n_boot: int = 2000,
    small_n: int = 30,
) -> pd.DataFrame:
    """Per library and dimension: mentions, importance, NSS with interval, derived importance.

    A final block of pooled rows (library "ALL") is appended. The random generator is created
    once and consumed in a fixed order, so intervals are reproducible.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for library in [*libraries, "ALL"]:
        group = scored if library == "ALL" else scored[scored["lib"] == library]
        for dimension in dimensions:
            mentioned = group[group[f"mentions_{dimension}"] == 1]
            labels = mentioned[f"sentiment_{dimension}"].values
            low, high = bootstrap_interval(labels, rng, n_boot)
            correlation = derived_importance(
                (group[f"sentiment_{dimension}"] == NEGATIVE).astype(int), group["rating"]
            )
            rows.append(
                {
                    "library": library,
                    "dimension": dimension,
                    "n_mentions": len(mentioned),
                    "stated_importance_pct": stated_importance(len(mentioned), len(group)),
                    "nss": net_sentiment_score(labels),
                    "nss_ci_low": low,
                    "nss_ci_high": high,
                    "nss_vader": net_sentiment_score(mentioned[f"vader_{dimension}"].values),
                    "derived_importance": correlation,
                    "n_negative": int((group[f"sentiment_{dimension}"] == NEGATIVE).sum()),
                    "mean_rating_when_mentioned": mentioned["rating"].mean(),
                    "small_sample": len(mentioned) < small_n,
                }
            )
    return pd.DataFrame(rows)


def add_ipa_quadrants(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assign IPA quadrants using stated and derived importance separately.

    Cut-offs are grand means over the library-level rows; the derived cut-off uses only the
    pairs where derived importance is defined.
    """
    table = table.copy()
    core = table[table["library"] != "ALL"]
    cutoffs = {
        "stated_importance_mean": float(core["stated_importance_pct"].mean()),
        "nss_mean": float(core["nss"].mean()),
        "derived_importance_mean": float(core["derived_importance"].abs().mean()),
        "n_pairs_stated": int(core["stated_importance_pct"].notna().sum()),
        "n_pairs_derived": int(core["derived_importance"].notna().sum()),
    }

    def quadrant(importance, performance, importance_cut):
        if pd.isna(importance) or pd.isna(performance):
            return "not computable"
        return QUADRANTS[(importance >= importance_cut, performance >= cutoffs["nss_mean"])]

    table["quadrant_stated"] = [
        quadrant(i, p, cutoffs["stated_importance_mean"]) if lib != "ALL" else ""
        for lib, i, p in zip(table["library"], table["stated_importance_pct"], table["nss"])
    ]
    table["quadrant_derived"] = [
        quadrant(abs(d) if pd.notna(d) else np.nan, p, cutoffs["derived_importance_mean"]) if lib != "ALL" else ""
        for lib, d, p in zip(table["library"], table["derived_importance"], table["nss"])
    ]
    return table, cutoffs


def asymmetry_table(scored: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Equation (7): mean star rating when a dimension is praised and when it is criticised."""
    rows = []
    for dimension in dimensions:
        praised = scored.loc[scored[f"sentiment_{dimension}"] == POSITIVE, "rating"]
        criticised = scored.loc[scored[f"sentiment_{dimension}"] == NEGATIVE, "rating"]
        rows.append(
            {
                "dimension": dimension,
                "n_praised": len(praised),
                "n_criticised": len(criticised),
                "mean_rating_praised": praised.mean(),
                "mean_rating_criticised": criticised.mean(),
                "gap_stars": praised.mean() - criticised.mean(),
            }
        )
    return pd.DataFrame(rows)


def cramers_v(chi2: float, n: int, table_shape: tuple[int, int]) -> float:
    """Equation (8)."""
    if not np.isfinite(chi2) or n == 0 or min(table_shape) < 2:
        return float("nan")
    return float(np.sqrt(chi2 / (n * (min(table_shape) - 1))))


def sentiment_by_library(scored: pd.DataFrame, libraries: list[str]) -> tuple[pd.DataFrame, dict]:
    """Sentiment distribution per library with a chi-square test of independence."""
    counts = pd.crosstab(scored["lib"], scored["tb"]).reindex(libraries)[[POSITIVE, NEUTRAL, NEGATIVE]]
    chi2, p_value, dof = safe_chi2(counts)
    test = {
        "chi2": round(float(chi2), 3),
        "dof": int(dof),
        "p": float(p_value),
        "cramers_v": round(cramers_v(chi2, int(counts.values.sum()), counts.shape), 4),
    }
    return counts, test


def rating_tests(rated: pd.DataFrame, libraries: list[str]) -> dict:
    """Kruskal-Wallis across libraries, Mann-Whitney against the lowest-rated library, and
    a chi-square test on the share of 1-2 star ratings."""
    samples = [rated.loc[rated["lib"] == library, "rating"] for library in libraries]
    h_stat, p_value = kruskal(*samples)
    lowest = rated.groupby("lib", observed=True)["rating"].mean().idxmin()
    pairwise = {
        f"{lowest}_vs_{library}": float(
            mannwhitneyu(rated.loc[rated["lib"] == lowest, "rating"], rated.loc[rated["lib"] == library, "rating"]).pvalue
        )
        for library in libraries
        if library != lowest
    }
    low_star = pd.crosstab(rated["lib"], rated["rating"] <= 2)
    chi2, p_low, _ = safe_chi2(low_star)
    return {
        "kruskal_h": round(float(h_stat), 3),
        "kruskal_p": float(p_value),
        "lowest_rated": lowest,
        "mannwhitney_p": pairwise,
        "low_star_chi2": round(float(chi2), 3),
        "low_star_p": float(p_low),
    }


def depository_comparison(scored: pd.DataFrame, libraries: list[str], dimension: str = "DF") -> dict:
    """Compare reviews that mention the depository function with all other reviews."""
    mentions = scored[f"mentions_{dimension}"] == 1
    counts = pd.crosstab(mentions, scored["tb"])
    chi2, p_value, _ = safe_chi2(counts)
    return {
        "n_mentions": int(mentions.sum()),
        "share_pct": round(float(mentions.mean() * 100), 3),
        "share_by_library_pct": (
            scored.groupby("lib", observed=True)[f"mentions_{dimension}"].mean() * 100
        ).round(3).reindex(libraries).to_dict(),
        "positive_pct_mentioning": round(float((scored.loc[mentions, "tb"] == POSITIVE).mean() * 100), 3),
        "positive_pct_other": round(float((scored.loc[~mentions, "tb"] == POSITIVE).mean() * 100), 3),
        "mean_rating_mentioning": round(float(scored.loc[mentions, "rating"].mean()), 4),
        "mean_rating_other": round(float(scored.loc[~mentions, "rating"].mean()), 4),
        "n_criticised": int((scored[f"sentiment_{dimension}"] == NEGATIVE).sum()),
        "chi2": round(float(chi2), 3),
        "p": float(p_value),
    }


def temporal_comparison(
    rated: pd.DataFrame, scored: pd.DataFrame, libraries: list[str], split_year: int = 2024, first_year: int = 2016
) -> pd.DataFrame:
    """Compare low-rating share and positive text share before and from the split year.

    The chi-square test on the 2x2 table uses Yates' continuity correction (SciPy's default).
    """
    rows = []
    for library in libraries:
        window = rated[(rated["lib"] == library) & rated["year"].between(first_year, 9999)]
        before = window[window["year"] < split_year]
        after = window[window["year"] >= split_year]
        table = [
            [int((before["rating"] <= 2).sum()), int((before["rating"] > 2).sum())],
            [int((after["rating"] <= 2).sum()), int((after["rating"] > 2).sum())],
        ]
        chi2, p_value, _ = safe_chi2(table)
        text_before = scored[(scored["lib"] == library) & scored["year"].between(first_year, split_year - 1)]
        text_after = scored[(scored["lib"] == library) & (scored["year"] >= split_year)]
        rows.append(
            {
                "library": library,
                "n_before": len(before),
                "n_after": len(after),
                "low_star_pct_before": round(float((before["rating"] <= 2).mean() * 100), 3),
                "low_star_pct_after": round(float((after["rating"] <= 2).mean() * 100), 3),
                "chi2_yates": round(float(chi2), 3),
                "p_yates": float(p_value),
                "positive_pct_before": round(float((text_before["tb"] == POSITIVE).mean() * 100), 3),
                "positive_pct_after": round(float((text_after["tb"] == POSITIVE).mean() * 100), 3),
            }
        )
    return pd.DataFrame(rows)


def dimension_shift(scored: pd.DataFrame, library: str, dimension: str, split_year: int = 2024, first_year: int = 2016) -> dict:
    """Importance and NSS of one dimension before and from the split year, for one library."""
    window = scored[(scored["lib"] == library) & scored["year"].between(first_year, 9999)]
    out = {}
    for label, part in (("before", window[window["year"] < split_year]), ("after", window[window["year"] >= split_year])):
        mentioned = part[part[f"mentions_{dimension}"] == 1]
        out[label] = {
            "n": len(mentioned),
            "importance_pct": round(stated_importance(len(mentioned), len(part)), 3),
            "nss": round(net_sentiment_score(mentioned[f"sentiment_{dimension}"]), 3),
        }
    return out


def sensitivity(scored: pd.DataFrame, libraries: list[str]) -> pd.DataFrame:
    """Positive share by library, and the sentiment chi-square test, under each exclusion."""
    def positive_share(frame):
        return frame.groupby("lib", observed=True)["tb"].apply(lambda s: (s == POSITIVE).mean() * 100).reindex(libraries)

    base = positive_share(scored)
    subsets = {
        "all in scope": scored,
        "excluding truncated": scored[scored["flag_truncated"] == 0],
        "excluding visit prompts": scored[scored["flag_visit_prompt"] == 0],
        "excluding very short": scored[~scored["very_short"]],
    }
    rows = []
    for name, subset in subsets.items():
        shares = positive_share(subset)
        counts = pd.crosstab(subset["lib"], subset["tb"])
        _, p_value, _ = safe_chi2(counts)
        row = {"subset": name, "n": len(subset), "max_change_vs_base": float((shares - base).abs().max()),
               "sentiment_chi2_p": round(float(p_value), 4)}
        row.update({library: round(float(shares[library]), 2) for library in libraries})
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_with_rating(scored: pd.DataFrame) -> dict:
    """Rank correlation between continuous sentiment scores and star ratings."""
    rated = scored[scored["rating"].notna()]
    tb_rho, tb_p = spearmanr(rated["tb_polarity"], rated["rating"])
    vader_rho, vader_p = spearmanr(rated["vader_compound"], rated["rating"])
    return {
        "textblob_rho": round(float(tb_rho), 4), "textblob_p": float(tb_p),
        "vader_rho": round(float(vader_rho), 4), "vader_p": float(vader_p),
    }
