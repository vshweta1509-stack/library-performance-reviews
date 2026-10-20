"""Shareable derived data: every variable the analysis needs except the review text.

Google Maps reviews are written by identifiable people, so the review text, reviewer names and
Google's review identifiers are not redistributed. This module writes, and reads back, a table
with one row per collected review that keeps only derived variables: library, star rating,
year, cleaning flags, scope, word count, sentiment scores and labels, and dimension labels.
Each review is given a study identifier (R00001, R00002, ...) in the order of the input file;
the identifier cannot be linked to the platform.

From this table and the human coding (keyed by the same identifiers), every published value
can be recomputed except the five that train a text classifier (see README).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

DERIVED_COLUMNS = [
    "review_id", "lib", "rating", "year", "date_exact", "has_text", "in_scope", "dropped_reason",
    "n_words", "very_short", "flag_visit_prompt", "flag_truncated", "flag_translated",
    "flag_edited_notice", "tb_polarity", "tb", "vader_compound", "vader", "n_dimensions",
] + [f"{prefix}_{d}" for d in config.DIMENSIONS for prefix in ("mentions", "sentiment", "vader")]

# Columns that must never be written to the shareable table.
WITHHELD_COLUMNS = {"text", "review_text", "review_text_translated", "source_text", "raw_text_block",
                    "author", "date", "date_absolute", "date_relative", "platform_review_id"}


def dropped_reason(prepared: pd.DataFrame) -> pd.Series:
    """Why a review with text was not analysed (Section 3.2); empty for all other rows."""
    reason = pd.Series("", index=prepared.index, dtype=object)
    dropped = prepared["has_text"] & ~prepared["in_scope"]
    raw = prepared.loc[dropped, "review_text"].astype(str)
    visit_prompt = raw.str.contains("Weekday|Weekend|Public holiday|Wait time|Reservation", regex=True)
    edit_notice = ~visit_prompt & raw.str.match(r"\s*Edited\s")
    reason.loc[raw.index] = "other"
    reason.loc[raw.index[visit_prompt.to_numpy()]] = "visit_prompt_only"
    reason.loc[raw.index[edit_notice.to_numpy()]] = "edit_notice_only"
    return reason


def study_ids(n: int) -> list[str]:
    return [f"R{i:05d}" for i in range(1, n + 1)]


def build_derived(prepared: pd.DataFrame, scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the shareable table and the private map from platform review_id to study id."""
    table = prepared.copy()
    table["platform_review_id"] = table.get("review_id", pd.Series(index=table.index, dtype=object))
    table["review_id"] = study_ids(len(table))
    table["dropped_reason"] = dropped_reason(prepared)
    score_columns = [c for c in DERIVED_COLUMNS if c in scored.columns and c not in prepared.columns]
    table = table.join(scored[score_columns])
    for column in ("n_words", "very_short", "flag_visit_prompt", "flag_truncated", "flag_translated",
                   "flag_edited_notice"):
        table[column] = table[column].astype(object).where(table["in_scope"], None)
    for column in ("n_words", "flag_visit_prompt", "flag_truncated", "flag_translated", "flag_edited_notice",
                   "n_dimensions", *[f"mentions_{d}" for d in config.DIMENSIONS]):
        table[column] = table[column].astype("Int64")
    table["year"] = table["year"].astype("Int64")
    key = dict(zip(table["platform_review_id"].astype(str), table["review_id"]))
    out = table[DERIVED_COLUMNS].copy()
    assert not WITHHELD_COLUMNS & set(out.columns)
    return out, key


def recode_human(codes: pd.DataFrame, key: dict[str, str]) -> pd.DataFrame:
    """Replace platform review ids in a coding sheet with study ids."""
    out = codes.copy()
    out["review_id"] = out["review_id"].astype(str).map(key)
    if out["review_id"].isna().any():
        missing = int(out["review_id"].isna().sum())
        raise ValueError(f"{missing} coded review(s) not found in the review table")
    return out


def load_derived(path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the shareable table and return frames shaped like the pipeline's prepared and scored data."""
    table = pd.read_csv(path, dtype={"review_id": str, "dropped_reason": str}, keep_default_na=True)
    table["dropped_reason"] = table["dropped_reason"].fillna("")
    for column in ("date_exact", "has_text", "in_scope"):
        table[column] = table[column].astype(str).str.lower().eq("true")
    table["year"] = table["year"].astype("Int64")
    table["lib"] = pd.Categorical(table["lib"], config.LIBRARIES, ordered=True)
    prepared = table
    scored = table[table["in_scope"]].copy()
    scored["very_short"] = scored["very_short"].astype(str).str.lower().eq("true")
    for column in ("n_words", "flag_visit_prompt", "flag_truncated", "flag_translated", "flag_edited_notice",
                   "n_dimensions", *[f"mentions_{d}" for d in config.DIMENSIONS]):
        scored[column] = scored[column].astype(int)
    for column in [f"{p}_{d}" for d in config.DIMENSIONS for p in ("sentiment", "vader")]:
        scored[column] = scored[column].astype(object).where(scored[column].notna(), None)
    return prepared, scored
