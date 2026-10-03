"""Tests for the analysis code. Run with: pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from libperf.cleaning import clean_text, estimate_date, is_scorable, prepare_reviews, word_count
from libperf import config, indicators
from libperf.sentiment import polarity_label, split_sentences, vader_label


# ---------------------------------------------------------------- cleaning
@pytest.mark.parametrize(
    "raw, expected_text, flag",
    [
        ("Weekday Wait time No wait Reservation recommended No", "", "visit_prompt"),
        ("Visited on Weekend Wait time 10-30 min", "", "visit_prompt"),
        ("Great reading room … More", "Great reading room", "truncated"),
        ("Best library ever Translated by Google ・ See original (Nepali) 1", "Best library ever", "translated"),
        ("2 years ago Nice place", "Nice place", "leaked_date"),
        ("Weekday Wait time No wait Reservation recommended Not sure", "", "visit_prompt"),
        ("Awesome library Public holiday", "Awesome library", "visit_prompt"),
        ("Edited 3 years ago Great collection", "Great collection", "edited_notice"),
        ("Edited a year ago", "", "edited_notice"),
        ("Mumbai city center Translated by Google ・ See original (Vietnamese) 0:19 0:12", "Mumbai city center", "translated"),
    ],
)
def test_platform_text_is_removed_and_flagged(raw, expected_text, flag):
    result = clean_text(raw)
    assert result.text == expected_text
    assert getattr(result, flag) is True


def test_ordinary_review_is_untouched():
    raw = "The staff were helpful and the reading hall is quiet."
    result = clean_text(raw)
    assert result.text == raw
    assert not any([result.visit_prompt, result.truncated, result.translated, result.leaked_date])


def test_reviewer_profile_summary_is_removed():
    assert clean_text("21 reviews · 2,698 photos").text == ""
    assert clean_text("655 reviews · 8,811 photos Huge collection").text == "Huge collection"


def test_edited_inside_a_sentence_is_kept():
    raw = "I edited 3 years ago my notes here"
    assert clean_text(raw).text == raw


def test_open_24_hours_keeps_its_number():
    # the trailing-number rule must not eat numbers that belong to the sentence
    assert clean_text("Open 24 hours").text == "Open 24 hours"


def test_word_count_and_scope():
    assert word_count("Nice place") == 2
    assert word_count("👍 5") == 0
    assert is_scorable("Good") is True
    assert is_scorable("🙂") is False
    assert clean_text(None).text == ""


# ---------------------------------------------------------------- dates
def test_relative_dates_use_the_midpoint_for_coarse_units():
    collected = pd.Timestamp("2026-08-25")
    one_year = estimate_date("a year ago", collected)
    # 1.5 years before collection, not 1.0
    assert abs((collected - one_year).days - int(1.5 * 365.25)) <= 1
    three_days = estimate_date("3 days ago", collected)
    assert (collected - three_days).days == 3  # no correction for fine units
    assert pd.isna(estimate_date("last week", collected))
    assert pd.isna(estimate_date(None, collected))


def test_exact_dates_take_precedence_over_relative_ones():
    frame = pd.DataFrame(
        {
            "library": ["National Library"],
            "rating": [5.0],
            "review_text": ["A quiet reading room."],
            "date_relative": ["3 years ago"],
            "date_absolute": ["2025-01-15T00:00:00Z"],
        }
    )
    prepared = prepare_reviews(frame, config.COLLECTION_DATE, config.LIBRARY_ORDER, config.LIBRARY_CODES)
    assert bool(prepared.loc[0, "date_exact"]) is True
    assert prepared.loc[0, "year"] == 2025


# ---------------------------------------------------------------- labels
def test_label_thresholds():
    assert polarity_label(0.3) == "positive"
    assert polarity_label(0.0) == "neutral"
    assert polarity_label(-0.2) == "negative"
    assert vader_label(0.05) == "positive"
    assert vader_label(0.049) == "neutral"
    assert vader_label(-0.05) == "negative"


def test_sentence_splitting():
    assert split_sentences("Staff rude. Hall quiet; garden nice") == ["Staff rude.", "Hall quiet", "garden nice"]


# ---------------------------------------------------------------- indicators
def test_net_sentiment_score():
    labels = ["positive"] * 8 + ["negative"] * 2
    assert indicators.net_sentiment_score(labels) == pytest.approx(60.0)
    assert np.isnan(indicators.net_sentiment_score([]))


def test_stated_importance():
    assert indicators.stated_importance(25, 200) == pytest.approx(12.5)


def test_derived_importance_matches_pearson_correlation():
    rng = np.random.default_rng(0)
    negative = pd.Series(rng.integers(0, 2, 200))
    rating = pd.Series(5 - 3 * negative + rng.normal(0, 0.5, 200))
    expected = np.corrcoef(negative, rating)[0, 1]
    assert indicators.derived_importance(negative, rating) == pytest.approx(expected, abs=1e-10)


def test_derived_importance_undefined_with_one_negative_mention():
    assert np.isnan(indicators.derived_importance(pd.Series([0, 0, 1, 0]), pd.Series([5, 4, 1, 5])))


def test_cramers_v():
    assert indicators.cramers_v(chi2=40.0, n=1000, table_shape=(4, 3)) == pytest.approx(np.sqrt(40 / 2000))


def test_bootstrap_interval_is_reproducible_and_brackets_the_estimate():
    labels = ["positive"] * 40 + ["negative"] * 10
    first = indicators.bootstrap_interval(labels, np.random.default_rng(2026), n_boot=500)
    second = indicators.bootstrap_interval(labels, np.random.default_rng(2026), n_boot=500)
    assert first == second
    assert first[0] < indicators.net_sentiment_score(labels) < first[1]


def test_quadrant_rule():
    table = pd.DataFrame(
        {
            "library": ["A", "A", "B", "B"],
            "dimension": ["AS", "LP", "AS", "LP"],
            "n_mentions": [10, 10, 10, 10],
            "stated_importance_pct": [5.0, 45.0, 5.0, 45.0],
            "nss": [10.0, 90.0, 90.0, 10.0],
            "derived_importance": [-0.4, -0.1, -0.4, -0.1],
            "small_sample": [False] * 4,
        }
    )
    result, cutoffs = indicators.add_ipa_quadrants(table)
    assert cutoffs["nss_mean"] == pytest.approx(50.0)
    assert result.loc[0, "quadrant_stated"] == "Low priority"        # rarely mentioned, poor
    assert result.loc[0, "quadrant_derived"] == "Concentrate here"   # but strongly tied to ratings
    assert result.loc[1, "quadrant_stated"] == "Keep up the good work"


def test_quadrant_not_computable_when_derived_importance_missing():
    table = pd.DataFrame(
        {
            "library": ["A", "A"],
            "dimension": ["AS", "DF"],
            "n_mentions": [10, 3],
            "stated_importance_pct": [20.0, 2.0],
            "nss": [40.0, 60.0],
            "derived_importance": [-0.3, np.nan],
            "small_sample": [False, True],
        }
    )
    result, _ = indicators.add_ipa_quadrants(table)
    assert result.loc[1, "quadrant_derived"] == "not computable"


# ---------------------------------------------------------------- end to end
def test_pipeline_runs_on_the_sample_data(tmp_path):
    from scripts.make_sample_data import build
    from scripts.run_analysis import main

    sample = tmp_path / "sample.csv"
    build(120, seed=1).to_csv(sample, index=False)
    results = main(["--input", str(sample), "--output", str(tmp_path / "out"), "--bootstrap-draws", "50",
                    "--skip-classifier"])

    assert results["sample"]["n_reviews"] == 120
    assert results["sample"]["n_in_scope"] < 120          # platform-only rows are dropped
    assert results["sample"]["n_dropped_visit_prompt_only"] > 0
    for name in ["table1_sample_sentiment.csv", "table2_dimensions_ipa.csv", "figure1_ipa.png"]:
        assert (tmp_path / "out" / name).exists()


def test_unknown_library_is_rejected(tmp_path):
    from scripts.run_analysis import load_reviews

    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {"library": ["Some Other Library"], "rating": [5], "review_text": ["Nice"], "date_relative": ["a year ago"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown library"):
        load_reviews(path)


def test_missing_column_is_rejected(tmp_path):
    from scripts.run_analysis import load_reviews

    path = tmp_path / "bad.csv"
    pd.DataFrame({"library": ["National Library"], "rating": [5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_reviews(path)
