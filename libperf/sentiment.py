"""Sentiment scoring, aspect (dimension) detection and label validation.

Reference: Sections 3.3 and 3.4 and equations (1)-(3) of the manuscript.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE, NEUTRAL, NEGATIVE = "positive", "neutral", "negative"
VADER_THRESHOLD = 0.05
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s*[;•]\s*")

DICTIONARY_PATH = Path(__file__).resolve().parent.parent / "data" / "aspect_dictionary.json"
DIMENSION_NAMES = {
    "AS": "Affect of Service",
    "IC": "Information Control",
    "LP": "Library as Place",
    "DF": "Depository and Heritage Function",
}


def load_dictionary(path: Path | str = DICTIONARY_PATH) -> dict[str, list[str]]:
    """Load the aspect dictionary (Appendix / supplementary material)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["dimensions"]


def compile_dictionary(dictionary: dict[str, list[str]]) -> dict[str, re.Pattern]:
    """Compile one whole-word alternation pattern per dimension (longest term first)."""
    patterns = {}
    for dimension, terms in dictionary.items():
        ordered = sorted(map(re.escape, terms), key=len, reverse=True)
        patterns[dimension] = re.compile(r"\b(" + "|".join(ordered) + r")\b", re.IGNORECASE)
    return patterns


def polarity_label(polarity: float) -> str:
    """Equation (1): sign of the polarity score."""
    if polarity > 0:
        return POSITIVE
    if polarity < 0:
        return NEGATIVE
    return NEUTRAL


def vader_label(compound: float) -> str:
    """Conventional VADER thresholds of +/- 0.05."""
    if compound >= VADER_THRESHOLD:
        return POSITIVE
    if compound <= -VADER_THRESHOLD:
        return NEGATIVE
    return NEUTRAL


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]


def score_reviews(in_scope: pd.DataFrame, dictionary: dict[str, list[str]]) -> pd.DataFrame:
    """Add review-level sentiment (TextBlob, VADER) and sentence-level dimension sentiment."""
    analyzer = SentimentIntensityAnalyzer()
    scored = in_scope.copy()
    scored["tb_polarity"] = scored["text"].map(lambda t: TextBlob(t).sentiment.polarity)
    scored["tb"] = scored["tb_polarity"].map(polarity_label)
    scored["vader_compound"] = scored["text"].map(lambda t: analyzer.polarity_scores(t)["compound"])
    scored["vader"] = scored["vader_compound"].map(vader_label)

    patterns = compile_dictionary(dictionary)
    for dimension in patterns:
        scored[f"mentions_{dimension}"] = 0
        scored[f"sentiment_{dimension}"] = None
        scored[f"vader_{dimension}"] = None

    for index, text in scored["text"].items():
        sentences = split_sentences(text)
        for dimension, pattern in patterns.items():
            matched = [s for s in sentences if pattern.search(s)]
            if not matched:
                continue
            tb_mean = float(np.mean([TextBlob(s).sentiment.polarity for s in matched]))
            vader_mean = float(np.mean([analyzer.polarity_scores(s)["compound"] for s in matched]))
            scored.at[index, f"mentions_{dimension}"] = 1
            scored.at[index, f"sentiment_{dimension}"] = polarity_label(tb_mean)
            scored.at[index, f"vader_{dimension}"] = vader_label(vader_mean)

    mention_columns = [f"mentions_{d}" for d in patterns]
    scored["n_dimensions"] = scored[mention_columns].sum(axis=1)
    return scored


def lexicon_agreement(scored: pd.DataFrame) -> dict:
    """Equation (2): agreement between TextBlob and VADER, overall and per library."""
    overall = {
        "agreement_pct": round((scored["tb"] == scored["vader"]).mean() * 100, 3),
        "kappa": round(cohen_kappa_score(scored["tb"], scored["vader"]), 4),
    }
    per_library = {
        library: {
            "agreement_pct": round((group["tb"] == group["vader"]).mean() * 100, 2),
            "kappa": round(cohen_kappa_score(group["tb"], group["vader"]), 3),
        }
        for library, group in scored.groupby("lib", observed=True)
    }
    return {"overall": overall, "per_library": per_library}


def rating_convergence(scored: pd.DataFrame) -> dict:
    """Equation (3): share of non-neutral text labels whose polarity matches the star rating.

    Three-star ratings have sign 0 and therefore count as mismatches.
    """
    usable = scored[scored["rating"].notna() & (scored["tb"] != NEUTRAL)]
    star_sign = np.sign(usable["rating"] - 3)
    text_sign = np.where(usable["tb"] == POSITIVE, 1, -1)
    return {"n": int(len(usable)), "C": round(float((star_sign == text_sign).mean()), 4)}


def classifier_check(scored: pd.DataFrame, random_state: int = 42) -> dict:
    """Supervised consistency check: can lexicon labels be reproduced from the text?"""
    x_train, x_test, y_train, y_test = train_test_split(
        scored["text"], scored["tb"], test_size=0.2, stratify=scored["tb"], random_state=random_state
    )
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, stop_words="english")
    model = LinearSVC(class_weight="balanced", random_state=random_state)
    model.fit(vectoriser.fit_transform(x_train), y_train)
    predicted = model.predict(vectoriser.transform(x_test))
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    return {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "accuracy_pct": round(accuracy_score(y_test, predicted) * 100, 2),
        "f1_weighted_pct": round(f1_score(y_test, predicted, average="weighted") * 100, 2),
        "f1_macro_pct": round(f1_score(y_test, predicted, average="macro") * 100, 2),
        "f1_negative": round(report[NEGATIVE]["f1-score"], 3),
    }
