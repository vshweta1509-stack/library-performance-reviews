"""Validate the automatic labels against human coding (Sections 3.3, 3.4 and 4.1-4.2).

Run after scripts/run_analysis.py:

    python -m scripts.validate_human --output outputs \
        --sentiment-codes sentiment_codes.csv --aspect-codes aspect_codes.csv

Inputs (not distributed; available from the authors on request):

    sentiment_codes.csv  one row per sampled review: review_id, coder_A, coder_B, gold
                         (Positive / Neutral / Negative; gold = adjudicated label)
    aspect_codes.csv     one row per sampled review: review_id, AS, IC, LP, DF
                         (Positive / Negative / Neutral / Mixed / Not mentioned)

Only sampled reviews that are in the analysed set (outputs/scored_reviews.csv) are used.
Writes outputs/validation_human.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.svm import LinearSVC

from libperf import config

LABELS = ["positive", "neutral", "negative"]
NOT_MENTIONED = "not mentioned"


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def compare(gold: pd.Series, predicted: pd.Series) -> dict:
    """Accuracy, Cohen's kappa, macro F1 and per-class F1 of predicted against gold labels."""
    per_class = f1_score(gold, predicted, labels=LABELS, average=None, zero_division=0)
    return {
        "n": int(len(gold)),
        "accuracy_pct": round(accuracy_score(gold, predicted) * 100, 2),
        "kappa": round(cohen_kappa_score(gold, predicted), 4),
        "f1_macro": round(f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0), 4),
        "f1_by_class": {label: round(float(score), 4) for label, score in zip(LABELS, per_class)},
        "negative_found": int(((gold == "negative") & (predicted == "negative")).sum()),
    }


def retrained_classifier(scored: pd.DataFrame, sample: pd.DataFrame, random_state: int) -> pd.Series:
    """Classifier retrained without the sampled reviews, then applied to them.

    Every review whose text matches a sampled text is left out of training, so identical short
    reviews ("Good", "Nice") cannot leak from the test set into the training set.
    """
    sample_texts = set(sample["text"])
    train = scored[~scored["text"].isin(sample_texts)]
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, stop_words="english")
    model = LinearSVC(class_weight="balanced", random_state=random_state)
    model.fit(vectoriser.fit_transform(train["text"]), train["tb"])
    predicted = pd.Series(model.predict(vectoriser.transform(sample["text"])), index=sample.index)
    predicted.attrs["n_train"] = int(len(train))
    return predicted


def sentiment_validation(scored: pd.DataFrame, codes: pd.DataFrame, random_state: int) -> dict:
    lookup = scored.drop_duplicates("review_id").set_index("review_id")
    in_scope = codes[codes["review_id"].isin(lookup.index)].copy()
    for column in ("coder_A", "coder_B", "gold"):
        in_scope[column] = _norm(in_scope[column])
    for column in ("text", "tb", "vader"):
        in_scope[column] = lookup.loc[in_scope["review_id"], column].to_numpy()

    both = in_scope[in_scope["coder_A"].isin(LABELS) & in_scope["coder_B"].isin(LABELS)]
    classifier = retrained_classifier(scored, in_scope, random_state)
    return {
        "n_sampled": int(len(codes)),
        "n_in_scope": int(len(in_scope)),
        "gold_counts": in_scope["gold"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "inter_coder": {
            "n": int(len(both)),
            "agreement_pct": round(float((both["coder_A"] == both["coder_B"]).mean() * 100), 2),
            "kappa": round(cohen_kappa_score(both["coder_A"], both["coder_B"]), 4),
        },
        "textblob": compare(in_scope["gold"], in_scope["tb"]),
        "vader": compare(in_scope["gold"], in_scope["vader"]),
        "classifier": {**compare(in_scope["gold"], classifier), "n_train": classifier.attrs["n_train"]},
    }


def aspect_validation(scored: pd.DataFrame, codes: pd.DataFrame) -> dict:
    lookup = scored.drop_duplicates("review_id").set_index("review_id")
    in_scope = codes[codes["review_id"].isin(lookup.index)].copy()
    results: dict = {"n_sampled": int(len(codes)), "n_in_scope": int(len(in_scope)), "dimensions": {}}
    for dimension in config.DIMENSIONS:
        human_label = _norm(in_scope[dimension])
        human = human_label.ne(NOT_MENTIONED).to_numpy()
        machine = lookup.loc[in_scope["review_id"], f"mentions_{dimension}"].astype(int).eq(1).to_numpy()
        machine_label = _norm(lookup.loc[in_scope["review_id"], f"sentiment_{dimension}"]).to_numpy()
        both = human & machine
        true_positive = int(both.sum())
        precision = true_positive / machine.sum() if machine.sum() else np.nan
        recall = true_positive / human.sum() if human.sum() else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
        results["dimensions"][dimension] = {
            "human_n": int(human.sum()),
            "dictionary_n": int(machine.sum()),
            "both_n": true_positive,
            "agreement_pct": round(float((human == machine).mean() * 100), 2),
            "kappa": round(cohen_kappa_score(human, machine), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tone_agree_n": int((human_label.to_numpy()[both] == machine_label[both]).sum()),
        }
    tone = [
        d["tone_agree_n"] / d["both_n"] * 100 for d in results["dimensions"].values() if d["both_n"]
    ]
    results["tone_agree_pct_range"] = [round(min(tone), 1), round(max(tone), 1)]
    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Validate automatic labels against human coding")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="run_analysis output directory")
    parser.add_argument("--sentiment-codes", required=True, type=Path)
    parser.add_argument("--aspect-codes", required=True, type=Path)
    args = parser.parse_args(argv)

    scored = pd.read_csv(args.output / "scored_reviews.csv", dtype={"review_id": str})
    results = {
        "sentiment": sentiment_validation(
            scored, pd.read_csv(args.sentiment_codes, dtype={"review_id": str}), config.CLASSIFIER_SEED
        ),
        "aspects": aspect_validation(scored, pd.read_csv(args.aspect_codes, dtype={"review_id": str})),
    }
    with open(args.output / "validation_human.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    s, a = results["sentiment"], results["aspects"]
    print(
        f"Sentiment: {s['n_in_scope']} of {s['n_sampled']} sampled reviews in scope; "
        f"Aspects: {a['n_in_scope']} of {a['n_sampled']}. Written to {args.output / 'validation_human.json'}"
    )
    return results


if __name__ == "__main__":
    main()
