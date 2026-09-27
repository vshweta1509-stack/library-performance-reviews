"""Cleaning of scraped Google Maps reviews.

Removes text that the platform (not the reviewer) contributed, decides which reviews can be
scored, and estimates dates for reviews that carry only a relative age ("3 years ago").

Reference: Section 3.2 and equation (9) of the manuscript.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# Google's structured "visit information" prompts, captured by scrapers as if they were review text.
VISIT_PROMPT_PATTERNS = [
    r"Visited on\s+(Weekday|Weekend)s?",
    r"\b(Weekday|Weekend)s?\s*…?",
    r"Wait time\s*(No wait|Up to \d+ min|\d+\s*[–-]\s*\d+\s*min|\d+ min\+?|Over an hour|\d+\+? ?hours?)?",
    r"Reservation recommended\s*(Yes|No|Not sure)?",
    r"Recommended for\s*\w+",
]
# Google's translation notices.
TRANSLATION_PATTERNS = [
    r"Translated by Google\s*・?\s*See original\s*\([^)]*\)(\s*\d{1,4})?\s*$",
    r"\(Translated by Google\)",
    r"\(Original\).*$",
]
# A relative date occasionally leaks into the start of the text block.
LEAKED_DATE_PATTERN = r"^\s*(an?|\d+)\s+(minute|hour|day|week|month|year)s?\s+ago\s*(NEW)?\s*"
TRUNCATION_MARKER = r"…\s*More"          # marks a review Google displayed collapsed
TRUNCATION_PATTERN = r"…\s*More(\s*\d{1,4})?\s*$"  # removed only when it ends the text
TRAILING_COUNT_PATTERN = r"(?<=[.!?)\s])\s*\d{1,4}\s*$"  # like-count left by the scraper

RELATIVE_AGE_PATTERN = re.compile(
    r"(an?|\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", re.IGNORECASE
)
UNIT_DAYS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1.0, "week": 7.0, "month": 30.44, "year": 365.25}
HALF_UNIT_CORRECTION = {"month", "year"}  # coarse units; see equation (9)


@dataclass
class CleanedText:
    """Result of cleaning one review's text."""

    text: str
    visit_prompt: bool = False
    truncated: bool = False
    translated: bool = False
    leaked_date: bool = False

    def flags(self) -> dict:
        return {
            "flag_visit_prompt": int(self.visit_prompt),
            "flag_truncated": int(self.truncated),
            "flag_translated": int(self.translated),
            "flag_leaked_date": int(self.leaked_date),
        }


def clean_text(raw: object) -> CleanedText:
    """Strip platform-generated text from one review and record what was found."""
    if not isinstance(raw, str):
        return CleanedText("")
    result = CleanedText(raw)
    text = raw

    if re.search(LEAKED_DATE_PATTERN, text):
        result.leaked_date = True
        text = re.sub(LEAKED_DATE_PATTERN, "", text)

    if re.search(TRUNCATION_MARKER, text):
        result.truncated = True
    text = re.sub(TRUNCATION_PATTERN, " ", text)

    for pattern in TRANSLATION_PATTERNS:
        if re.search(pattern, text, flags=re.I | re.S):
            result.translated = True
            text = re.sub(pattern, " ", text, flags=re.I | re.S)

    for pattern in VISIT_PROMPT_PATTERNS:
        if re.search(pattern, text, flags=re.I | re.S):
            result.visit_prompt = True
            text = re.sub(pattern, " ", text, flags=re.I | re.S)

    if re.search(r"…\s*$", text):
        result.truncated = True
    text = text.replace("…", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(TRAILING_COUNT_PATTERN, "", text).strip()

    result.text = text
    return result


def word_count(text: str) -> int:
    """Number of alphabetic tokens of two or more characters."""
    return len(re.findall(r"[A-Za-z]{2,}", text))


def is_scorable(text: str) -> bool:
    """A review can be scored if at least one alphabetic word survives cleaning."""
    return word_count(text) >= 1


def estimate_date(relative_age: object, collection_date: pd.Timestamp) -> pd.Timestamp:
    """Estimate a timestamp from a relative age, per equation (9).

    Google displays "N units ago" for ages between N and N + 1 units, so coarse units
    (months, years) are placed at the midpoint of that interval.
    """
    if not isinstance(relative_age, str):
        return pd.NaT
    match = RELATIVE_AGE_PATTERN.match(relative_age.strip())
    if match is None:
        return pd.NaT
    count_text, unit = match.group(1).lower(), match.group(2).lower()
    count = 1 if count_text in {"a", "an"} else int(count_text)
    if unit in HALF_UNIT_CORRECTION:
        count += 0.5
    return collection_date - pd.Timedelta(days=count * UNIT_DAYS[unit])


def prepare_reviews(
    reviews: pd.DataFrame,
    collection_date: pd.Timestamp,
    library_order: list[str],
    library_codes: dict[str, str],
    short_review_words: int = 2,
) -> pd.DataFrame:
    """Clean, scope and date a raw review table.

    Adds: text, cleaning flags, has_text, in_scope, n_words, very_short, date, date_exact, year.
    Uses Google's English translation when the original review is not in English.
    """
    data = reviews.copy()
    data["library"] = pd.Categorical(data["library"], library_order, ordered=True)
    data["lib"] = data["library"].map(library_codes)

    original_language = data.get("original_language", pd.Series(index=data.index, dtype=object))
    translation = data.get("review_text_translated", pd.Series(index=data.index, dtype=object))
    use_translation = (
        translation.notna()
        & original_language.notna()
        & ~original_language.fillna("").astype(str).str.startswith("en")
    )
    data["source_text"] = translation.where(use_translation, data["review_text"])

    cleaned = data["source_text"].map(clean_text)
    data["text"] = [c.text for c in cleaned]
    for key in ("flag_visit_prompt", "flag_truncated", "flag_translated", "flag_leaked_date"):
        data[key] = [c.flags()[key] for c in cleaned]

    data["has_text"] = data["review_text"].notna() & data["review_text"].astype(str).str.strip().ne("")
    data["in_scope"] = data["has_text"] & data["text"].map(is_scorable)
    data["n_words"] = data["text"].map(word_count)
    data["very_short"] = data["in_scope"] & (data["n_words"] <= short_review_words)

    exact = pd.to_datetime(data.get("date_absolute"), utc=True, errors="coerce")
    exact = exact.dt.tz_localize(None).astype("datetime64[ns]")
    approximate = data["date_relative"].map(lambda v: estimate_date(v, collection_date))
    approximate = pd.to_datetime(approximate).astype("datetime64[ns]")
    data["date_exact"] = exact.notna()
    data["date"] = exact.fillna(approximate)
    data["year"] = data["date"].dt.year.astype("Int64")
    return data
