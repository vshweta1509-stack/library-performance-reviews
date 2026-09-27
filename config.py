"""Study constants: the libraries, the collection date and the analysis settings.

Edit this file to apply the method to a different set of libraries.
"""
from __future__ import annotations

import pandas as pd

# Collection date of the scrape; relative ages ("3 years ago") are counted back from it.
COLLECTION_DATE = pd.Timestamp("2026-08-25")

LIBRARY_ORDER = [
    "National Library",
    "Connemara Public Library",
    "The Asiatic Society, Mumbai",
    "Delhi Public Library",
]
LIBRARY_CODES = {
    "National Library": "NLI",
    "Connemara Public Library": "CPL",
    "The Asiatic Society, Mumbai": "ASM",
    "Delhi Public Library": "DPL",
}
LIBRARY_NAMES = {
    "NLI": "National Library of India",
    "CPL": "Connemara Public Library",
    "ASM": "Asiatic Society of Mumbai",
    "DPL": "Delhi Public Library",
}
LIBRARIES = ["NLI", "CPL", "ASM", "DPL"]
DIMENSIONS = ["AS", "IC", "LP", "DF"]

BOOTSTRAP_SEED = 2026
BOOTSTRAP_DRAWS = 2000
CLASSIFIER_SEED = 42
SMALL_SAMPLE = 30          # cells below this are flagged in Table 2
SHORT_REVIEW_WORDS = 2     # "very short" reviews
SPLIT_YEAR = 2024          # before/after comparison in Section 4.4
FIRST_YEAR = 2016          # first year with usable review volume

REQUIRED_COLUMNS = ["library", "rating", "review_text", "date_relative"]
OPTIONAL_COLUMNS = ["date_absolute", "review_text_translated", "original_language", "review_id"]
