# Derived dataset: codebook

Study: *Staff, space and the deposit mandate: measuring the performance of India's legal deposit
libraries from Google Maps reviews*. Reviews collected 23–25 August 2026.

These files contain no review text, reviewer names, profile links, Google review identifiers or
exact dates. `review_id` is a study identifier assigned in the order of the collected file; the
key linking it to Google's identifiers is held privately by the authors.

## reviews_derived.csv

One row per collected review (5,411 rows). Columns from `n_words` onward are blank for reviews
that were not analysed (`in_scope` = False).

| Variable | Type | Values / meaning |
| --- | --- | --- |
| `review_id` | string | Study identifier, R00001–R05411 |
| `lib` | string | NLI National Library of India; CPL Connemara Public Library; ASM Asiatic Society of Mumbai; DPL Delhi Public Library |
| `rating` | integer-valued | Star rating 1–5 given by the reviewer; blank for one review |
| `year` | integer | Year of the review: exact for NLI, otherwise estimated from Google's relative age by equation (9) of the article |
| `date_exact` | boolean | True when the year comes from an exact timestamp (NLI) |
| `has_text` | boolean | The review had any text as collected |
| `in_scope` | boolean | At least one word remained after platform text was removed; these 2,445 reviews are analysed |
| `dropped_reason` | string | For reviews with text that were not analysed: `visit_prompt_only`, `edit_notice_only` or `other` |
| `n_words` | integer | Alphabetic words of two or more letters after cleaning |
| `very_short` | boolean | Two words or fewer |
| `flag_visit_prompt` | 0/1 | Google visit prompts ("Wait time", "Reservation recommended" …) were removed from the text |
| `flag_truncated` | 0/1 | Text was stored truncated ("… More") |
| `flag_translated` | 0/1 | Text was Google's English translation of a non-English review |
| `flag_edited_notice` | 0/1 | An "Edited N years ago" notice was removed |
| `tb_polarity` | float | TextBlob polarity, −1 to 1 |
| `tb` | string | TextBlob label: positive (> 0), neutral (0), negative (< 0); equation (1) |
| `vader_compound` | float | VADER compound score, −1 to 1 |
| `vader` | string | VADER label with ±0.05 thresholds |
| `n_dimensions` | integer | Number of the four dimensions mentioned, 0–4 |
| `mentions_D` | 0/1 | At least one sentence matches the dictionary for dimension D |
| `sentiment_D` | string | Label of the mean TextBlob polarity of the matching sentences; blank if not mentioned |
| `vader_D` | string | The same with VADER |

D is one of AS (affect of service: staff), IC (information control: collections and access),
LP (library as place) and DF (depository and heritage function). The dictionary is
`data/aspect_dictionary.json` (Table S1 of the article).

## sentiment_codes.csv

Human sentiment coding of a random sample of 250 analysed reviews plus 6 replacement draws for
sampled reviews that held only platform text (256 rows; 250 are in scope).

| Variable | Meaning |
| --- | --- |
| `code_id` | Coding-sheet identifier (S001–S256) |
| `review_id` | Study identifier, as above |
| `coder_A`, `coder_B` | Independent labels: Positive, Neutral, Negative |
| `gold` | Adjudicated reference label |

## aspect_codes.csv

Human aspect coding of a stratified random sample of 200 analysed reviews (50 per library) plus
5 replacement draws (205 rows; 200 are in scope).

| Variable | Meaning |
| --- | --- |
| `code_id` | Coding-sheet identifier (A001–A205) |
| `review_id` | Study identifier, as above |
| `AS`, `IC`, `LP`, `DF` | Positive, Negative, Neutral, Mixed or Not mentioned |

Coding rules are Tables S2 and S3 of the article's supplementary material.
