# Measuring library performance from Google Maps reviews

Analysis code for the study *Staff, space and the deposit mandate: measuring the performance of
India's legal deposit libraries from Google Maps reviews*.

The code turns public Google Maps reviews into library performance indicators. It cleans text that
the platform (rather than the reviewer) contributed, scores sentiment and validates it, classifies
review sentences into four service dimensions, and computes importance–performance indicators,
statistical tests and the published figure, and validates the automatic labels against human coding.

It is written to be reusable: to apply it to other libraries, edit `libperf/config.py` and, if
needed, the dictionary in `data/aspect_dictionary.json`. Nothing else needs to change.

## What it reproduces

`scripts/verify_paper.py` checks 87 values published in the paper, including every count, test
statistic, p-value and indicator, plus 29 human-validation values when the coding sheets are
supplied (116 in all). On the study data all 116 reproduce, and all 80 cells of Table 2 match the
published table.

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start (no study data needed)

A small synthetic dataset is included so the pipeline can be run immediately. Its text is generated
from templates; it contains no real reviews and reproduces no findings.

```bash
python -m scripts.run_analysis --input data/sample_reviews.csv --output outputs_demo
```

## Reproducing the published analysis

With the review table described under **Data** below:

```bash
python -m scripts.run_analysis --input four_libraries_combined.xlsx --output outputs
python -m scripts.validate_human --output outputs \
    --sentiment-codes sentiment_codes.csv --aspect-codes aspect_codes.csv
python -m scripts.verify_paper --output outputs
```

The second command compares the automatic labels with the human coding (see **Human validation**
below); it can be skipped, in which case only the 87 values that do not depend on the coding sheets
are checked. The last command prints how many of the published values were reproduced and exits
non-zero if any differ.

## Input format

A `.xlsx` or `.csv` table, one row per review.

| Column | Required | Description |
| --- | --- | --- |
| `library` | yes | Library name, matching `LIBRARY_ORDER` in `libperf/config.py` |
| `rating` | yes | Star rating, 1–5 (may be blank for text-only rows) |
| `review_text` | yes | Review text as scraped, including any platform text |
| `date_relative` | yes | Relative age as displayed, e.g. "3 years ago" |
| `date_absolute` | no | Exact timestamp where the scraper provides one; takes precedence |
| `review_text_translated`, `original_language` | no | Google's translation and the detected language; the translation is used when the original is not English |
| `review_id` | no | Identifier carried through to the output |

## Outputs

| File | Contents |
| --- | --- |
| `results.json` | Every statistic quoted in the text |
| `table1_sample_sentiment.csv` | Table 1: reviews, ratings and sentiment by library |
| `table2_dimensions_ipa.csv` | Table 2: importance, Net Sentiment Score with bootstrap interval, derived importance, IPA quadrants |
| `table3_asymmetry.csv` | Mean ratings when a dimension is praised and criticised (equation 7) |
| `table4_temporal.csv` | Before/after 2024 comparison |
| `table5_sensitivity.csv` | Positive share and sentiment test under each exclusion |
| `scored_reviews.csv` | One row per analysed review with all labels |
| `validation_human.json` | Agreement with human coding (written by `scripts/validate_human.py`) |
| `figure1_ipa.png` | Figure 1 |
| `figure2_temporal.png` | Supporting figure: mean annual rating |

Figures are regenerated at run time and may differ from the published file in font rendering; the
underlying values are identical.

## Method summary

1. **Cleaning** (`libperf/cleaning.py`) removes Google's structured visit prompts ("Wait time",
   "Reservation recommended", "Public holiday"), edit notices ("Edited 3 years ago"), translation
   notices, truncation markers, reviewer profile summaries, like counts and leaked date stamps, then keeps reviews that still contain at least one alphabetic word. Reviews carrying only
   a relative age are dated at the midpoint of the interval that age covers (equation 9).
2. **Sentiment** (`libperf/sentiment.py`) scores each review with TextBlob and, independently, with
   VADER, and validates the labels four ways: agreement between the lexicons (Cohen's κ), agreement
   with the reviewer's own star rating, a supervised classifier check, and comparison with human
   coding (`scripts/validate_human.py`).
3. **Dimensions** (`libperf/sentiment.py`) splits reviews into sentences and assigns each sentence to
   every dimension whose dictionary terms it contains: affect of service, information control,
   library as place, and depository and heritage function.
4. **Indicators** (`libperf/indicators.py`) computes Net Sentiment Score, stated importance (share of
   reviews mentioning a dimension) and derived importance (point-biserial correlation between
   negative comment and star rating), assigns IPA quadrants under both importance measures, and runs
   the chi-square, Kruskal–Wallis, Mann–Whitney and temporal tests.

Bootstrap intervals use a seeded generator (`BOOTSTRAP_SEED`), so results are reproducible. The
chi-square tests on 2×2 temporal tables use Yates' continuity correction (SciPy's default).

## Known behaviours

- The truncation flag marks any review displaying "… More"; the marker is removed only when it ends
  the text, so a few reviews keep a trailing "More" and a video timestamp. Excluding all truncated
  reviews changes any library's positive share by at most 1.3 points (see `table5_sensitivity.csv`).
- Derived importance is undefined where a dimension attracts at most one negative mention; such
  cells are reported as missing rather than zero.
- Chi-square tests return a missing value instead of failing when a table is too sparse, which can
  happen on small datasets.

## Tests

```bash
pip install -r requirements-dev.txt && pytest -q
```

30 tests cover the cleaning rules, date estimation, label thresholds, each indicator formula,
quadrant assignment, bootstrap reproducibility, input validation and an end-to-end run.

## Data

The review data are not included. Google Maps reviews are public but are written by identifiable
people, so the analysed file is available from the author on reasonable request rather than being
redistributed here. Reviewer names are discarded before analysis, and no personal data are used.
Aggregate results for the published study are in `results/`.

## Human validation

`scripts/validate_human.py` compares the automatic labels with human coding:

- **Sentiment.** Two LIS research scholars independently labelled a random sample of 250 reviews
  (positive, neutral, negative); disagreements were adjudicated into a reference label. Six sampled
  reviews that consisted only of platform text were replaced by random draws (S251–S256). The script
  reports inter-coder agreement and the accuracy, κ and F1 of TextBlob, VADER and a classifier
  retrained without the sampled reviews.
- **Dimensions.** A third LIS research scholar coded a stratified random sample of 200 reviews
  (50 per library) for whether, and with what tone, each dimension is discussed; five reviews that
  consisted only of platform text were replaced by random draws from the same library (A201–A205).
  The script reports κ, precision, recall and F1 for dictionary assignment and tone
  agreement where both identify a dimension.

Input files (`sentiment_codes.csv`: `review_id, coder_A, coder_B, gold`; `aspect_codes.csv`:
`review_id, AS, IC, LP, DF`) are available from the authors on request with the review data.

## Changes

- **1.1.0** Cleaning now also removes "Public holiday" visit prompts, "Reservation recommended: Not
  sure" (previously left as "t sure"), "Edited N years ago" notices, reviewer profile summaries and
  translation notices followed by video timestamps; 45 reviews that contained only such text are no
  longer analysed (2,445 rather than 2,490). Platform-text-only reviews in the validation samples were
  replaced by random draws and coded by the same coders. Added `scripts/validate_human.py` and the
  human-validation checks in `scripts/verify_paper.py`.
- **1.0.1** Same code as 1.0.0, re-released so that Zenodo would archive it and issue a DOI.
- **1.0.0** First release.

## Citation

See `CITATION.cff`. Please cite the paper and, if you use the code, this repository.

## Use of generative AI

Claude (Anthropic; model Claude Opus 5) was used to write the code in this repository and to draft
the manuscript. The authors reviewed the code and results and are responsible for them.

## Licence

MIT (see `LICENSE`). The aspect dictionary and documentation may be reused under the same terms.
