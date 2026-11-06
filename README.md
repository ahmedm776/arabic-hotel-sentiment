# Arabic Hotel Review Sentiment (HARD dataset)

Binary sentiment classification (positive / negative) of Arabic hotel reviews,
using the HARD balanced dataset (Elnagar, Khalifa & Einea, 2018).

## Source and license
- Data: https://github.com/elnagara/HARD-Arabic-Dataset (`data/balanced-reviews.zip`),
  collected from Booking.com in June/July 2016. Modern Standard + dialectal Arabic.
- The repository ships **no explicit LICENSE file**; the authors distribute it for
  research use with a citation request. This project is educational portfolio work
  and cites: Elnagar A., Khalifa Y.S., Einea A. (2018), Hotel Arabic-Reviews Dataset
  Construction for Sentiment Analysis Applications, Springer SCI vol 740, pp 35-52,
  doi:10.1007/978-3-319-67056-0_3.
- **Rights state (verified 2026-11-04):** the GitHub repo ships no LICENSE file
  (GitHub license API: 404) and the Hugging Face card says `license: unknown`.
  No redistribution permission exists, so the original dataset archive and raw
  reviews are NOT included in this project. Get the data from the canonical
  source with `src/download_data.sh` (checks SHA-256
  `1939c1ca59ff50bd3887223b153c7fb5c9fd232b405320244c55791bc8b5d448`).
  The trained model artifact is a small derived work included so the app runs;
  no rights over the dataset or derived artifacts are claimed or granted.

## Evaluation protocol (frozen 2026-11-04 before any model fitting)
1. Labels: rating 1-2 -> negative (0), rating 4-5 -> positive (1). Ratings of 3 do
   not occur in this balanced subset.
2. Light normalization only: remove Arabic diacritics and tatweel, unify alef
   variants and yaa forms. No stemming, no external lexicons.
3. Split: **group split by hotel** (GroupShuffleSplit, test 20%, seed 42) - no hotel
   appears in both train and test, so the evaluation measures generalization to
   hotels the model never saw, not memorized property-specific wording.
4. The TF-IDF vectorizer is fit on the training reviews only.
5. Models compared on the untouched test set exactly once: majority-class dummy,
   MultinomialNB, and logistic regression. No post-hoc tuning after viewing test
   results. All metrics reported as-is, including any baseline we fail to beat.
6. Limitations stated plainly: reviews are from 2016 Booking.com hotels; the label
   is derived from star ratings; single frozen split, no cross-validation claim.

## Model selection and amendment audit trail
- Logistic regression was PRESELECTED as the primary model before viewing test
  results; the majority dummy and MultinomialNB are baselines reported for
  context. No model was selected or tuned after viewing test metrics.
- Amendment (2026-11-04, after the first evaluation): a normalized-duplicate
  check found 475 of 22,871 test reviews (2.08%) duplicated a training review
  verbatim after normalization. These were removed from the TEST set only
  (training set untouched, no refit needed) and metrics re-evaluated once on the
  deduplicated 22,396-row test set. Both facts are recorded here and in
  metrics.json (`test_duplicates_removed`). First-evaluation figures (before
  dedup) were accuracy 0.9391 / AUC 0.9788; deduplicated: 0.9379 / 0.9782.
- Improvement round (2026-11-04, v3): five candidate configurations were
  compared with 3-fold GroupKFold BY HOTEL over the TRAINING SPLIT ONLY
  (improve/improve.py; the test set played no role in selection). Winner:
  word(1,2)+char_wb(3,5) TF-IDF union, sublinear tf, LR C=2.0 (CV accuracy
  0.9460 vs 0.9362 for the v2 configuration; +/- 0.0016 is the fold-to-fold
  standard deviation across the 3 folds, ddof=0 - a spread of fold scores,
  not a confidence interval). Refit on the full training
  split and evaluated ONCE on the same deduplicated test set - this is the
  SECOND evaluation of that test set overall, i.e. benchmark reuse, not a
  fresh unseen evaluation. Result: accuracy 0.9462 / F1 0.9480 / AUC 0.9830
  (vs 0.9379 / 0.9400 / 0.9782 in v2). Label semantics, hotel-disjoint split
  and dedup rule are unchanged.

## Reproduce

```bash
pip install -r requirements.txt
sh src/download_data.sh   # downloads from the canonical source, verifies SHA-256, unpacks
python3 src/prepare.py     # normalize, label, group-split by hotel -> data/train.pkl, data/test.pkl
python3 src/train.py       # fit on train only, evaluate once on test -> data/metrics.json, data/model.joblib
python3 src/figures.py     # figures/evaluation.png
python3 -m pytest tests/   # 5 tests (needs the regenerated data/*.pkl from prepare.py)
uvicorn src.app:app        # Arabic UI + /predict API (runs directly on the shipped model.joblib)
```

The package ships code, tests, metrics, figures and the trained model only - no
dataset files (see Source and license). download_data.sh fetches the dataset
from the canonical source with a SHA-256 check; prepare.py deterministically
rebuilds the exact train/test split (seed 42, hotel-grouped).

## Layout
- `src/prepare.py` - load, normalize, label, group-split; writes data/train.pkl, data/test.pkl
- `src/train.py` - fit vectorizer + models, evaluate once, write metrics.json + model.joblib
- `src/app.py` - FastAPI serving the saved pipeline (Arabic input)
- `tests/` - pytest suite (split integrity, label mapping, train-only fitting, prediction sanity, API)
