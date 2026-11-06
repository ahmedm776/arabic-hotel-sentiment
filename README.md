# Arabic Hotel Sentiment - Corrected Preprocessing Repair (round 2, 2026-11-05)

Supersedes the 2026-11-05 20:52 repair (model_v3_repaired.joblib, acc 88.85), which is
PROVISIONAL: its strip regex carried two descriptors in raw hamza-on-yaa orthography
(استثنائي, رائع) while stripping AFTER normalization (which folds ئ->ي), so 9,845 rows
(7,294 استثنائي + 2,551 رائع) kept their label-leaking header. That artifact is preserved
for audit only and must not be used.

## What this package is
A correctness repair of the v3 preprocessing, not an accuracy experiment. Same source
data, same split, same v3 hyperparameters; no tuning, no CV, no new architecture, no new
data. Trained ONCE under a 10-minute compute cap (fit+vectorize: 20.9s).

## Pipeline (canonical module: header_re.py, 44 tests in test_header_re.py)
1. Load raw (sha256 86c1ae1b...ab02, identical to audit-r2 input; 105,698 rows).
2. ORIGINAL seed-42 hotel-group split (GroupShuffleSplit, test_size 0.2): 82,827 / 22,871.
   Asserted identical hotel assignment to the previous repair; zero hotel overlap.
3. Mask parity assertion vs audit r2 on ALL raw rows: 43,685 headers, per-descriptor
   counts identical (byte-identical regex pattern).
4. STRIP header on RAW text FIRST (repeated: 176 rows carried a double header).
   Train 34,316 + test 9,369 stripped rows = 43,685 exactly.
5. Normalize (v3/audit-identical folding), then strip the 2 headers written in
   normalized orthography ("رايع."). Header-only reviews dropped (77 train / 13 test).
6. ASSERT ZERO headers remain under BOTH raw and normalized patterns (train and test).
7. Conflicting-label texts within train: ALL copies dropped (117 texts / 1,021 rows,
   conservative). Then dedup within train (924 rows; disjoint from conflict set by
   construction: dedup ran on the post-conflict frame).
8. Remove ALL train-test normalized-text overlap from BOTH sides: 240 texts
   (240 train rows, 370 test rows; test side 336 same-label, 34 conflicting-label).
   Assert zero cross-split overlap.
Final: train 80,565 / test 22,488.

## Frozen config (freeze2.json, hashes asserted again before fit)
word TF-IDF (1,2) min_df 3 max 150k sublinear; char_wb (3,5) min_df 3 max 100k sublinear;
LogisticRegression C=2.0 max_iter 1000 seed 42. Raw/code/text/label sha256 in freeze2.json.

## Result (single eval on the exposed-test remainder - benchmark reuse, NOT fresh validation)
accuracy 0.8806, F1 0.8843, AUC 0.9458 (n_test 22,488).
The test remainder is development-exposed: these numbers are a development check on a
reused benchmark, not independent confirmation. The corrected clean number is lower than
the provisional 88.85 because the remaining 9,845 leaked headers are actually removed now.

## Inference
from header_re import preprocess  # raw header(s) -> normalize -> normalized header(s)
# The included FastAPI app applies this preprocessing before prediction.
Bundle: model_v3_repaired2.joblib {vec, vec_c, lr, config, preprocessing, raw_header_re, freeze}.

## Rights note
Raw review data rights remain unresolved; package contains code, config, metrics and the
model only - no raw reviews. No dataset rights or redistribution permission are granted by this repository.


## Run the public package

```bash
pip install -r requirements.txt
python test_header_re.py  # 44 preprocessing assertions, no raw data required
uvicorn src.app:app --port 8000
```

The included model is the corrected round-2 artifact, not the withdrawn v3 model.
Earlier 94.62% accuracy was inflated by rating-header label leakage and is withdrawn.
The app uses uncalibrated logistic-regression scores, not guaranteed probabilities.

For reproduction, obtain the HARD balanced reviews from
https://github.com/elnagara/HARD-Arabic-Dataset (data/balanced-reviews.zip),
unpack balanced-reviews.txt into data/, then run `python repair2.py prepare`.
This checks the raw-file SHA-256 and recreates the cleaned split. `python repair2.py train`
refits the frozen configuration; no refit is needed for inference or the preprocessing tests.
The historical comparison with an earlier private split artifact is omitted from
this public script; the fixed-seed split and zero-overlap assertions remain.

## Attribution

Portfolio project of Ahmed Khalifa, packaged and corrected with AI-assistant help.
Dataset: Elnagar et al. (2018), HARD. Raw reviews are not redistributed.
