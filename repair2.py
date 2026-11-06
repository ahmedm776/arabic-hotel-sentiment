"""Corrected Arabic preprocessing repair, round 2 (supersedes 20:52 repair).

Order per assignment: load raw -> split (original seed-42 hotel-group assignment on the
raw frame) -> STRIP header on RAW text first -> drop empties -> normalize -> assert zero
headers remain -> conflicting-label texts dropped entirely (train) -> dedup within train
-> remove all train-test normalized text overlap from BOTH sides -> freeze hashes/accounting
BEFORE training -> train once (frozen v3 config, <10min) -> single benchmark-reuse eval.
"""
import os, re, sys, json, time, hashlib
import pandas as pd
from header_re import (RAW_HEADER_RE, NORM_HEADER_RE, RAW_DESCRIPTOR_FORMS,
                       normalize, strip_header_raw, strip_header_raw_all, strip_header_norm_all)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("HARD_RAW_PATH", os.path.join(HERE, "data", "balanced-reviews.txt"))
AUDIT_RAW_SHA = "86c1ae1b41267262d806275df4c0385b7948b18a685c93212ca858aa8df2ab02"
AUDIT_R2_PER_DESCRIPTOR = {"استثنائي": 7294, "رائع": 2551, "جيد جداً": 2552, "جيد جدا": 0,
                           "جيد": 7464, "مقبول": 0, "ضعيف جداً": 5168, "ضعيف جدا": 0,
                           "ضعيف": 6360, "مخيب للأمل": 9521, "ممتاز": 2775}
stage = sys.argv[1] if len(sys.argv) > 1 else "prepare"

def sha(b): return hashlib.sha256(b).hexdigest()
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

if stage == "prepare":
    t0 = time.time()
    log = {}
    assert sha_file(RAW) == AUDIT_RAW_SHA, "raw input differs from audit-r2 input"
    log["raw_sha256"] = AUDIT_RAW_SHA
    df = pd.read_csv(RAW, sep="\t", encoding="utf-16").dropna(subset=["review", "Hotel name", "rating"]).copy()
    df = df.reset_index(drop=True)
    log["rows_loaded"] = int(len(df))
    df["label"] = (pd.to_numeric(df["rating"]) >= 4).astype(int)

    # MASK PARITY with audit r2 on ALL raw rows BEFORE any split/strip
    mask_all = df["review"].astype(str).str.match(RAW_HEADER_RE)
    log["mask_total"] = int(mask_all.sum())
    assert log["mask_total"] == 43685, f"mask parity failed: {log['mask_total']} != 43685"
    per_desc = {d: int(df["review"].astype(str).str.match(re.compile("^" + re.escape(d) + r"\.\s*")).sum())
                for d in RAW_DESCRIPTOR_FORMS}
    assert per_desc == AUDIT_R2_PER_DESCRIPTOR, f"per-descriptor parity failed: {per_desc}"
    log["mask_per_descriptor_matches_audit_r2"] = True

    # ORIGINAL split assignment on the raw frame (identical to v3/audit/previous repair)
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                  .split(df, df["label"], groups=df["Hotel name"]))
    train, test = df.iloc[tr].copy(), df.iloc[te].copy()
    log["split_train"], log["split_test"] = int(len(train)), int(len(test))
    assert (log["split_train"], log["split_test"]) == (82827, 22871), "split sizes drifted"
    assert not (set(train["Hotel name"]) & set(test["Hotel name"])), "hotel leakage"
    log["split_assignment_matches_previous_repair"] = "historical check not repeated; deterministic seed-42 split"

    # STRIP RAW FIRST
    tr_s = train["review"].map(strip_header_raw_all)
    te_s = test["review"].map(strip_header_raw_all)
    train["text_raw_stripped"] = [a for a, _ in tr_s]
    test["text_raw_stripped"] = [a for a, _ in te_s]
    log["train_stripped_descriptor_rows"] = int(sum(1 for _, n in tr_s if n > 0))
    log["test_stripped_descriptor_rows"] = int(sum(1 for _, n in te_s if n > 0))
    assert log["train_stripped_descriptor_rows"] + log["test_stripped_descriptor_rows"] == 43685
    log["rows_with_second_header_also_stripped"] = int(sum(1 for _, n in tr_s if n > 1) + sum(1 for _, n in te_s if n > 1))

    # zero headers remain on RAW after strip (both splits)
    assert int(train["text_raw_stripped"].astype(str).str.match(RAW_HEADER_RE).sum()) == 0, "raw headers remain in train"
    assert int(test["text_raw_stripped"].astype(str).str.match(RAW_HEADER_RE).sum()) == 0, "raw headers remain in test"
    log["assert_zero_raw_headers_after_strip"] = True

    # drop empties (header-only reviews)
    train = train[train["text_raw_stripped"].str.strip().str.len() > 0]
    test = test[test["text_raw_stripped"].str.strip().str.len() > 0]
    log["train_empty_after_strip_dropped"] = log["split_train"] - int(len(train))
    log["test_empty_after_strip_dropped"] = log["split_test"] - int(len(test))

    # THEN normalize, then strip any header written in normalized orthography
    train["text"] = train["text_raw_stripped"].map(normalize)
    test["text"] = test["text_raw_stripped"].map(normalize)
    tr_n = train["text"].map(strip_header_norm_all)
    te_n = test["text"].map(strip_header_norm_all)
    log["train_normalized_orthography_headers_stripped"] = int(sum(1 for _, n in tr_n if n > 0))
    log["test_normalized_orthography_headers_stripped"] = int(sum(1 for _, n in te_n if n > 0))
    train["text"] = [a for a, _ in tr_n]
    test["text"] = [a for a, _ in te_n]
    train = train[train["text"].str.strip().str.len() > 0]
    test = test[test["text"].str.strip().str.len() > 0]
    log["train_empty_after_strip_dropped"] = log["split_train"] - int(len(train))
    log["test_empty_after_strip_dropped"] = log["split_test"] - int(len(test))

    # ZERO-HEADER ASSERTIONS after both layers (normalized AND raw orthography patterns)
    assert int(train["text"].str.match(NORM_HEADER_RE).sum()) == 0, "normalized headers remain in train"
    assert int(test["text"].str.match(NORM_HEADER_RE).sum()) == 0, "normalized headers remain in test"
    assert int(train["text"].str.match(RAW_HEADER_RE).sum()) == 0, "raw-form headers remain in train"
    assert int(test["text"].str.match(RAW_HEADER_RE).sum()) == 0, "raw-form headers remain in test"
    log["assert_zero_headers_after_strip_and_normalize"] = True

    # conflicting-label texts within train: drop ALL copies (conservative)
    g = train.groupby("text")["label"].nunique()
    conflict_texts = set(g[g > 1].index)
    log["train_conflict_texts"] = len(conflict_texts)
    log["train_conflict_rows_dropped"] = int(train["text"].isin(conflict_texts).sum())
    train = train[~train["text"].isin(conflict_texts)]
    # dedup within train (on post-conflict frame -> disjoint by construction)
    n_before = len(train)
    train = train.drop_duplicates(subset=["text"], keep="first")
    log["train_duplicate_rows_dropped"] = int(n_before - len(train))
    log["conflict_disjoint_from_dedup"] = "dedup ran on post-conflict frame; intersection empty by construction"

    # cross-split overlap: remove from BOTH sides
    overlap = set(train["text"]) & set(test["text"])
    ov_te = test[test["text"].isin(overlap)]
    tr_lab = train.drop_duplicates("text").set_index("text")["label"]
    same_lab = int((ov_te["label"] == ov_te["text"].map(tr_lab)).sum())
    log["overlap_texts"] = len(overlap)
    log["overlap_rows_dropped_train"] = int(train["text"].isin(overlap).sum())
    log["overlap_rows_dropped_test"] = int(len(ov_te))
    log["overlap_test_rows_same_label"] = same_lab
    log["overlap_test_rows_conflicting_label"] = int(len(ov_te) - same_lab)
    train = train[~train["text"].isin(overlap)].reset_index(drop=True)
    test = test[~test["text"].isin(overlap)].reset_index(drop=True)
    assert not (set(train["text"]) & set(test["text"])), "cross-split overlap remains"
    log["assert_zero_cross_split_overlap"] = True
    log["final_train"], log["final_test"] = int(len(train)), int(len(test))
    log["prepare_wall_sec"] = round(time.time() - t0, 1)

    train[["text", "label", "Hotel name"]].to_pickle(os.path.join(HERE, "train_final.pkl"))
    test[["text", "label", "Hotel name"]].to_pickle(os.path.join(HERE, "test_final.pkl"))
    freeze = {
        "frozen_at": "2026-11-05T21:02 EET", "stage": "prepare-complete-pre-training",
        "supersedes": "20:52 repair (normalized-before-strip, hamza-on-yaa regex blind spot; provisional 88.85)",
        "pipeline_order": "load raw -> label -> ORIGINAL seed-42 hotel-group split on raw frame -> STRIP header on RAW first (canonical header_re, audit-r2 pattern, repeated for double headers) -> normalize -> strip normalized-orthography headers -> drop empties -> ASSERT zero headers (raw and normalized patterns) -> drop conflicting-label texts entirely (train) -> dedup within train -> remove all train-test overlap BOTH sides -> freeze -> train once (v3 config) -> single benchmark-reuse eval",
        "counts": log,
        "hashes": {
            "raw_sha256": AUDIT_RAW_SHA,
            "header_re_sha256": sha_file(os.path.join(HERE, "header_re.py")),
            "repair2_sha256": sha_file(os.path.abspath(__file__)),
            "train_texts_sha256": sha("\n".join(train["text"]).encode()),
            "train_labels_sha256": sha(train["label"].to_numpy().tobytes()),
            "test_texts_sha256": sha("\n".join(test["text"]).encode()),
            "test_labels_sha256": sha(test["label"].to_numpy().tobytes()),
        },
        "config": {"vec_word": {"ngram_range": [1, 2], "min_df": 3, "max_features": 150000, "sublinear_tf": True},
                   "vec_char": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 3, "max_features": 100000, "sublinear_tf": True},
                   "lr": {"C": 2.0, "max_iter": 1000, "random_state": 42},
                   "split": {"method": "GroupShuffleSplit", "groups": "Hotel name", "test_size": 0.2, "random_state": 42},
                   "source": "HARD balanced-reviews.txt (local, unchanged)"},
    }
    json.dump(freeze, open(os.path.join(HERE, "freeze2.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(log, indent=1, ensure_ascii=False), flush=True)

elif stage == "train":
    import joblib
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    t0 = time.time()
    train = pd.read_pickle(os.path.join(HERE, "train_final.pkl"))
    test = pd.read_pickle(os.path.join(HERE, "test_final.pkl"))
    freeze = json.load(open(os.path.join(HERE, "freeze2.json")))
    assert sha("\n".join(train["text"]).encode()) == freeze["hashes"]["train_texts_sha256"], "train hash drift since freeze"
    assert sha("\n".join(test["text"]).encode()) == freeze["hashes"]["test_texts_sha256"], "test hash drift since freeze"
    cfg = freeze["config"]
    vec = TfidfVectorizer(**{k: tuple(v) if k == "ngram_range" else v for k, v in cfg["vec_word"].items()})
    vec_c = TfidfVectorizer(**{k: tuple(v) if k == "ngram_range" else v for k, v in cfg["vec_char"].items()})
    Xtr = hstack([vec.fit_transform(train["text"]), vec_c.fit_transform(train["text"])]).tocsr()
    Xte = hstack([vec.transform(test["text"]), vec_c.transform(test["text"])]).tocsr()
    lr = LogisticRegression(**cfg["lr"]).fit(Xtr, train["label"].values)
    fit_sec = time.time() - t0
    assert fit_sec < 600, "compute cap exceeded"
    p = lr.predict_proba(Xte)[:, 1]; pred = lr.predict(Xte)
    res = {"protocol": "corrected repair round 2: strip RAW first then normalize (canonical header_re), train dedup, both-side overlap removal; v3 config frozen pre-training; single eval on exposed-test remainder (benchmark reuse, NOT fresh validation)",
           "n_train": int(len(train)), "n_test": int(len(test)),
           "accuracy": float(accuracy_score(test["label"], pred)),
           "f1": float(f1_score(test["label"], pred)),
           "auc": float(roc_auc_score(test["label"], p)),
           "fit_plus_vectorize_sec": round(fit_sec, 1)}
    json.dump(res, open(os.path.join(HERE, "repair2_results.json"), "w"), indent=1)
    joblib.dump({"vec": vec, "vec_c": vec_c, "lr": lr, "config": cfg,
                 "preprocessing": "header_re.strip_header_raw on RAW text, then header_re.normalize",
                 "raw_header_re": RAW_HEADER_RE.pattern,
                 "freeze": freeze["hashes"]},
                os.path.join(HERE, "model_v3_repaired2.joblib"))
    print(json.dumps(res, indent=1), flush=True)
