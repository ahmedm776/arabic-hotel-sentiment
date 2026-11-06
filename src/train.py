"""Fit TF-IDF + baselines on the training split; evaluate once on the test split."""
import json, os, pickle
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.naive_bayes import MultinomialNB

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def main():
    with open(os.path.join(DATA, "train.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(DATA, "test.pkl"), "rb") as f:
        test = pickle.load(f)

    # Leakage amendment (added after a first evaluation - audit trail in README):
    # drop test reviews whose normalized text duplicates a training review.
    n_test_raw = len(test)
    test = test[~test["text"].isin(set(train["text"]))].reset_index(drop=True)
    n_dup_removed = n_test_raw - len(test)

    # Model selection 2026-11-04 (see improve/): 5 candidate configs compared with
    # 3-fold GroupKFold by hotel on TRAIN ONLY. Winner: word(1,2)+char_wb(3,5) union,
    # sublinear tf, LR C=2.0 (CV acc 0.9460 vs 0.9362 baseline). Test set below is
    # evaluated for the SECOND time overall - this is benchmark reuse, not a fresh
    # unseen evaluation; the first-eval numbers stay in README history.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150_000, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                            max_features=100_000, sublinear_tf=True)
    from scipy.sparse import hstack
    Xtr = hstack([vec.fit_transform(train["text"]), vec_c.fit_transform(train["text"])]).tocsr()   # fit on TRAIN ONLY
    Xte = hstack([vec.transform(test["text"]), vec_c.transform(test["text"])]).tocsr()
    ytr, yte = train["label"].values, test["label"].values

    results = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    results["majority_dummy"] = {"accuracy": float(accuracy_score(yte, dummy.predict(Xte)))}

    nb = MultinomialNB().fit(Xtr, ytr)
    p = nb.predict_proba(Xte)[:, 1]
    results["multinomial_nb"] = {"accuracy": float(accuracy_score(yte, nb.predict(Xte))),
                                 "f1": float(f1_score(yte, nb.predict(Xte))),
                                 "auc": float(roc_auc_score(yte, p))}

    lr = LogisticRegression(max_iter=1000, C=2.0, random_state=42).fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    results["logistic_regression"] = {"accuracy": float(accuracy_score(yte, lr.predict(Xte))),
                                      "f1": float(f1_score(yte, lr.predict(Xte))),
                                      "auc": float(roc_auc_score(yte, p)),
                                      "brier": float(brier_score_loss(yte, p)),
                                      "log_loss": float(log_loss(yte, p))}

    meta = {"protocol": "group split by hotel (seed 42), vectorizer fit on train only, evaluation on deduplicated test set",
            "n_train": int(len(train)), "n_test": int(len(test)),
            "test_duplicates_removed": int(n_dup_removed),
            "primary_model": "logistic_regression word+char union, C=2.0 (selected by 3-fold hotel-grouped CV on train only; NB and dummy reported as baselines)",
            "selection_protocol": "candidate configs compared on GroupKFold(n_splits=3) by hotel over the training split only; test evaluated for the second time overall (benchmark reuse, not fresh unseen)",
            "vocabulary": int(len(vec.vocabulary_)) + int(len(vec_c.vocabulary_)),
            "results": results,
            "limitations": ["Reviews from Booking.com, June/July 2016 - not current language",
                            "Label derived from star rating (1-2 neg, 4-5 pos)",
                            "Single frozen split; no cross-validation claim"]}
    with open(os.path.join(DATA, "metrics.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    joblib.dump({"vectorizer": vec, "vectorizer_char": vec_c, "model": lr, "kind": "word+char"},
                os.path.join(DATA, "model.joblib"))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
