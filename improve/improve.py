"""Resumable hotel-grouped CV over candidate configs. TRAIN ONLY - test untouched.
Each unit = (config, fold); results checkpointed to cv_results.json after each unit.
Selection happens only after all units complete, using CV means.
"""
import json, os, pickle, sys, time
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, "cv_results.json")

AR_MAP = str.maketrans({"\u0623":"\u0627","\u0625":"\u0627","\u0622":"\u0627","\u0671":"\u0627",
                        "\u0649":"\u064a","\u0629":"\u0647","\u0640":""})
def norm(t):
    return " ".join(t.translate(AR_MAP).split())

def word_vec(sublinear=False, normalize=False, maxf=200_000):
    return TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=maxf,
                           sublinear_tf=sublinear, preprocessor=norm if normalize else None)

CONFIGS = {
    "A_baseline":      {"kind":"word", "sub":False, "norm":False, "C":1.0},
    "B_sublinear_C2":  {"kind":"word", "sub":True,  "norm":False, "C":2.0},
    "C_word_plus_char":{"kind":"union","sub":True,  "norm":False, "C":2.0},
    "D_word_C4":       {"kind":"word", "sub":False, "norm":False, "C":4.0},
    "E_norm_C1":       {"kind":"word", "sub":True,  "norm":True,  "C":1.0},
}
N_SPLITS = 3

def featurize(cfg, tr_texts, va_texts, tag):
    cache = os.path.join(OUT, f"feat_{tag}.joblib")
    if os.path.exists(cache):
        return joblib.load(cache)
    if cfg["kind"] == "word":
        v = word_vec(cfg["sub"], cfg["norm"])
        Xtr = v.fit_transform(tr_texts); Xva = v.transform(va_texts)
    else:
        w = word_vec(cfg["sub"], cfg["norm"], 150_000)
        c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3,
                            max_features=100_000, sublinear_tf=True)
        Xtr = hstack([w.fit_transform(tr_texts), c.fit_transform(tr_texts)]).tocsr()
        Xva = hstack([w.transform(va_texts), c.transform(va_texts)]).tocsr()
    joblib.dump((Xtr, Xva), cache)
    return Xtr, Xva

def main():
    budget = float(sys.argv[sys.argv.index("--budget")+1]) if "--budget" in sys.argv else 90
    t0 = time.time()
    train = pickle.load(open(os.path.join(DATA, "train.pkl"), "rb"))
    groups = train["Hotel name"].values
    texts = train["text"].values
    y = train["label"].values
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    gkf = list(GroupKFold(n_splits=N_SPLITS).split(texts, y, groups))
    done = True
    for name, cfg in CONFIGS.items():
        for fi, (tri, vai) in enumerate(gkf):
            key = f"{name}|fold{fi}"
            if key in results:
                continue
            if time.time() - t0 > budget:
                print(f"budget exhausted; {sum(1 for k in results)} units done")
                done = False
                break
            Xtr, Xva = featurize(cfg, texts[tri], texts[vai], f"{name}_{fi}")
            lr = LogisticRegression(max_iter=1000, C=cfg["C"], random_state=42).fit(Xtr, y[tri])
            pred = lr.predict(Xva)
            results[key] = {"accuracy": float(accuracy_score(y[vai], pred)),
                            "f1": float(f1_score(y[vai], pred))}
            json.dump(results, open(RESULTS, "w"), indent=1)
            print(f"{key} acc={results[key]['accuracy']:.4f} f1={results[key]['f1']:.4f}")
            del Xtr, Xva
            cache = os.path.join(OUT, f"feat_{name}_{fi}.joblib")
            if os.path.exists(cache):
                os.remove(cache)  # free disk; unit done
        if not done:
            break
    total = len(CONFIGS) * N_SPLITS
    print(f"units {len(results)}/{total}")
    if len(results) == total:
        print("=== CV SUMMARY (hotel-grouped, train only) ===")
        for name in CONFIGS:
            accs = [results[f"{name}|fold{i}"]["accuracy"] for i in range(N_SPLITS)]
            f1s = [results[f"{name}|fold{i}"]["f1"] for i in range(N_SPLITS)]
            print(f"{name}: acc {np.mean(accs):.4f}+-{np.std(accs):.4f}  f1 {np.mean(f1s):.4f}")
        print("PHASE_DONE")

if __name__ == "__main__":
    main()
