"""Load HARD balanced reviews, normalize, map labels, group-split by hotel."""
import os, re, pickle
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "balanced-reviews.txt")

DIACRITICS = re.compile(r"[ً-ْٰ]")  # tanween, harakat, etc.
TATWEEL = "ـ"

def normalize(text):
    t = DIACRITICS.sub("", str(text))
    t = t.replace(TATWEEL, "")
    t = re.sub("[أإآٱ]", "ا", t)
    t = re.sub("ى", "ي", t)
    t = re.sub("ؤ", "و", t)
    t = re.sub("ئ", "ي", t)
    return t.strip()

def main():
    if not os.path.exists(RAW):
        raise SystemExit(f"Input file missing: {RAW} - see README.md (Source and license).")
    df = pd.read_csv(RAW, sep="\t", encoding="utf-16")
    required = {"Hotel name", "rating", "review"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing expected columns: {sorted(missing)}")
    bad = set(df["rating"].unique()) - {1, 2, 4, 5}
    if bad:
        raise SystemExit(f"Unexpected ratings present: {sorted(bad)} - the balanced subset should only contain 1, 2, 4, 5.")
    df = df.dropna(subset=["review", "Hotel name", "rating"]).copy()
    df["label"] = (df["rating"] >= 4).astype(int)
    df["text"] = df["review"].map(normalize)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(df, df["label"], groups=df["Hotel name"]))
    train, test = df.iloc[tr], df.iloc[te]
    overlap = set(train["Hotel name"]) & set(test["Hotel name"])
    assert not overlap, f"hotel leakage across split: {len(overlap)} hotels"
    with open(os.path.join(HERE, "..", "data", "train.pkl"), "wb") as f:
        pickle.dump(train[["text", "label", "Hotel name"]], f)
    with open(os.path.join(HERE, "..", "data", "test.pkl"), "wb") as f:
        pickle.dump(test[["text", "label", "Hotel name"]], f)
    print(f"train {len(train):,} ({train['label'].mean():.3f} pos) | test {len(test):,} ({test['label'].mean():.3f} pos) | hotels train {train['Hotel name'].nunique():,} test {test['Hotel name'].nunique():,} overlap 0")

if __name__ == "__main__":
    main()
