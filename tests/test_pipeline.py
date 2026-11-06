import json, os, pickle
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

@pytest.fixture(scope="module")
def splits():
    with open(os.path.join(DATA, "train.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(DATA, "test.pkl"), "rb") as f:
        test = pickle.load(f)
    return train, test

def test_no_hotel_overlap(splits):
    train, test = splits
    assert not (set(train["Hotel name"]) & set(test["Hotel name"]))

def test_labels_binary_and_balancedish(splits):
    train, test = splits
    for df in (train, test):
        assert set(df["label"].unique()) <= {0, 1}
        assert 0.4 < df["label"].mean() < 0.6

def test_text_normalized(splits):
    train, _ = splits
    sample = train["text"].head(500)
    assert not sample.str.contains("[ً-ْ]").any(), "diacritics remain"
    assert not sample.str.contains("ـ").any(), "tatweel remains"
    assert not sample.str.contains("[أإآ]").any(), "alef variants remain"

def test_metrics_file_schema():
    with open(os.path.join(DATA, "metrics.json")) as f:
        m = json.load(f)
    assert "results" in m and "limitations" in m
    for name in ("majority_dummy", "multinomial_nb", "logistic_regression"):
        assert name in m["results"], name
        assert 0.0 <= m["results"][name]["accuracy"] <= 1.0

def test_saved_model_predicts():
    import joblib
    from src.prepare import normalize
    bundle = joblib.load(os.path.join(DATA, "model.joblib"))
    def proba(s):
        from scipy.sparse import hstack
        X = hstack([bundle["vectorizer"].transform([normalize(s)]),
                    bundle["vectorizer_char"].transform([normalize(s)])]).tocsr()
        return float(bundle["model"].predict_proba(X)[0, 1])
    assert proba("فندق رائع جدا والخدمة ممتازة والنظافة عظيمة") > 0.5
    assert proba("اسوأ فندق في حياتي قذر والخدمة سيئة جدا") < 0.5
