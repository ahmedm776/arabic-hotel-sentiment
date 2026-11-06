"""Confusion matrix + score distribution on the frozen test split."""
import os, pickle, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
bundle = joblib.load(os.path.join(DATA, "model.joblib"))
with open(os.path.join(DATA, "train.pkl"), "rb") as f:
    train = pickle.load(f)
with open(os.path.join(DATA, "test.pkl"), "rb") as f:
    test = pickle.load(f)
test = test[~test["text"].isin(set(train["text"]))].reset_index(drop=True)  # same dedup as train.py
X = bundle["vectorizer"].transform(test["text"])
y = test["label"].values
p = bundle["model"].predict_proba(X)[:, 1]
pred = (p >= 0.5).astype(int)
cm = confusion_matrix(y, pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13)
ax.set_xticks([0, 1], ["pred neg", "pred pos"]); ax.set_yticks([0, 1], ["true neg", "true pos"])
ax.set_title("Confusion matrix - held-out hotels (n=22,871)")
ax = axes[1]
ax.hist(p[y == 0], bins=50, alpha=0.6, label="true negative", density=True)
ax.hist(p[y == 1], bins=50, alpha=0.6, label="true positive", density=True)
ax.set_xlabel("predicted P(positive)"); ax.set_title("Score separation by true label")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "figures", "evaluation.png"), dpi=140)
print("figure written")
