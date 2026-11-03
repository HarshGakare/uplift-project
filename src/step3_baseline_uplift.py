"""
Step 3: Baseline uplift model (T-learner with two LightGBM models).
Run from the project root:  python src/step3_baseline_uplift.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ---------- 1. Load ----------
train = pd.read_csv(PROC_DIR / "train.csv")
test = pd.read_csv(PROC_DIR / "test.csv")
with open(PROC_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

treated = train[train["treatment"] == 1]
control = train[train["treatment"] == 0]

# ---------- 2. Train the two models ----------
# Settings are deliberately simple and cautious because only ~1% of
# customers buy. Big, complex trees would just memorise noise.
# We do NOT use class weights: we need honest probabilities,
# because uplift is a difference between two probabilities.
params = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=8,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=42,
    verbose=-1,
)

model_t = LGBMClassifier(**params).fit(treated[feature_cols], treated["conversion"])
model_c = LGBMClassifier(**params).fit(control[feature_cols], control["conversion"])

joblib.dump(model_t, MODEL_DIR / "model_treated.joblib")
joblib.dump(model_c, MODEL_DIR / "model_control.joblib")

# ---------- 3. Predict uplift on the test set ----------
X_test = test[feature_cols]
test = test.copy()
test["p_treated"] = model_t.predict_proba(X_test)[:, 1]
test["p_control"] = model_c.predict_proba(X_test)[:, 1]
test["uplift_score"] = test["p_treated"] - test["p_control"]

print("Uplift score summary:")
print(test["uplift_score"].describe().round(4))

# ---------- 4. Qini curve ----------
def qini_curve(y, t, score):
    """Extra purchases (caused by the email) when targeting the top-k customers."""
    order = np.argsort(-score)
    y = y[order]
    t = t[order]
    n = len(y)

    y_t = np.cumsum(y * t)          # buyers so far, emailed
    y_c = np.cumsum(y * (1 - t))    # buyers so far, not emailed
    n_t = np.cumsum(t)              # customers so far, emailed
    n_c = np.cumsum(1 - t)          # customers so far, not emailed

    ratio = np.divide(n_t, n_c, out=np.zeros(n, dtype=float), where=n_c > 0)
    qini = y_t - y_c * ratio

    x = np.insert(np.arange(1, n + 1) / n, 0, 0.0)
    qini = np.insert(qini, 0, 0.0)
    return x, qini


def area(x, y):
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


y = test["conversion"].to_numpy()
t = test["treatment"].to_numpy()
x, q = qini_curve(y, t, test["uplift_score"].to_numpy())
random_line = x * q[-1]

qini_coef = area(x, q) - area(x, random_line)
print(f"\nQini coefficient (higher is better, 0 = random): {qini_coef:.3f}")

for frac in (0.2, 0.3, 0.5):
    i = int(frac * (len(x) - 1))
    print(
        f"Target top {int(frac*100)}%: extra buyers = {q[i]:.1f} "
        f"vs random targeting = {random_line[i]:.1f} "
        f"(max possible with everyone = {q[-1]:.1f})"
    )

# ---------- 5. Decile table ----------
# Rank customers by score, cut into 10 equal groups (decile 1 = highest score)
test["decile"] = pd.qcut(
    test["uplift_score"].rank(method="first", ascending=False),
    10,
    labels=range(1, 11),
)

rows = []
for d, g in test.groupby("decile", observed=True):
    g_t = g[g["treatment"] == 1]["conversion"]
    g_c = g[g["treatment"] == 0]["conversion"]
    rows.append(
        {
            "decile": d,
            "customers": len(g),
            "avg_predicted_uplift": g["uplift_score"].mean(),
            "conv_emailed": g_t.mean(),
            "conv_not_emailed": g_c.mean(),
            "actual_uplift": g_t.mean() - g_c.mean(),
        }
    )
decile_table = pd.DataFrame(rows)
print("\nDecile table (1 = customers the model thinks respond most):")
print(decile_table.round(4).to_string(index=False))

# ---------- 6. Charts ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(x, q, label="Uplift model")
axes[0].plot(x, random_line, "--", label="Random targeting")
axes[0].set_xlabel("Fraction of customers targeted")
axes[0].set_ylabel("Extra buyers caused by email")
axes[0].set_title("Qini curve (test set)")
axes[0].legend()

axes[1].bar(decile_table["decile"].astype(int), decile_table["actual_uplift"])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Decile (1 = highest predicted uplift)")
axes[1].set_ylabel("Actual uplift in conversion")
axes[1].set_title("Actual uplift by predicted decile")

plt.tight_layout()
plt.savefig(OUT_DIR / "step3_baseline_uplift.png", dpi=150)
print("\nSaved chart: outputs/step3_baseline_uplift.png")

# ---------- 7. Which features matter? ----------
for name, m in [("Model T (emailed)", model_t), ("Model C (not emailed)", model_c)]:
    imp = pd.Series(
        m.booster_.feature_importance(importance_type="gain"), index=feature_cols
    ).sort_values(ascending=False)
    print(f"\nTop 5 features, {name}:")
    print((imp / imp.sum()).head(5).round(3).to_string())

# ---------- 8. Save predictions for later steps ----------
test.to_csv(PROC_DIR / "test_with_baseline_scores.csv", index=False)
print("\nSaved: data/processed/test_with_baseline_scores.csv")