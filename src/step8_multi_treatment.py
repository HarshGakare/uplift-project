"""
Step 8: Multi-treatment uplift (no email / men's email / women's email).
Run from the project root:  python src/step8_multi_treatment.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from uplift_utils import qini_coefficient

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")

SEED = 42
POPULATION = 100_000
N_BOOT = 300
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]   # action codes 0, 1, 2
ARM_FILES = {
    "No E-Mail": "model_arm_none.joblib",
    "Mens E-Mail": "model_arm_mens.joblib",
    "Womens E-Mail": "model_arm_womens.joblib",
}

# ---------- 1. Load ----------
train = pd.read_csv(PROC_DIR / "train.csv")
test = pd.read_csv(PROC_DIR / "test.csv")
with open(PROC_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)
with open(OUT_DIR / "step4_best_params.json") as f:
    params = json.load(f)["t_learner"]          # tuned settings from Step 4
with open(OUT_DIR / "business_config.json") as f:
    config = json.load(f)

COST = config["cost_per_email"]
VALUE = config["value_per_buyer"]
THRESHOLD = COST / VALUE
print(f"Break-even uplift: {THRESHOLD:.4f}  (cost ${COST}, profit per extra buyer ${VALUE})")

print("\nTraining customers per group:")
print(train["segment"].value_counts().to_string())

# ---------- 2. Train one model per group ----------
models = {}
for arm in ARMS:
    d = train[train["segment"] == arm]
    m = LGBMClassifier(**params, subsample_freq=1, random_state=SEED, verbose=-1)
    m.fit(d[feature_cols], d["conversion"])
    models[arm] = m
    joblib.dump(m, MODEL_DIR / ARM_FILES[arm])

# ---------- 3. Predict on the test set ----------
X = test[feature_cols]
p_none = models["No E-Mail"].predict_proba(X)[:, 1]
p_mens = models["Mens E-Mail"].predict_proba(X)[:, 1]
p_womens = models["Womens E-Mail"].predict_proba(X)[:, 1]

up_mens = p_mens - p_none
up_womens = p_womens - p_none
print(f"\nMean predicted uplift: men's {up_mens.mean():.4f}, women's {up_womens.mean():.4f}")

# ---------- 4. Per-email Qini check ----------
# Compare each email type against the no-email group, using only those two groups.
print("\nQini per email type (test set, higher is better, 0 = random):")
for arm, col in [("Mens E-Mail", up_mens), ("Womens E-Mail", up_womens)]:
    mask = test["segment"].isin(["No E-Mail", arm]).to_numpy()
    q = qini_coefficient(
        test.loc[mask, "conversion"].to_numpy(),
        (test.loc[mask, "segment"] == arm).astype(int).to_numpy(),
        col[mask],
    )
    print(f"  {arm:14s} {q:6.2f}")

# ---------- 5. Build the six policies ----------
rng = np.random.default_rng(SEED)
n = len(test)

# Policy: multi-treatment model
best_up = np.maximum(up_mens, up_womens)
best_arm = np.where(up_mens >= up_womens, 1, 2)
policy_multi = np.where(best_up > THRESHOLD, best_arm, 0)

# Policy: Step 5 rule (binary model decides email yes/no, email type is random)
bin_t = joblib.load(MODEL_DIR / "tuned_model_treated.joblib")
bin_c = joblib.load(MODEL_DIR / "tuned_model_control.joblib")
bin_score = bin_t.predict_proba(X)[:, 1] - bin_c.predict_proba(X)[:, 1]
random_type = rng.integers(1, 3, n)
policy_step5 = np.where(bin_score > THRESHOLD, random_type, 0)

# Policy: match the email to past purchases (no ML)
only_mens = (test["mens"] == 1) & (test["womens"] == 0)
only_womens = (test["womens"] == 1) & (test["mens"] == 0)
policy_match = np.where(only_mens, 1, np.where(only_womens, 2, rng.integers(1, 3, n)))

policies = {
    "Email nobody": np.zeros(n, dtype=int),
    "Everyone: men's email": np.ones(n, dtype=int),
    "Everyone: women's email": np.full(n, 2),
    "Match past purchases": policy_match,
    "Step 5 rule (random type)": policy_step5,
    "Multi-treatment model": policy_multi,
}

print("\nWhat the multi-treatment model recommends:")
rec = pd.Series(policy_multi).map({0: "Do not email", 1: "Men's email", 2: "Women's email"})
print(rec.value_counts(normalize=True).round(3).to_string())

# ---------- 6. Evaluate policies with IPW ----------
arm_idx = test["segment"].map({a: i for i, a in enumerate(ARMS)}).to_numpy()
conv = test["conversion"].to_numpy()


def policy_gain(policy, idx):
    """Profit gain vs sending nothing, per POPULATION customers (IPW estimate)."""
    a, y, p = arm_idx[idx], conv[idx], policy[idx]
    prop = np.array([(a == k).mean() for k in range(3)])    # chance of each action

    def conv_rate(pol):
        return np.mean((a == pol) * y / prop[a])

    extra_rate = conv_rate(p) - conv_rate(np.zeros_like(p))
    share_emailed = (p != 0).mean()
    extra_buyers = extra_rate * POPULATION
    profit = extra_buyers * VALUE - share_emailed * POPULATION * COST
    return profit, extra_buyers, share_emailed


full = np.arange(n)
rows = []
for name, pol in policies.items():
    profit, buyers, share = policy_gain(pol, full)
    rows.append({"policy": name, "emailed": f"{share:.0%}",
                 "extra_buyers": round(buyers), "profit": round(profit)})
print(f"\nEstimated results per {POPULATION:,} customers (IPW, test set):")
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 7. Bootstrap: how sure are we? ----------
names = list(policies)
boot = np.zeros((N_BOOT, len(names)))
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    for j, name in enumerate(names):
        boot[b, j] = policy_gain(policies[name], idx)[0]

print("\nProfit 90% bootstrap range:")
for j, name in enumerate(names):
    lo, hi = np.percentile(boot[:, j], [5, 95])
    print(f"  {name:28s} ${lo:8,.0f} to ${hi:8,.0f}")

multi_j = names.index("Multi-treatment model")
print("\nHow often the multi-treatment model beats each alternative (bootstrap):")
for j, name in enumerate(names):
    if j != multi_j:
        print(f"  vs {name:28s} {np.mean(boot[:, multi_j] > boot[:, j]):.0%}")

# ---------- 8. Chart ----------
means = [policy_gain(policies[nm], full)[0] for nm in names]
lows = [np.percentile(boot[:, j], 5) for j in range(len(names))]
highs = [np.percentile(boot[:, j], 95) for j in range(len(names))]
err = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]

plt.figure(figsize=(9, 4.8))
plt.barh(names, means, xerr=err, color="#4C78A8", capsize=4)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel(f"Estimated profit gain per {POPULATION:,} customers ($), with 90% range")
plt.title("Targeting policies compared (test set, IPW)")
plt.tight_layout()
plt.savefig(OUT_DIR / "step8_policies.png", dpi=150)
print("\nSaved chart: outputs/step8_policies.png")

# ---------- 9. Save ----------
out = test.copy()
out["uplift_mens"] = up_mens
out["uplift_womens"] = up_womens
out["recommended_action"] = rec.to_numpy()
out.to_csv(PROC_DIR / "test_with_multi_scores.csv", index=False)
print("Saved: models/model_arm_*.joblib and data/processed/test_with_multi_scores.csv")