"""
Step 9: Out-of-fold scores on all 64,000 customers, repeated over seeds,
with a visit-based cross-check and a calibration check.
Run from the project root:  python src/step9_robustness.py
(Takes a couple of minutes.)
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs")

SEEDS = [42, 7, 2024]
N_FOLDS = 5
N_BOOT = 300
POPULATION = 100_000
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]   # action codes 0, 1, 2

# ---------- 1. Load everything (train + test together) ----------
train = pd.read_csv(PROC_DIR / "train.csv")
test = pd.read_csv(PROC_DIR / "test.csv")
train["from_test_split"] = 0
test["from_test_split"] = 1
data = pd.concat([train, test], ignore_index=True)
n = len(data)

with open(PROC_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)
with open(OUT_DIR / "step4_best_params.json") as f:
    params = json.load(f)["t_learner"]
with open(OUT_DIR / "business_config.json") as f:
    config = json.load(f)

COST = config["cost_per_email"]
VALUE = config["value_per_buyer"]
THRESHOLD = COST / VALUE

arm = data["segment"].map({a: i for i, a in enumerate(ARMS)}).to_numpy()
conv = data["conversion"].to_numpy()
visit = data["visit"].to_numpy()
strata = data["segment"] + "_" + data["conversion"].astype(str)

print(f"Customers: {n:,} | buyers: {conv.sum()} | visitors: {visit.sum():,}")
print(f"Break-even uplift: {THRESHOLD:.4f}")


# ---------- 2. Out-of-fold uplift scores ----------
def oof_uplift(seed):
    """Every customer is scored by models that never saw that customer."""
    up = np.zeros((n, 2))            # column 0 = men's uplift, column 1 = women's uplift
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(data, strata):
        tr, va = data.iloc[tr_idx], data.iloc[va_idx]
        p = {}
        for k, a in enumerate(ARMS):
            d = tr[tr["segment"] == a]
            m = LGBMClassifier(**params, subsample_freq=1, random_state=seed, verbose=-1)
            m.fit(d[feature_cols], d["conversion"])
            p[k] = m.predict_proba(va[feature_cols])[:, 1]
        up[va_idx, 0] = p[1] - p[0]
        up[va_idx, 1] = p[2] - p[0]
    return up


def decide(up):
    """Pick the better email, but only if its uplift beats the break-even."""
    best_arm = np.where(up[:, 0] >= up[:, 1], 1, 2)
    return np.where(up.max(axis=1) > THRESHOLD, best_arm, 0)


print("\nComputing out-of-fold scores ...")
oof = {}
for s in SEEDS:
    oof[s] = oof_uplift(s)
    print(f"  seed {s} done")

up_avg = np.mean([oof[s] for s in SEEDS], axis=0)     # average over seeds

# ---------- 3. Policy evaluation (IPW) ----------
def evaluate(policy, idx):
    """Estimated gain vs sending nothing, per POPULATION customers."""
    a, p = arm[idx], policy[idx]
    prop = np.array([(a == k).mean() for k in range(3)])   # chance of each action

    def rate(outcome, pol):
        return np.mean((a == pol) * outcome[idx] / prop[a])

    zero = np.zeros_like(p)
    buyers = (rate(conv, p) - rate(conv, zero)) * POPULATION
    visits = (rate(visit, p) - rate(visit, zero)) * POPULATION
    share = (p != 0).mean()
    profit = buyers * VALUE - share * POPULATION * COST
    return {"buyers": buyers, "visits": visits, "share": share, "profit": profit}


rng = np.random.default_rng(0)
only_mens = ((data["mens"] == 1) & (data["womens"] == 0)).to_numpy()
only_womens = ((data["womens"] == 1) & (data["mens"] == 0)).to_numpy()
policy_match = np.where(only_mens, 1, np.where(only_womens, 2, rng.integers(1, 3, n)))

fixed = {
    "Email nobody": np.zeros(n, dtype=int),
    "Everyone: men's email": np.ones(n, dtype=int),
    "Everyone: women's email": np.full(n, 2),
    "Match past purchases": policy_match,
}
full = np.arange(n)

# ---------- 4. Results per seed ----------
print("\nMulti-treatment model, one row per seed (all 64,000 customers, per 100k):")
rows = []
for s in SEEDS:
    r = evaluate(decide(oof[s]), full)
    rows.append({"seed": s, "emailed": f"{r['share']:.0%}",
                 "extra_buyers": round(r["buyers"]), "profit": round(r["profit"])})
print(pd.DataFrame(rows).to_string(index=False))
seed_profits = [row["profit"] for row in rows]
print(f"Profit across seeds: mean ${np.mean(seed_profits):,.0f}, std ${np.std(seed_profits):,.0f}")

# ---------- 5. Comparison on the seed-averaged policy ----------
policies = dict(fixed)
policies["Multi-treatment (OOF)"] = decide(up_avg)

print("\nPolicy comparison (seed-averaged OOF scores, all customers, per 100k):")
rows = []
for name, pol in policies.items():
    r = evaluate(pol, full)
    rows.append({"policy": name, "emailed": f"{r['share']:.0%}",
                 "extra_buyers": round(r["buyers"]),
                 "extra_visits": round(r["visits"]),
                 "profit": round(r["profit"])})
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 6. Bootstrap ----------
names = list(policies)
boot = np.zeros((N_BOOT, len(names)))
brng = np.random.default_rng(42)
for b in range(N_BOOT):
    idx = brng.integers(0, n, n)
    for j, nm in enumerate(names):
        boot[b, j] = evaluate(policies[nm], idx)["profit"]

print("\nProfit 90% bootstrap range:")
for j, nm in enumerate(names):
    lo, hi = np.percentile(boot[:, j], [5, 95])
    print(f"  {nm:26s} ${lo:8,.0f} to ${hi:8,.0f}")

mj = names.index("Multi-treatment (OOF)")
print("\nHow often the multi-treatment model beats each alternative:")
for nm in ["Everyone: men's email", "Everyone: women's email", "Match past purchases"]:
    print(f"  vs {nm:26s} {np.mean(boot[:, mj] > boot[:, names.index(nm)]):.0%}")

# ---------- 7. Check on rows that tuning never touched ----------
# Optuna (Step 4) used only the training part. Rows from the original test
# split were never used for tuning, so this is the cleanest subset.
idx_test = np.where(data["from_test_split"].to_numpy() == 1)[0]
print(f"\nOnly the {len(idx_test):,} customers from the original test split (per 100k):")
for nm in ["Everyone: men's email", "Match past purchases", "Multi-treatment (OOF)"]:
    r = evaluate(policies[nm], idx_test)
    print(f"  {nm:26s} profit ${r['profit']:8,.0f}   extra buyers {r['buyers']:5.0f}")

# ---------- 8. Calibration + visit cross-check ----------
prop_all = np.array([(arm == k).mean() for k in range(3)])
best_up = up_avg.max(axis=1)
rec = np.where(up_avg[:, 0] >= up_avg[:, 1], 1, 2)     # recommended email type
rank_bin = pd.qcut(
    pd.Series(best_up).rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]
).to_numpy().astype(int)


def actual_effect(outcome, mask):
    """Effect of the recommended email vs no email, from randomized data (IPW)."""
    a, r, o = arm[mask], rec[mask], outcome[mask]
    return np.mean((a == r) * o / prop_all[r]) - np.mean((a == 0) * o / prop_all[0])


cal_rows = []
for b in range(1, 6):
    m = rank_bin == b
    cal_rows.append({
        "group": b,
        "customers": int(m.sum()),
        "predicted_buy_uplift": best_up[m].mean(),
        "actual_buy_uplift": actual_effect(conv, m),
        "actual_visit_uplift": actual_effect(visit, m),
    })
cal = pd.DataFrame(cal_rows)
print("\nCalibration table (group 1 = customers the model likes most):")
print(cal.round(4).to_string(index=False))

pol = policies["Multi-treatment (OOF)"]
emailed = pol != 0
pred_buyers = best_up[emailed].sum() / n * POPULATION
actual_buyers = evaluate(pol, full)["buyers"]
print(f"\nExtra buyers per 100k under the model's policy: "
      f"predicted {pred_buyers:.0f} vs estimated from randomized data {actual_buyers:.0f}")
print(f"Ratio actual / predicted: {actual_buyers / pred_buyers:.2f}  "
      f"(1.0 = scores have the right size; below 1 = overstated)")

# ---------- 9. Chart ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
x = cal["group"].to_numpy()
axes[0].bar(x - 0.2, cal["predicted_buy_uplift"], 0.4, label="Predicted")
axes[0].bar(x + 0.2, cal["actual_buy_uplift"], 0.4, label="Actual (randomized data)")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Group (1 = highest predicted uplift)")
axes[0].set_ylabel("Uplift in purchase rate")
axes[0].set_title("Purchases: predicted vs actual")
axes[0].legend()

axes[1].bar(x, cal["actual_visit_uplift"], color="#59A14F")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Group (1 = highest predicted PURCHASE uplift)")
axes[1].set_ylabel("Uplift in visit rate")
axes[1].set_title("Visits: does the same ranking hold?")
plt.tight_layout()
plt.savefig(OUT_DIR / "step9_robustness.png", dpi=150)
print("\nSaved chart: outputs/step9_robustness.png")

# ---------- 10. Save ----------
out = data.copy()
out["uplift_mens_oof"] = up_avg[:, 0]
out["uplift_womens_oof"] = up_avg[:, 1]
out["recommended"] = pd.Series(pol).map({0: "Do not email", 1: "Men's email", 2: "Women's email"}).to_numpy()
out.to_csv(PROC_DIR / "oof_scores.csv", index=False)
print("Saved: data/processed/oof_scores.csv")