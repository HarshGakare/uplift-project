"""
Step 10: Cost sweep, shrunken targeting policy, final models.
Run from the project root:  python src/step10_cost_and_shrinkage.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")

SEED = 42
POPULATION = 100_000
N_BOOT = 200
LAMBDA = 0.5                       # fixed in advance (Step 9 calibration ratio was 0.56)
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
COSTS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]   # action codes 0, 1, 2
SHRUNK = f"Shrunken model (x{LAMBDA})"

# ---------- 1. Load ----------
data = pd.read_csv(PROC_DIR / "oof_scores.csv")
n = len(data)
with open(PROC_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)
with open(OUT_DIR / "step4_best_params.json") as f:
    params = json.load(f)["t_learner"]
with open(OUT_DIR / "business_config.json") as f:
    config = json.load(f)
VALUE = config["value_per_buyer"]

arm = data["segment"].map({a: i for i, a in enumerate(ARMS)}).to_numpy()
conv = data["conversion"].to_numpy()
up = data[["uplift_mens_oof", "uplift_womens_oof"]].to_numpy()   # out-of-fold scores

# Average effect of each email, straight from the randomized data
rate = np.array([conv[arm == k].mean() for k in range(3)])
g = np.array([rate[1] - rate[0], rate[2] - rate[0]])
print(f"Average effect from randomized data: men's {g[0]:.4f}, women's {g[1]:.4f} "
      f"(= {g[0]*POPULATION:.0f} and {g[1]*POPULATION:.0f} extra buyers per 100k)")
print(f"Profit per extra buyer: ${VALUE}")


# ---------- 2. Policies ----------
def shrink(scores, lam):
    """Pull scores toward the average effect of each email."""
    return g + lam * (scores - scores.mean(axis=0))


def decide(scores, threshold):
    """Better email of the two, but only if its score beats the break-even."""
    best_arm = np.where(scores[:, 0] >= scores[:, 1], 1, 2)
    return np.where(scores.max(axis=1) > threshold, best_arm, 0)


def make_policies(cost):
    thr = cost / VALUE
    return {
        "Email nobody": np.zeros(n, dtype=int),
        "Everyone: men's email": np.ones(n, dtype=int),
        "Raw model": decide(up, thr),
        SHRUNK: decide(shrink(up, LAMBDA), thr),
    }


# ---------- 3. Evaluation (IPW) and bootstrap ----------
def estimate(policy, cost):
    """Estimated gain vs sending nothing, per POPULATION customers."""
    prop = np.array([(arm == k).mean() for k in range(3)])
    r = np.mean((arm == policy) * conv / prop[arm])
    base = np.mean((arm == 0) * conv / prop[arm])
    share = (policy != 0).mean()
    buyers = (r - base) * POPULATION
    return buyers * VALUE - share * POPULATION * cost, buyers, share


def bootstrap(policies, cost):
    """Same resamples every call (same seed), so comparisons are paired."""
    rng = np.random.default_rng(SEED)
    names = list(policies)
    out = np.zeros((N_BOOT, len(names)))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a, y = arm[idx], conv[idx]
        prop = np.array([(a == k).mean() for k in range(3)])
        base = np.mean((a == 0) * y / prop[a])
        for j, nm in enumerate(names):
            p = policies[nm][idx]
            r = np.mean((a == p) * y / prop[a])
            share = (p != 0).mean()
            out[b, j] = (r - base) * POPULATION * VALUE - share * POPULATION * cost
    return names, out


# ---------- 4. Cost sweep ----------
rows = []
wins = []
for cost in COSTS:
    pols = make_policies(cost)
    names, boot = bootstrap(pols, cost)
    j_men, j_raw, j_shr = (names.index("Everyone: men's email"),
                           names.index("Raw model"), names.index(SHRUNK))
    for j, nm in enumerate(names):
        profit, buyers, share = estimate(pols[nm], cost)
        lo, hi = np.percentile(boot[:, j], [5, 95])
        rows.append({"cost": cost, "policy": nm, "emailed": f"{share:.0%}",
                     "extra_buyers": round(buyers), "profit": round(profit),
                     "range_90pct": f"{lo:,.0f} to {hi:,.0f}"})
    wins.append({
        "cost": cost,
        "raw_beats_men_all": f"{np.mean(boot[:, j_raw] > boot[:, j_men]):.0%}",
        "shrunk_beats_men_all": f"{np.mean(boot[:, j_shr] > boot[:, j_men]):.0%}",
        "shrunk_profit_above_0": f"{np.mean(boot[:, j_shr] > 0):.0%}",
        "men_all_profit_above_0": f"{np.mean(boot[:, j_men] > 0):.0%}",
    })

sweep = pd.DataFrame(rows)
print(f"\nCOST SWEEP (all {n:,} customers, out-of-fold scores, per {POPULATION:,}):")
print(sweep.to_string(index=False))
print("\nShare of bootstrap resamples where each statement is true:")
print(pd.DataFrame(wins).to_string(index=False))

# ---------- 5. How much to shrink? (shown for transparency, NOT used to pick) ----------
for cost in (0.10, 0.30):
    thr = cost / VALUE
    pols = {f"lambda={lam}": decide(shrink(up, lam), thr) for lam in LAMBDA_GRID}
    names, boot = bootstrap(pols, cost)
    print(f"\nShrink grid at cost ${cost} (lambda 0 = same average effect for all customers; "
          f"1 = raw scores):")
    grid_rows = []
    for j, nm in enumerate(names):
        profit, buyers, share = estimate(pols[nm], cost)
        lo, hi = np.percentile(boot[:, j], [5, 95])
        grid_rows.append({"policy": nm, "emailed": f"{share:.0%}",
                          "extra_buyers": round(buyers), "profit": round(profit),
                          "range_90pct": f"{lo:,.0f} to {hi:,.0f}"})
    print(pd.DataFrame(grid_rows).to_string(index=False))

# ---------- 6. Chart ----------
fig, ax = plt.subplots(figsize=(8, 4.8))
for nm in ["Email nobody", "Everyone: men's email", "Raw model", SHRUNK]:
    vals = sweep[sweep["policy"] == nm]["profit"].to_numpy()
    ax.plot(COSTS, vals, marker="o", label=nm)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Cost per email ($)")
ax.set_ylabel(f"Estimated profit per {POPULATION:,} customers ($)")
ax.set_title("Which policy wins as emails get more expensive?")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "step10_cost_sweep.png", dpi=150)
print("\nSaved chart: outputs/step10_cost_sweep.png")

# ---------- 7. Final models on ALL customers (method already validated) ----------
final_files = {"none": "final_arm_none.joblib",
               "mens": "final_arm_mens.joblib",
               "womens": "final_arm_womens.joblib"}
final_models = {}
for key, arm_name in zip(["none", "mens", "womens"], ARMS):
    d = data[data["segment"] == arm_name]
    m = LGBMClassifier(**params, subsample_freq=1, random_state=SEED, verbose=-1)
    m.fit(d[feature_cols], d["conversion"])
    final_models[key] = m
    joblib.dump(m, MODEL_DIR / final_files[key])

X = data[feature_cols]
p0 = final_models["none"].predict_proba(X)[:, 1]
final_up = np.column_stack([
    final_models["mens"].predict_proba(X)[:, 1] - p0,
    final_models["womens"].predict_proba(X)[:, 1] - p0,
])
centre = final_up.mean(axis=0)
print(f"\nMean score: out-of-fold {up.mean(axis=0).round(4)} vs final models {centre.round(4)}")

policy_config = {
    "lambda": LAMBDA,
    "global_uplift": {"mens": float(g[0]), "womens": float(g[1])},
    "score_centre": {"mens": float(centre[0]), "womens": float(centre[1])},
    "extra_buyers_per_100k": {"mens": round(float(g[0] * POPULATION)),
                              "womens": round(float(g[1] * POPULATION))},
    "model_files": {k: "models/" + v for k, v in final_files.items()},
    "defaults": {
        "profit_margin": config["profit_margin"],
        "cost_per_email": config["cost_per_email"],
        "avg_spend_of_buyers": config["avg_spend_of_buyers"],
    },
}
with open(OUT_DIR / "policy_config.json", "w") as f:
    json.dump(policy_config, f, indent=2)
print("Saved: models/final_arm_*.joblib and outputs/policy_config.json")