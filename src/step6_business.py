"""
Step 6: Business decision layer.
Turns uplift scores into profit, finds the best share of customers to email,
checks it on held-out halves, and saves a policy file for the app.
Run from the project root:  python src/step6_business.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------- ASSUMPTIONS (state these in your README) ----------
COST_PER_EMAIL = 0.10     # dollars to send one email
PROFIT_MARGIN = 0.30      # share of customer spend that is profit
MIN_FRACTION = 0.05       # never choose a cutoff below 5% of customers
N_SPLITS = 20             # repeated split-half checks
N_BOOTSTRAPS = 200

METHOD_TO_COLUMN = {
    "T-learner (Step 3, untuned)": "uplift_score",
    "T-learner (tuned)": "uplift_tuned",
    "Transformed-outcome model": "uplift_transformed",
}

# ---------- 1. Load ----------
test = pd.read_csv(PROCESSED_DIR / "test_scored_v2.csv")

# Use the method that Step 5 found best (change this line to force a method).
try:
    with open(OUT_DIR / "step5_metrics.json") as f:
        BEST_METHOD = json.load(f)["best_method"]
except FileNotFoundError:
    BEST_METHOD = "T-learner (tuned)"
SCORE_COL = METHOD_TO_COLUMN[BEST_METHOD]

N = len(test)
spend = test["spend"].to_numpy(dtype=float)
t = test["treatment"].to_numpy()
y = test["target"].to_numpy()
score = test[SCORE_COL].to_numpy()

print(f"Test customers: {N}")
print(f"Method used: {BEST_METHOD}  (column: {SCORE_COL})")
print(f"Assumptions: cost per email = ${COST_PER_EMAIL:.2f}, "
      f"profit margin = {PROFIT_MARGIN:.0%}")


# ---------- 2. Helper functions ----------
def profit_curve(spend, t, score, cost, margin):
    """
    Profit if we email the top-k customers (best score first), for every k.
    Returns an array where index k = profit when emailing k customers.
    """
    order = np.argsort(-score, kind="mergesort")
    s, tt = spend[order], t[order]

    n_t = np.cumsum(tt)
    n_c = np.cumsum(1 - tt)
    sum_t = np.cumsum(s * tt)
    sum_c = np.cumsum(s * (1 - tt))
    k = np.arange(1, len(s) + 1)

    mean_t = np.divide(sum_t, n_t, out=np.zeros(len(s)), where=n_t > 0)
    mean_c = np.divide(sum_c, n_c, out=np.zeros(len(s)), where=n_c > 0)

    # Extra spend per customer caused by the email x number of customers emailed
    extra_spend = np.where((n_t > 0) & (n_c > 0), k * (mean_t - mean_c), 0.0)
    profit = extra_spend * margin - cost * k
    return np.r_[0.0, profit]


def best_k(profit, n):
    """Best number of customers to email. 0 means 'do not email anyone'."""
    lo = max(1, int(MIN_FRACTION * n))
    candidate = lo + int(np.argmax(profit[lo:]))
    return candidate if profit[candidate] > 0 else 0


def per_1000(value, n):
    return value / n * 1000


# ---------- 3. Profit curve on the full test set ----------
profit = profit_curve(spend, t, score, COST_PER_EMAIL, PROFIT_MARGIN)
k_best = best_k(profit, N)
frac_best = k_best / N
cutoff_score = float(np.sort(score)[::-1][k_best - 1]) if k_best > 0 else None

print("\n--- Best policy on the full test set (optimistic: cutoff chosen here) ---")
print(f"Best share to email : {frac_best:.0%}  ({k_best} customers)")
print(f"Profit per 1,000 customers: ${per_1000(profit[k_best], N):,.2f}")
print(f"Email everyone      : ${per_1000(profit[-1], N):,.2f} per 1,000 customers")
print(f"Share of customers with negative predicted uplift: "
      f"{(score < 0).mean():.0%}")

# ---------- 4. Compare policies at the same budget ----------
rng = np.random.default_rng(RANDOM_STATE)
policy_profit = {}

policy_profit["Uplift model"] = profit[k_best]
baseline_curve = profit_curve(
    spend, t, test["propensity_score"].to_numpy(), COST_PER_EMAIL, PROFIT_MARGIN
)
policy_profit["Baseline buy-model"] = baseline_curve[k_best]

random_values = [
    profit_curve(spend, t, rng.random(N), COST_PER_EMAIL, PROFIT_MARGIN)[k_best]
    for _ in range(20)
]
policy_profit["Random selection"] = float(np.mean(random_values))
policy_profit["Email everyone"] = profit[-1]
policy_profit["Email nobody"] = 0.0

print(f"\nProfit per 1,000 customers (all emailing {frac_best:.0%} unless noted):")
for name, value in policy_profit.items():
    print(f"  {name:20s} ${per_1000(value, N):10,.2f}")

# ---------- 5. Honest check: choose on one half, measure on the other ----------
strat = test["treatment"].astype(str) + "_" + test["target"].astype(str)
held_out, chosen_fracs, everyone_b = [], [], []

for seed in range(N_SPLITS):
    idx_a, idx_b = train_test_split(
        np.arange(N), test_size=0.5, random_state=seed, stratify=strat
    )
    p_a = profit_curve(spend[idx_a], t[idx_a], score[idx_a],
                       COST_PER_EMAIL, PROFIT_MARGIN)
    k_a = best_k(p_a, len(idx_a))
    frac_a = k_a / len(idx_a)

    p_b = profit_curve(spend[idx_b], t[idx_b], score[idx_b],
                       COST_PER_EMAIL, PROFIT_MARGIN)
    k_b = int(frac_a * len(idx_b))
    held_out.append(per_1000(p_b[k_b], len(idx_b)))
    everyone_b.append(per_1000(p_b[-1], len(idx_b)))
    chosen_fracs.append(frac_a)

held_out = np.array(held_out)
everyone_b = np.array(everyone_b)
print("\n--- Honest check (cutoff chosen on one half, profit measured on the other) ---")
print(f"Chosen share to email across splits: "
      f"{np.mean(chosen_fracs):.0%} (range {np.min(chosen_fracs):.0%} "
      f"to {np.max(chosen_fracs):.0%})")
print(f"Held-out profit per 1,000, uplift policy: "
      f"${held_out.mean():,.2f}  (range ${held_out.min():,.2f} "
      f"to ${held_out.max():,.2f})")
print(f"Held-out profit per 1,000, email everyone: "
      f"${everyone_b.mean():,.2f}")
print(f"Uplift policy beats 'email everyone' in "
      f"{np.mean(held_out > everyone_b):.0%} of splits.")

# ---------- 6. Bootstrap: uplift policy minus 'email everyone' ----------
diffs = []
for _ in range(N_BOOTSTRAPS):
    idx = rng.integers(0, N, N)
    p_b = profit_curve(spend[idx], t[idx], score[idx],
                       COST_PER_EMAIL, PROFIT_MARGIN)
    diffs.append(per_1000(p_b[k_best] - p_b[-1], N))
diffs = np.array(diffs)
low, high = np.percentile(diffs, [2.5, 97.5])
print(f"\nBootstrap: uplift policy minus 'email everyone' per 1,000 customers: "
      f"${diffs.mean():,.2f}  95% interval [${low:,.2f}, ${high:,.2f}]")
print("(If the interval includes 0, the advantage is not certain. Say so.)")

# ---------- 7. Sensitivity: what if costs or margins are different? ----------
print("\nSensitivity: best share of customers to email "
      "(0% = do not email anyone)")
costs = [0.05, 0.10, 0.25, 0.50, 1.00]
margins = [0.20, 0.30, 0.50]
grid = pd.DataFrame(index=[f"cost ${c:.2f}" for c in costs],
                    columns=[f"margin {m:.0%}" for m in margins])
for c in costs:
    for m in margins:
        p = profit_curve(spend, t, score, c, m)
        grid.loc[f"cost ${c:.2f}", f"margin {m:.0%}"] = f"{best_k(p, N) / N:.0%}"
print(grid.to_string())

# ---------- 8. Chart ----------
sns.set_theme(style="whitegrid")
x_pct = np.arange(N + 1) / N * 100

plt.figure(figsize=(10, 5.5))
plt.plot(x_pct, per_1000(profit, N), label="Uplift model", linewidth=2)
plt.plot(x_pct, per_1000(baseline_curve, N), label="Baseline buy-model",
         linewidth=2)
random_curve = np.mean(
    [profit_curve(spend, t, rng.random(N), COST_PER_EMAIL, PROFIT_MARGIN)
     for _ in range(10)], axis=0)
plt.plot(x_pct, per_1000(random_curve, N), label="Random", color="gray",
         linewidth=2)
plt.axhline(0, color="black", linewidth=0.8)
if k_best > 0:
    plt.axvline(frac_best * 100, color="green", linestyle="--",
                label=f"Best cutoff ({frac_best:.0%})")
plt.title(f"Profit per 1,000 customers vs. share emailed "
          f"(cost ${COST_PER_EMAIL:.2f}, margin {PROFIT_MARGIN:.0%})")
plt.xlabel("% of customers emailed (highest score first)")
plt.ylabel("Profit per 1,000 customers ($)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "step6_profit_curve.png", dpi=150)

# ---------- 9. Save the policy for the Streamlit app ----------
policy = {
    "method": BEST_METHOD,
    "score_column": SCORE_COL,
    "cost_per_email": COST_PER_EMAIL,
    "profit_margin": PROFIT_MARGIN,
    "best_fraction": frac_best,
    "score_cutoff": cutoff_score,
    "profit_per_1000_full_test": per_1000(profit[k_best], N),
    "profit_per_1000_email_everyone": per_1000(profit[-1], N),
    "held_out_profit_per_1000_mean": float(held_out.mean()),
    "held_out_profit_per_1000_min": float(held_out.min()),
    "held_out_profit_per_1000_max": float(held_out.max()),
    "bootstrap_diff_vs_everyone_ci": [float(low), float(high)],
}
with open(OUT_DIR / "policy.json", "w") as f:
    json.dump(policy, f, indent=2)

# ---------- 10. A sentence for your README ----------
print("\n--- Draft README sentence (check the numbers before using) ---")
print(
    f"Assuming ${COST_PER_EMAIL:.2f} per email and a {PROFIT_MARGIN:.0%} margin, "
    f"emailing the top {frac_best:.0%} of customers by predicted uplift gave "
    f"about ${held_out.mean():,.0f} profit per 1,000 customers on held-out data, "
    f"versus about ${everyone_b.mean():,.0f} for emailing everyone."
)

print("\nSaved: outputs/step6_profit_curve.png, outputs/policy.json")