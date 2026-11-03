"""
Step 5: Turn uplift scores into a profit-based targeting decision.
Run from the project root:  python src/step5_business_decision.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
SCORE_COL = "score_t_learner_tuned"

# ---------- Business assumptions (change these) ----------
PROFIT_MARGIN = 0.30     # share of a buyer's spend that is profit
COST_PER_EMAIL = 0.10    # dollars per email sent
POPULATION = 100_000     # we report results "per 100,000 customers"

# ---------- 1. Load ----------
train = pd.read_csv(PROC_DIR / "train.csv")
test = pd.read_csv(PROC_DIR / "test_with_tuned_scores.csv")

avg_spend_of_buyers = train.loc[train["conversion"] == 1, "spend"].mean()
value_per_buyer = avg_spend_of_buyers * PROFIT_MARGIN
print(f"Average spend of a buyer: ${avg_spend_of_buyers:.2f}")
print(f"Profit per extra buyer:   ${value_per_buyer:.2f}")


# ---------- 2. Helper functions ----------
def incremental_rate(d):
    """Conversion rate with email minus conversion rate without email."""
    t = d.loc[d["treatment"] == 1, "conversion"]
    c = d.loc[d["treatment"] == 0, "conversion"]
    if len(t) == 0 or len(c) == 0:
        return 0.0
    return t.mean() - c.mean()


def profit_of_group(target, n_total, cost, value):
    """Profit per POPULATION customers if we email only the `target` group."""
    share = len(target) / n_total
    emails = share * POPULATION
    extra_buyers = incremental_rate(target) * emails
    profit = extra_buyers * value - emails * cost
    return profit, extra_buyers, emails


def profit_curve(df_sorted, cost, value, fracs):
    n = len(df_sorted)
    return np.array(
        [
            profit_of_group(df_sorted.iloc[: int(round(f * n))], n, cost, value)[0]
            for f in fracs
        ]
    )


test_sorted = test.sort_values(SCORE_COL, ascending=False).reset_index(drop=True)
fracs = np.arange(0.05, 1.0001, 0.05)
n = len(test_sorted)
overall_rate = incremental_rate(test_sorted)

# ---------- 3. Main scenario ----------
threshold = COST_PER_EMAIL / value_per_buyer
print(f"\nBreak-even rule: email if predicted uplift > {threshold:.4f}")

rule_mask = test_sorted[SCORE_COL] > threshold
rule_group = test_sorted[rule_mask]
all_group = test_sorted

p_rule, b_rule, e_rule = profit_of_group(rule_group, n, COST_PER_EMAIL, value_per_buyer)
p_all, b_all, e_all = profit_of_group(all_group, n, COST_PER_EMAIL, value_per_buyer)

print(f"\nPer {POPULATION:,} customers (cost per email = ${COST_PER_EMAIL}):")
print(
    f"  Email everyone : {e_all:,.0f} emails, {b_all:,.0f} extra buyers, profit ${p_all:,.0f}"
)
print(
    f"  Model rule     : {e_rule:,.0f} emails ({rule_mask.mean():.0%} of customers), "
    f"{b_rule:,.0f} extra buyers, profit ${p_rule:,.0f}"
)
print(f"  Gain from model rule: ${p_rule - p_all:,.0f}")

# ---------- 4. How sure are we? Bootstrap ----------
rng = np.random.default_rng(SEED)
diffs = []
for _ in range(300):
    d = test.iloc[rng.integers(0, n, n)]
    target = d[d[SCORE_COL] > threshold]
    gain = (
        profit_of_group(target, n, COST_PER_EMAIL, value_per_buyer)[0]
        - profit_of_group(d, n, COST_PER_EMAIL, value_per_buyer)[0]
    )
    diffs.append(gain)
lo, hi = np.percentile(diffs, [5, 95])
print(f"\nBootstrap 90% range of the gain over 'email everyone': ${lo:,.0f} to ${hi:,.0f}")
print(f"Model rule beat 'email everyone' in {np.mean(np.array(diffs) > 0):.0%} of resamples")

# ---------- 5. Sensitivity: what if email cost changes? ----------
print("\nSensitivity to email cost:")
rows = []
for cost in [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]:
    thr = cost / value_per_buyer
    grp = test_sorted[test_sorted[SCORE_COL] > thr]
    p_r = profit_of_group(grp, n, cost, value_per_buyer)[0]
    p_a = profit_of_group(test_sorted, n, cost, value_per_buyer)[0]
    rows.append(
        {
            "cost_per_email": cost,
            "threshold": round(thr, 4),
            "share_emailed": round(len(grp) / n, 2),
            "profit_rule": round(p_r),
            "profit_email_all": round(p_a),
            "gain": round(p_r - p_a),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 6. Chart: profit vs how many customers we email ----------
model_curve = profit_curve(test_sorted, COST_PER_EMAIL, value_per_buyer, fracs)
random_curve = (
    overall_rate * fracs * POPULATION * value_per_buyer
    - fracs * POPULATION * COST_PER_EMAIL
)

plt.figure(figsize=(8, 5))
plt.plot(fracs * 100, model_curve, marker="o", label="Model ranking")
plt.plot(fracs * 100, random_curve, "--", label="Random targeting")
plt.axvline(rule_mask.mean() * 100, color="green", linestyle=":", label="Model rule cut-off")
plt.xlabel("% of customers emailed (highest score first)")
plt.ylabel(f"Profit per {POPULATION:,} customers ($)")
plt.title("Profit vs share of customers emailed")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "step5_profit_curve.png", dpi=150)
print("\nSaved chart: outputs/step5_profit_curve.png")

# ---------- 7. Save settings for the Streamlit app (Step 6) ----------
config = {
    "score_column": SCORE_COL,
    "profit_margin": PROFIT_MARGIN,
    "cost_per_email": COST_PER_EMAIL,
    "avg_spend_of_buyers": round(float(avg_spend_of_buyers), 2),
    "value_per_buyer": round(float(value_per_buyer), 2),
    "threshold": round(float(threshold), 5),
    "model_files": ["models/tuned_model_treated.joblib", "models/tuned_model_control.joblib"],
}
with open(OUT_DIR / "business_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Saved: outputs/business_config.json")