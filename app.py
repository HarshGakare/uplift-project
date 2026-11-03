"""
Email decision app.
1) Break-even guide from randomized averages (what the evidence supports)
2) Optional customer scoring for an experimental targeted plan
Run from the project root:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))
from features import RAW_REQUIRED, build_features  # noqa: E402

st.set_page_config(page_title="Email Decision Guide", layout="wide")
POP = 100_000


@st.cache_resource
def load_assets():
    with open(ROOT / "outputs" / "policy_config.json") as f:
        cfg = json.load(f)
    models = {k: joblib.load(ROOT / v) for k, v in cfg["model_files"].items()}
    with open(ROOT / "data" / "processed" / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return models, feature_cols, cfg


@st.cache_data
def load_demo(n=2000):
    raw = pd.read_csv(ROOT / "data" / "hillstrom.csv")
    return raw.sample(n, random_state=1).reset_index(drop=True)


models, feature_cols, cfg = load_assets()
lam = cfg["lambda"]
g_m, g_w = cfg["global_uplift"]["mens"], cfg["global_uplift"]["womens"]
mu_m, mu_w = cfg["score_centre"]["mens"], cfg["score_centre"]["womens"]
d = cfg["defaults"]

# Results from validation (Step 10), at the default margin and spend
VALIDATED = pd.DataFrame(
    {
        "Cost per email": ["$0.02", "$0.05", "$0.10", "$0.20", "$0.30", "$0.40"],
        "Email nobody": [0, 0, 0, 0, 0, 0],
        "Men's email to everyone": [21879, 18879, 13879, 3879, -6121, -16121],
        "Targeted (shrunken model)": [19615, 16446, 12255, 4781, -1081, -1201],
        "Targeted beats men's-to-all in": ["10%", "7%", "22%", "66%", "96%", "100%"],
    }
)

# ---------- Title ----------
st.title("Which email should we send?")
st.write(
    "In randomized test data, the **type of email** mattered far more than "
    "**which customer** received it. This app turns that into a decision guide."
)

# ---------- Sidebar ----------
st.sidebar.header("Business settings")
margin = st.sidebar.slider("Profit margin", 0.05, 0.80, float(d["profit_margin"]), 0.01)
cost = st.sidebar.slider("Cost per email ($)", 0.01, 1.00, float(d["cost_per_email"]), 0.01)
avg_spend = st.sidebar.number_input(
    "Average spend of a buyer ($)", min_value=1.0, value=float(d["avg_spend_of_buyers"]), step=5.0
)
value = avg_spend * margin
be_m, be_w = g_m * value, g_w * value
st.sidebar.markdown(f"**Profit per extra buyer:** ${value:,.2f}")

# ---------- Section 1: break-even guide ----------
st.header("1. What the data says")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Men's email: extra buyers per 100k", f"{g_m * POP:,.0f}")
c2.metric("Women's email: extra buyers per 100k", f"{g_w * POP:,.0f}")
c3.metric("Men's email break-even cost", f"${be_m:.2f}")
c4.metric("Women's email break-even cost", f"${be_w:.2f}")
st.caption(
    "Averages from a randomized experiment on 64,000 customers. An email pays for itself "
    "on average when its cost is below its break-even cost."
)

st.header("2. Recommendation at your settings")
profit_men_all = POP * (g_m * value - cost)

if cost >= be_m:
    st.error(
        f"**Do not send either email to everyone.** At ${cost:.2f} per email, even the "
        f"better (men's) email costs more than it earns on average (break-even ${be_m:.2f}). "
        "In validation, the customer-level model did not reliably find a profitable "
        "subset at these costs. The safest plan is to send nothing, or to send only as "
        "a small, measured A/B test."
    )
else:
    st.success(
        f"**Send the men's email to everyone.** Estimated profit: about "
        f"${profit_men_all:,.0f} per 100,000 customers (from randomized averages). "
        f"The women's email "
        + ("also pays for itself on average but earns less." if cost < be_w
           else f"does not pay for itself at this cost (break-even ${be_w:.2f}), so don't send it to everyone.")
    )
    if cost >= 0.75 * be_m:
        st.info(
            "You are close to break-even. This is where customer-level targeting looked most "
            "promising in validation (at $0.20 it beat the simple plan in 66% of "
            "resamples), which is suggestive, not proven. If you have the volume, "
            "A/B test the targeted plan against the simple one before relying on it."
        )

# Profit vs cost chart (arithmetic from the averages, no model scores)
costs = np.linspace(0.0, max(0.5, 1.5 * be_m), 100)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(costs, POP * (g_m * value - costs), label="Men's email to everyone")
ax.plot(costs, POP * (g_w * value - costs), label="Women's email to everyone")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(cost, color="green", linestyle="--", label="Your cost")
ax.set_xlabel("Cost per email ($)")
ax.set_ylabel(f"Estimated profit per {POP:,} customers ($)")
ax.legend()
st.pyplot(fig)

with st.expander("Validation results (out-of-fold, per 100,000 customers)"):
    st.dataframe(VALIDATED, hide_index=True)
    st.caption(
        f"Computed with margin {d['profit_margin']:.0%} and average spend "
        f"${d['avg_spend_of_buyers']}. 'Targeted beats men's-to-all in' is the share of "
        "bootstrap resamples where the targeted plan had higher profit. Above 80% would be "
        "strong evidence; at 96% to 100% (high costs) it only means the targeted plan lost "
        "less money than a plan that loses a lot. Sending nothing was better there."
    )

# ---------- Section 3: optional customer scoring ----------
st.header("3. Optional: score individual customers (experimental)")
st.warning(
    "In validation, raw model scores were about 1.8x too high, and the ranking did not carry "
    "over to visit rates. Scores here are shrunk toward the averages, but customer-level "
    "targeting was **not shown to beat the simple plan**. Use this section to prepare an "
    "A/B test, not as proven advice."
)

source = st.radio("Customer data", ["Demo data (2,000 customers)", "Upload my CSV"], horizontal=True)
raw = None
if source.startswith("Demo"):
    raw = load_demo()
    st.caption("Demo rows come from the training data, so they cannot show accuracy.")
else:
    st.write("Required columns: `" + "`, `".join(RAW_REQUIRED) + "`")
    uploaded = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)

if raw is None:
    st.info("Choose demo data or upload a file to see customer-level scores.")
    st.stop()

try:
    X = build_features(raw, feature_cols)
except Exception as e:
    st.error(f"Could not prepare the data: {e}")
    st.stop()

p0 = models["none"].predict_proba(X)[:, 1]
up_m = models["mens"].predict_proba(X)[:, 1] - p0
up_w = models["womens"].predict_proba(X)[:, 1] - p0

# Shrink toward the average effect of each email
s_m = g_m + lam * (up_m - mu_m)
s_w = g_w + lam * (up_w - mu_w)
best = np.maximum(s_m, s_w)
threshold = cost / value
send = best > threshold

result = raw.copy()
result["uplift_mens_email"] = s_m
result["uplift_womens_email"] = s_w
result["experimental_plan"] = np.where(
    send, np.where(s_m >= s_w, "Men's email", "Women's email"), "Do not email"
)

n = len(result)
n_men = int((result["experimental_plan"] == "Men's email").sum())
n_women = int((result["experimental_plan"] == "Women's email").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{n:,}")
k2.metric("Simple plan emails", f"{n:,}" if cost < be_m else "0")
k3.metric("Experimental plan: men's / women's", f"{n_men:,} / {n_women:,}")
k4.metric("Experimental plan: skipped", f"{n - n_men - n_women:,}")
st.caption("No profit is shown for the experimental plan, because validation did not support one.")

st.subheader("Top 20 customers by score")
show = ["recency", "history", "channel", "zip_code", "uplift_mens_email", "uplift_womens_email", "experimental_plan"]
top = result.assign(_best=best).sort_values("_best", ascending=False)
st.dataframe(top[show].head(20).round(4))

st.download_button(
    "Download customer scores (CSV)",
    data=top.drop(columns="_best").to_csv(index=False).encode("utf-8"),
    file_name="uplift_scores.csv",
    mime="text/csv",
)