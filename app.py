"""
Email decision dashboard (plain-language version).
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
from features import build_features  # noqa: E402
from upload_checks import MAX_ROWS, clean_upload  # noqa: E402

st.set_page_config(page_title="Email Decision Guide", layout="wide")
POP = 100_000


def money(v):
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


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

# Results from our earlier testing (cost sweep), at the default margin and spend
TEST_COSTS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
TEST_MEN_ALL = [21879, 18879, 13879, 3879, -6121, -16121]
TEST_SMART = [19615, 16446, 12255, 4781, -1081, -1201]
TEST_WIN = ["10%", "7%", "22%", "66%", "96%", "100%"]

# Example file for the upload section (made-up customers, only the required columns)
TEMPLATE = pd.DataFrame(
    [
        [10, 142.44, 1, 0, "Suburban", 0, "Phone"],
        [6, 329.08, 1, 1, "Rural", 1, "Web"],
        [2, 910.00, 0, 1, "Urban", 0, "Multichannel"],
        [9, 45.50, 1, 0, "Urban", 1, "Web"],
    ],
    columns=["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"],
)

COLUMN_GUIDE = pd.DataFrame(
    {
        "Column name (row 1 of your file)": [
            "recency", "history", "mens", "womens", "zip_code", "newbie", "channel",
        ],
        "What it means": [
            "Months since the customer last bought something",
            "Total money the customer spent with you in the last 12 months",
            "Has the customer bought men's products?",
            "Has the customer bought women's products?",
            "What kind of area the customer lives in",
            "Is this a new customer (joined in the last 12 months)?",
            "How the customer usually buys",
        ],
        "What to type": [
            "A whole number from 1 to 12",
            "A number of dollars, 0 or more (a $ sign or commas are okay)",
            "1 for yes, 0 for no",
            "1 for yes, 0 for no",
            "Rural, Suburban or Urban",
            "1 for yes, 0 for no",
            "Phone, Web or Multichannel (Multichannel = buys in more than one way)",
        ],
        "Example": ["10", "142.44", "1", "0", "Suburban", "0", "Phone"],
    }
)

# =====================================================================
# Title
# =====================================================================
st.title("Which email should we send?")
st.write(
    "This page helps you decide **whether to send a marketing email, which one, and whether "
    "it is worth the money**. It is based on a real test where 64,000 customers were randomly "
    "given the men's email, the women's email, or no email at all. Afterwards we counted who bought."
)
st.info(
    "**How to use this page:** (1) Type your own numbers in the left panel. "
    "(2) Read the advice in section 2. "
    "(3) Optional: in section 3, check your own customers one by one."
)

# =====================================================================
# Sidebar
# =====================================================================
st.sidebar.header("Your numbers")
st.sidebar.caption("Change these to match your business. The whole page updates.")
margin = st.sidebar.slider(
    "Profit margin", 0.05, 0.80, float(d["profit_margin"]), 0.01,
    help="Out of every $1 a customer spends, how much is your profit? 0.30 means 30 cents.",
)
cost = st.sidebar.slider(
    "Cost of sending one email ($)", 0.01, 1.00, float(d["cost_per_email"]), 0.01,
    help="What it costs you to send one email to one customer.",
)
avg_spend = st.sidebar.number_input(
    "Average amount one buyer spends ($)", min_value=1.0,
    value=float(d["avg_spend_of_buyers"]), step=5.0,
    help="When a customer buys, about how much do they spend?",
)
value = avg_spend * margin
be_m, be_w = g_m * value, g_w * value
st.sidebar.markdown(f"**Profit from one extra customer: ${value:,.2f}**")
st.sidebar.caption(
    "An 'extra customer' is someone who bought only because of the email. "
    "Without the email, they would not have bought."
)

# =====================================================================
# Section 1
# =====================================================================
st.header("1. How well does each email work?")
st.write(
    f"The test shows that **the men's email works about {g_m / g_w:.0f} times as well as the women's email**. "
    "Numbers below are for every 100,000 customers (1 lakh) who receive the email."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Men's email: extra people who buy", f"{g_m * POP:,.0f}",
    help="If 100,000 customers get the men's email, about this many MORE people buy than if nobody was emailed.",
)
c2.metric(
    "Women's email: extra people who buy", f"{g_w * POP:,.0f}",
    help="If 100,000 customers get the women's email, about this many MORE people buy than if nobody was emailed.",
)
c3.metric(
    "Most you can pay per men's email", f"${be_m:.2f}",
    help="If one men's email costs less than this, it earns more than it costs (on average). If it costs more, you lose money.",
)
c4.metric(
    "Most you can pay per women's email", f"${be_w:.2f}",
    help="The same limit for the women's email. It is lower because this email brings fewer extra buyers.",
)
st.caption(
    "These limits come straight from the test results and your numbers on the left: "
    "extra buyers per customer × profit from one extra customer."
)

# =====================================================================
# Section 2
# =====================================================================
st.header("2. What should we do?")

men_spend = POP * cost
men_extra = g_m * POP
men_profit = men_extra * value
men_net = men_profit - men_spend

st.markdown(
    f"**Example: sending the men's email to {POP:,} customers (1 lakh)**\n"
    f"- You spend about **{money(men_spend)}** on emails.\n"
    f"- About **{men_extra:,.0f} more people** buy because of the email. That brings in about **{money(men_profit)}** profit.\n"
    f"- Result: **{'a gain' if men_net >= 0 else 'a loss'} of about {money(abs(men_net))}**."
)

if cost >= be_m:
    st.error(
        f"**Do not send either email to everyone.** At ${cost:.2f} per email, even the better "
        f"email (the men's one) costs more than it earns. The most you could pay per men's email is "
        f"${be_m:.2f}. Our testing also did not find a reliable group of customers where emailing "
        "still pays. The safest choice is to send nothing, or to try a small trial first."
    )
else:
    if cost < be_w:
        women_line = "The women's email also pays for itself, but it earns less."
    else:
        women_line = (
            f"Do **not** send the women's email to everyone: at this price it loses money "
            f"(the most you could pay for it is ${be_w:.2f})."
        )
    st.success(
        f"**Send the men's email to everyone.** At ${cost:.2f} per email you can expect to earn about "
        f"**{money(men_net)}** for every {POP:,} customers (this is an estimate). {women_line}"
    )
    if cost >= 0.75 * be_m:
        st.info(
            "You are close to the limit. Picking customers one by one looked most promising "
            "around this price, but our testing could not prove it. If you have enough customers, "
            "run a small trial (an A/B test) before relying on it."
        )

costs = np.linspace(0.0, max(0.5, 1.5 * be_m), 100)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(costs, POP * (g_m * value - costs), label="Men's email to everyone")
ax.plot(costs, POP * (g_w * value - costs), label="Women's email to everyone")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(cost, color="green", linestyle="--", label="Your email cost")
ax.set_xlabel("Cost of one email ($)")
ax.set_ylabel(f"Money made or lost, {POP:,} customers ($)")
ax.set_title("Where does emailing stop paying off?")
ax.legend()
st.pyplot(fig)
st.caption(
    "Above the black line you make money, below it you lose money. "
    "The point where a line crosses the black line is the most you can pay per email."
)

with st.expander("Proof from our testing (optional reading)"):
    proof = pd.DataFrame(
        {
            "Cost of one email": [f"${c:.2f}" for c in TEST_COSTS],
            "Send nothing": [money(0)] * len(TEST_COSTS),
            "Men's email to everyone": [money(v) for v in TEST_MEN_ALL],
            "Pick customers one by one": [money(v) for v in TEST_SMART],
            "How often picking one by one did better than 'men's email to everyone'": TEST_WIN,
        }
    )
    st.dataframe(proof, hide_index=True)
    st.markdown(
        f"- Money numbers are for {POP:,} customers, using a {d['profit_margin']:.0%} margin and "
        f"${d['avg_spend_of_buyers']} average spend.\n"
        "- 'Pick customers one by one' means using a prediction model to choose who gets which email.\n"
        "- The last column comes from re-running the comparison many times on re-shuffled data. "
        "If a plan wins 80 out of 100 times we trust it. Around 50 means we cannot tell which is better.\n"
        "- At a cost of $0.30 and above, the high percentages only mean that picking one by one "
        "**lost less money**. Sending nothing was the best choice there."
    )

# =====================================================================
# Section 3
# =====================================================================
st.header("3. Optional: look at individual customers")
st.warning(
    "**This part is experimental.** In our testing, the model's predictions for single customers "
    "were too optimistic (about 1.8 times too high) and did not clearly beat the simple advice "
    "in section 2. Use this list to choose people for a small trial, not as a final answer."
)

source = st.radio(
    "Which customers do you want to look at?",
    ["Try with sample customers", "Upload my own customer list"],
    horizontal=True,
)

raw = None
if source.startswith("Try"):
    raw = load_demo()
    st.caption(
        "These 2,000 sample customers come from the same data the tool learned from, "
        "so use them only to see how the page works."
    )
else:
    st.subheader("How to prepare your file")
    st.markdown(
        "1. Open your customer list in Excel or Google Sheets.\n"
        "2. **Row 1 must hold the column names exactly as written in the table below.** "
        "Every row after that is one customer.\n"
        "3. Fill in each column the way the table describes. Extra columns (like customer ID or name) "
        "are fine: we do not use them, but we keep them in the file you download. "
        "You do not need a spending-group column: we work it out from the 'history' amount.\n"
        f"4. Save the file as **CSV (Comma delimited)** and upload it below (up to {MAX_ROWS:,} customers)."
    )
    st.dataframe(COLUMN_GUIDE, hide_index=True)
    st.write("**What your file should look like** (made-up customers):")
    st.dataframe(TEMPLATE, hide_index=True)
    st.download_button(
        "Download this example as a CSV template",
        data=TEMPLATE.to_csv(index=False).encode("utf-8"),
        file_name="customer_list_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload your customer list (CSV file)", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception:
            st.error(
                "We could not read that file. Please open it in Excel, choose Save As, "
                "pick 'CSV (Comma delimited)', and upload the new file."
            )
            st.stop()

if raw is None:
    st.info("Choose the sample customers, or upload your file, to continue.")
    st.stop()

clean, problems, notes = clean_upload(raw)
if problems:
    st.error("Please fix these points in your file and upload it again:")
    for p in problems:
        st.markdown(f"- {p}")
    st.stop()
for note in notes:
    st.info(note)

X = build_features(clean, feature_cols)
p0 = models["none"].predict_proba(X)[:, 1]
up_m = models["mens"].predict_proba(X)[:, 1] - p0
up_w = models["womens"].predict_proba(X)[:, 1] - p0

# Pull the scores toward the average effect of each email (raw scores were too high in testing)
s_m = g_m + lam * (up_m - mu_m)
s_w = g_w + lam * (up_w - mu_w)
best = np.maximum(s_m, s_w)
threshold = cost / value
send = best > threshold
plan = np.where(send, np.where(s_m >= s_w, "Men's email", "Women's email"), "Do not email")

n = len(clean)
n_men = int((plan == "Men's email").sum())
n_women = int((plan == "Women's email").sum())
n_none = int((plan == "Do not email").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers checked", f"{n:,}")
k2.metric("Suggested: men's email", f"{n_men:,}")
k3.metric("Suggested: women's email", f"{n_women:,}")
k4.metric("Suggested: no email", f"{n_none:,}")
st.caption(
    "For comparison, the simple advice in section 2 would "
    + (f"send the men's email to all {n:,} customers." if cost < be_m else "send no emails.")
    + " We do not show a profit figure for the list above, because our testing could not prove one."
)

COL_M = "Extra buying chance if sent men's email (%)"
COL_W = "Extra buying chance if sent women's email (%)"
COL_PLAN = "Suggested action"

st.subheader("The 20 customers most likely to respond")
view = pd.DataFrame(
    {
        "Months since last purchase": clean["recency"].to_numpy(),
        "Spent in last 12 months ($)": clean["history"].to_numpy(),
        "Buys via": clean["channel"].to_numpy(),
        "Area": clean["zip_code"].to_numpy(),
        COL_M: (s_m * 100).round(2),
        COL_W: (s_w * 100).round(2),
        COL_PLAN: plan,
    }
)
view = view.assign(_best=best).sort_values("_best", ascending=False).drop(columns="_best")
st.dataframe(view.head(20), hide_index=True)
st.caption(
    "How to read 'extra buying chance': 0.70 means that out of 100 customers like this one, "
    "about 0.7 extra people buy because of the email. Higher is better. "
    "Most customers sit close to the average, because we deliberately pull the scores toward it."
)

out = raw.copy()
out[COL_M] = (s_m * 100).round(2)
out[COL_W] = (s_w * 100).round(2)
out[COL_PLAN] = plan
out = out.assign(_best=best).sort_values("_best", ascending=False).drop(columns="_best")
st.download_button(
    "Download the full list with suggestions (CSV)",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="customers_with_suggestions.csv",
    mime="text/csv",
)