"""
Step 7: Streamlit app for the Uplift Targeting project.
Run from the project root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from uplift_utils import best_k, profit_curve  # noqa: E402

st.set_page_config(
    page_title="Uplift Targeting Assistant",
    page_icon="📧",
    layout="wide",
)

REQUIRED_COLUMNS = ["recency", "history", "mens", "womens", "newbie",
                    "zip_code", "channel"]


# ---------------------------------------------------------------------
# Loading (cached so the app stays fast)
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    policy = json.loads((ROOT / "outputs" / "policy.json").read_text())
    features = json.loads(
        (ROOT / "data" / "processed" / "features.json").read_text()
    )
    method = policy["method"]
    models = {}
    if method == "T-learner (tuned)":
        models["t"] = joblib.load(ROOT / "models" / "model_t_tuned.joblib")
        models["c"] = joblib.load(ROOT / "models" / "model_c_tuned.joblib")
    elif method == "T-learner (Step 3, untuned)":
        models["t"] = joblib.load(ROOT / "models" / "model_t.joblib")
        models["c"] = joblib.load(ROOT / "models" / "model_c.joblib")
    else:  # transformed-outcome model
        models["transformed"] = joblib.load(
            ROOT / "models" / "model_transformed.joblib"
        )
    return policy, features, models


@st.cache_data
def load_test(score_column):
    df = pd.read_csv(ROOT / "data" / "processed" / "test_scored_v2.csv")
    return df[[score_column, "spend", "treatment", "target"]].copy()


@st.cache_data
def compute_policy(cost, margin, score_column):
    """Best share of customers to email for the chosen cost and margin."""
    test = load_test(score_column)
    score = test[score_column].to_numpy()
    profit = profit_curve(
        test["spend"].to_numpy(dtype=float),
        test["treatment"].to_numpy(),
        score, cost, margin,
    )
    n = len(test)
    k = best_k(profit, n)
    cutoff = float(np.sort(score)[::-1][k - 1]) if k > 0 else None
    return k, cutoff, profit


try:
    policy, FEATURES, models = load_artifacts()
except FileNotFoundError as err:
    st.error(
        f"Missing file: {err.filename}\n\n"
        "Please run Steps 1 to 6 first so the models and policy file exist."
    )
    st.stop()

SCORE_COL = policy["score_column"]
test_df = load_test(SCORE_COL)

ZIP_OPTIONS = [f[len("zip_code_"):] for f in FEATURES if f.startswith("zip_code_")]
CHANNEL_OPTIONS = [f[len("channel_"):] for f in FEATURES if f.startswith("channel_")]


# ---------------------------------------------------------------------
# Feature building and scoring (must match Step 2 exactly)
# ---------------------------------------------------------------------
def build_features(raw):
    df = raw.copy()
    df["log_history"] = np.log1p(df["history"])
    df["is_recent"] = (df["recency"] <= 3).astype(int)
    df["both_products"] = ((df["mens"] == 1) & (df["womens"] == 1)).astype(int)

    out = pd.DataFrame(index=df.index)
    for feat in FEATURES:
        if feat.startswith("zip_code_"):
            out[feat] = (df["zip_code"] == feat[len("zip_code_"):]).astype(int)
        elif feat.startswith("channel_"):
            out[feat] = (df["channel"] == feat[len("channel_"):]).astype(int)
        else:
            out[feat] = df[feat]
    return out[FEATURES]


def score_customers(raw):
    """Returns a DataFrame with uplift (and the two probabilities if available)."""
    feats = build_features(raw)
    result = pd.DataFrame(index=raw.index)
    if "transformed" in models:
        result["uplift"] = models["transformed"].predict(feats)
    else:
        result["p_email"] = models["t"].predict_proba(feats)[:, 1]
        result["p_no_email"] = models["c"].predict_proba(feats)[:, 1]
        result["uplift"] = result["p_email"] - result["p_no_email"]
    return result


# ---------------------------------------------------------------------
# Sidebar: business assumptions
# ---------------------------------------------------------------------
st.sidebar.header("Business assumptions")
cost = st.sidebar.slider(
    "Cost per email ($)", 0.01, 2.00,
    float(policy["cost_per_email"]), 0.01,
)
margin = st.sidebar.slider(
    "Profit margin on spend", 0.05, 1.00,
    float(policy["profit_margin"]), 0.05, format="%.2f",
)

k_best, cutoff, profit_arr = compute_policy(cost, margin, SCORE_COL)
n_test = len(test_df)

if k_best > 0:
    st.sidebar.success(
        f"Best policy: email the top **{k_best / n_test:.0%}** of customers"
    )
else:
    st.sidebar.warning("At these assumptions, emailing nobody is best.")
st.sidebar.caption(f"Model: {policy['method']}")


def decide(uplift_value):
    """True = email, False = skip."""
    return cutoff is not None and uplift_value >= cutoff


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.title("📧 Uplift Targeting Assistant")
st.write(
    "Instead of asking *who will buy?*, this app asks "
    "*who will buy **because** we emailed them?* "
    "so the email budget goes only where it changes behaviour."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Single customer", "Batch scoring", "Model results", "About and limits"]
)

# ---------------- Tab 1: single customer ----------------
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Customer details")
        recency = st.slider(
            "Months since last purchase", 1, 12, 6,
            help="1 = bought very recently, 12 = bought a year ago.",
        )
        history = st.number_input(
            "Dollars spent in the last year", 0.0, 5000.0, 250.0, 10.0
        )
        mens = st.checkbox("Bought men's items last year", value=True)
        womens = st.checkbox("Bought women's items last year", value=False)
        newbie = st.checkbox("New customer (joined in the last 12 months)")
        zip_code = st.selectbox("Area type", ZIP_OPTIONS)
        channel = st.selectbox("Purchase channel", CHANNEL_OPTIONS)

    raw_one = pd.DataFrame(
        [{
            "recency": recency, "history": history,
            "mens": int(mens), "womens": int(womens), "newbie": int(newbie),
            "zip_code": zip_code, "channel": channel,
        }]
    )
    res = score_customers(raw_one).iloc[0]
    uplift_value = float(res["uplift"])
    percentile = float((test_df[SCORE_COL] < uplift_value).mean())

    with right:
        st.subheader("Result")
        if decide(uplift_value):
            st.success("✅ **Send the email**")
        else:
            st.error("⛔ **Do not send the email**")

        c1, c2 = st.columns(2)
        c1.metric("Predicted uplift", f"{uplift_value * 100:+.2f} pts")
        c2.metric("Higher than", f"{percentile:.0%} of test customers")

        if "p_email" in res:
            c3, c4 = st.columns(2)
            c3.metric("Chance to buy WITH email", f"{res['p_email']:.2%}")
            c4.metric("Chance to buy WITHOUT email", f"{res['p_no_email']:.2%}")

        if uplift_value < 0:
            st.warning(
                "Negative uplift: this customer may be a 'sleeping dog', "
                "someone the email could push away."
            )
        if cutoff is not None:
            st.caption(
                f"Decision rule: email if uplift is at least "
                f"{cutoff * 100:.2f} points (cutoff from the profit curve)."
            )
        st.caption(
            "pts = percentage points of purchase chance. "
            "+1.00 pts means the email raises the chance to buy by 1 point."
        )

# ---------------- Tab 2: batch scoring ----------------
with tab2:
    st.subheader("Score many customers at once")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))

    template = pd.DataFrame(
        [
            [2, 450.0, 1, 0, 0, ZIP_OPTIONS[0], CHANNEL_OPTIONS[0]],
            [9, 60.0, 0, 1, 1, ZIP_OPTIONS[-1], CHANNEL_OPTIONS[-1]],
        ],
        columns=REQUIRED_COLUMNS,
    )
    st.download_button(
        "Download a template CSV",
        template.to_csv(index=False).encode("utf-8"),
        file_name="customers_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customers CSV", type="csv")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)

        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        for col in ["recency", "history", "mens", "womens", "newbie"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        bad_numbers = raw[REQUIRED_COLUMNS].isnull().any(axis=1)
        bad_zip = ~raw["zip_code"].isin(ZIP_OPTIONS)
        bad_channel = ~raw["channel"].isin(CHANNEL_OPTIONS)

        if bad_numbers.any() or bad_zip.any() or bad_channel.any():
            st.error(
                "Some rows have empty or invalid values. "
                f"Allowed zip_code: {ZIP_OPTIONS}. "
                f"Allowed channel: {CHANNEL_OPTIONS}."
            )
            bad_rows = raw[bad_numbers | bad_zip | bad_channel]
            st.dataframe(bad_rows.head(20))
        else:
            scored = score_customers(raw)
            out = raw.copy()
            out["uplift_score"] = scored["uplift"].round(5)
            out["decision"] = np.where(
                scored["uplift"].apply(decide), "EMAIL", "SKIP"
            )
            out["rank"] = out["uplift_score"].rank(
                ascending=False, method="first"
            ).astype(int)
            out = out.sort_values("rank")

            n_email = int((out["decision"] == "EMAIL").sum())
            a, b, c = st.columns(3)
            a.metric("Customers", len(out))
            b.metric("Email", n_email)
            c.metric("Skip", len(out) - n_email)

            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download decisions CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="uplift_decisions.csv",
                mime="text/csv",
            )

# ---------------- Tab 3: model results ----------------
with tab3:
    st.subheader("Live profit curve")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_pct = np.arange(len(profit_arr)) / n_test * 100
    ax.plot(x_pct, profit_arr / n_test * 1000, linewidth=2, label="Uplift policy")
    ax.axhline(0, color="black", linewidth=0.8)
    if k_best > 0:
        ax.axvline(k_best / n_test * 100, color="green", linestyle="--",
                   label=f"Best cutoff ({k_best / n_test:.0%})")
    ax.set_xlabel("% of customers emailed (highest score first)")
    ax.set_ylabel("Profit per 1,000 customers ($)")
    ax.set_title(f"Cost ${cost:.2f} per email, margin {margin:.0%}")
    ax.legend()
    st.pyplot(fig)

    if k_best > 0:
        st.write(
            f"At these assumptions: profit of about "
            f"**${profit_arr[k_best] / n_test * 1000:,.0f} per 1,000 customers** "
            f"when emailing the top {k_best / n_test:.0%}, versus "
            f"**${profit_arr[-1] / n_test * 1000:,.0f}** when emailing everyone. "
            "These numbers come from the same test data used to pick the cutoff, "
            "so they are optimistic. See the honest split-half check in Step 6."
        )

    st.subheader("Qini scores on the test set")
    metrics_path = ROOT / "outputs" / "step5_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        rows = [
            {
                "Method": name,
                "Qini (extra buyers)": round(vals["point"], 2),
                "95% low": round(vals["ci_low"], 2),
                "95% high": round(vals["ci_high"], 2),
            }
            for name, vals in metrics["test_qini"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Higher is better and 0 means no better than random. "
            "Overlapping ranges mean the difference is not certain."
        )

    img1, img2 = st.columns(2)
    qini_img = ROOT / "outputs" / "step5_qini_comparison.png"
    decile_img = ROOT / "outputs" / "step3_decile_uplift.png"
    if qini_img.exists():
        img1.image(str(qini_img), caption="Qini curves (Step 5)")
    if decile_img.exists():
        img2.image(str(decile_img), caption="Observed uplift by decile (Step 3)")

# ---------------- Tab 4: about and limits ----------------
with tab4:
    st.subheader("How it works")
    st.markdown(
        """
- **Data:** Hillstrom email marketing dataset. Customers were randomly given a
  men's email, a women's email, or no email.
- **Model:** two LightGBM models, one trained on emailed customers and one on
  non-emailed customers. **Uplift = chance to buy with email minus chance to
  buy without it.** (The app uses whichever method won in Step 5.)
- **Decision:** the profit curve picks how many top-ranked customers to email.
        """
    )
    st.subheader("Limits you should know")
    st.markdown(
        f"""
- The cost per email (${policy['cost_per_email']:.2f}) and margin
  ({policy['profit_margin']:.0%}) are **assumptions**, not real company numbers.
- Only about 1% of customers buy, so results are noisy. Check the confidence
  ranges before trusting any single number.
- Men's and women's emails are treated as one "email". The two were not compared.
- The data is from 2008 and one company. Before real use, run a live A/B test
  of this targeting against the current policy.
- Predictions describe groups of similar customers. They are not guarantees
  for one person.
        """
    )