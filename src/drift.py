"""
Data-drift check with PSI (Population Stability Index).
Compares the distribution of new customers with the training customers.
"""
import numpy as np
import pandas as pd

NUMERIC = ["recency", "history"]
CATEGORICAL = ["history_segment", "zip_code", "channel", "mens", "womens", "newbie"]
EPS = 1e-4            # avoids log(0) when a group is empty
OTHER = "__other__"   # bucket for categories never seen in training
MIN_CUSTOMERS = 200   # PSI is too noisy below this

STATUS_ORDER = ["stable", "moderate_shift", "major_shift"]


def _prepare(raw):
    df = raw.copy()
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df


def build_reference(raw, n_bins=10):
    """Summarise the training data (shares per bin / per category)."""
    df = _prepare(raw)
    ref = {"n_rows": int(len(df)), "numeric": {}, "categorical": {}}

    for col in NUMERIC:
        # Bin edges at the deciles of the training data (ends removed: open-ended bins)
        edges = np.unique(np.quantile(df[col], np.linspace(0, 1, n_bins + 1)))[1:-1]
        bins = np.digitize(df[col], edges)
        shares = np.bincount(bins, minlength=len(edges) + 1) / len(df)
        ref["numeric"][col] = {
            "edges": [float(e) for e in edges],
            "proportions": [float(p) for p in shares],
        }

    for col in CATEGORICAL:
        shares = df[col].value_counts(normalize=True)
        ref["categorical"][col] = {k: float(v) for k, v in shares.items()}

    return ref


def psi(expected, actual):
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status_of(value):
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate_shift"
    return "major_shift"


def compare(raw, ref):
    """PSI per feature for new customers vs the reference summary."""
    df = _prepare(raw)
    rows = []

    for col, spec in ref["numeric"].items():
        edges = np.array(spec["edges"], dtype=float)
        bins = np.digitize(df[col], edges)
        actual = np.bincount(bins, minlength=len(edges) + 1) / len(df)
        value = psi(spec["proportions"], actual)
        rows.append({"feature": col, "psi": value, "status": status_of(value)})

    for col, expected in ref["categorical"].items():
        shares = df[col].value_counts(normalize=True)
        actual = [float(shares.get(cat, 0.0)) for cat in expected]
        actual.append(max(0.0, 1.0 - sum(actual)))          # unseen categories
        exp_vec = list(expected.values()) + [0.0]
        value = psi(exp_vec, actual)
        rows.append({"feature": col, "psi": value, "status": status_of(value)})

    worst = max(rows, key=lambda r: STATUS_ORDER.index(r["status"]))["status"]
    return {"overall_status": worst, "features": rows}