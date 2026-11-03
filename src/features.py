"""
Turns raw customer columns into the 15 model features.
Must match Step 2 exactly, otherwise the model sees different data
than it was trained on (called "training-serving skew").
"""
import numpy as np
import pandas as pd

RAW_REQUIRED = [
    "recency", "history_segment", "history", "mens",
    "womens", "zip_code", "newbie", "channel",
]
ZIP_VALUES = ["Rural", "Suburban", "Urban"]
CHANNEL_VALUES = ["Multichannel", "Phone", "Web"]


def build_features(raw: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    missing = [c for c in RAW_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns in the uploaded file: {missing}")

    df = raw[RAW_REQUIRED].copy()

    # Same fixes and features as Step 2
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    df["history_segment_num"] = df["history_segment"].astype(str).str[0].astype(int)
    df["log_history"] = np.log1p(df["history"])
    # clip(lower=1) avoids dividing by zero; recency is never below 1 in the training data
    df["spend_per_recency"] = df["history"] / df["recency"].clip(lower=1)
    df["buys_both"] = ((df["mens"] == 1) & (df["womens"] == 1)).astype(int)

    # Manual one-hot encoding: a column is created even if a category
    # is missing in the uploaded file (get_dummies would skip it).
    for z in ZIP_VALUES:
        df[f"zip_code_{z}"] = (df["zip_code"] == z).astype(int)
    for c in CHANNEL_VALUES:
        df[f"channel_{c}"] = (df["channel"] == c).astype(int)

    return df[feature_cols]