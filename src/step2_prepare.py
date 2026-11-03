"""
Step 2: Clean, create features, and split into train/test.
Run from the project root:  python src/step2_prepare.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/hillstrom.csv")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------- 1. Load and clean ----------
df = pd.read_csv(DATA_PATH)

# Fix the spelling mistake in the original data
df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})

# Treatment: 1 = got any email, 0 = no email
df["treatment"] = (df["segment"] != "No E-Mail").astype(int)

# ---------- 2. Feature engineering ----------
# history_segment looks like "3) $200 - $350". The first character is
# the order (1 = lowest spender, 7 = highest), so we turn it into a number.
df["history_segment_num"] = df["history_segment"].str[0].astype(int)

# history (past spend) is skewed: a few customers spend a lot.
# log1p squeezes big values so the model sees a more balanced scale.
df["log_history"] = np.log1p(df["history"])

# Average spend per month of "recency" (recency = months since last purchase)
df["spend_per_recency"] = df["history"] / df["recency"]

# Does the customer buy both men's and women's products?
df["buys_both"] = ((df["mens"] == 1) & (df["womens"] == 1)).astype(int)

# One-hot encode text columns: each category becomes its own 0/1 column
df = pd.get_dummies(df, columns=["zip_code", "channel"], dtype=int)

# ---------- 3. Choose columns ----------
# Features = what we know about a customer BEFORE sending the email.
# visit, conversion, spend, and segment are results or labels, never features.
feature_cols = [
    "recency",
    "history",
    "log_history",
    "history_segment_num",
    "spend_per_recency",
    "mens",
    "womens",
    "buys_both",
    "newbie",
    "zip_code_Rural",
    "zip_code_Suburban",
    "zip_code_Urban",
    "channel_Multichannel",
    "channel_Phone",
    "channel_Web",
]
target_cols = ["treatment", "conversion", "visit", "spend", "segment"]

data = df[feature_cols + target_cols].copy()

# ---------- 4. Stratified train/test split ----------
# Strata = treatment + conversion combined, so all four combinations
# appear in the same proportion in train and test.
strata = data["treatment"].astype(str) + "_" + data["conversion"].astype(str)

train, test = train_test_split(
    data,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=strata,
)

# ---------- 5. Sanity checks ----------
def describe(name, d):
    print(f"\n{name}: {len(d)} rows")
    print(
        d.groupby("treatment")["conversion"]
        .agg(customers="size", conversion_rate="mean")
        .round(4)
    )
    uplift = (
        d.loc[d.treatment == 1, "conversion"].mean()
        - d.loc[d.treatment == 0, "conversion"].mean()
    )
    print(f"Overall uplift: {uplift:.4f}")

describe("TRAIN", train)
describe("TEST", test)

# ---------- 6. Save ----------
train.to_csv(PROC_DIR / "train.csv", index=False)
test.to_csv(PROC_DIR / "test.csv", index=False)

with open(PROC_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\nSaved: train.csv, test.csv, feature_cols.json in data/processed/")
print("Number of features:", len(feature_cols))