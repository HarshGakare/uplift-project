"""
Builds the reference summary of the training customers.
Run from the project root:  python src/make_reference.py
"""
import json
from pathlib import Path

import pandas as pd

from drift import build_reference

raw = pd.read_csv("data/hillstrom.csv")
ref = build_reference(raw)

Path("outputs").mkdir(exist_ok=True)
with open("outputs/reference_stats.json", "w") as f:
    json.dump(ref, f, indent=2)

print(f"Reference built from {ref['n_rows']:,} customers")
print("Numeric features:", list(ref["numeric"]))
print("Categorical features:", list(ref["categorical"]))
print("Saved: outputs/reference_stats.json")