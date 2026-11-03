"""
Sends 500 customers to the running API twice: normal, then deliberately shifted.
Start the server first (uvicorn api.main:app --reload), then:
python examples/check_drift.py
"""
import json

import pandas as pd
import requests

BASE = "http://127.0.0.1:8000"
COLS = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]

raw = pd.read_csv("data/hillstrom.csv")
sample = raw.sample(500, random_state=1)


def check(df, label):
    body = {"customers": json.loads(df[COLS].to_json(orient="records"))}
    resp = requests.post(f"{BASE}/drift", json=body)
    resp.raise_for_status()
    out = resp.json()
    print(f"\n{label}: overall = {out['overall_status']}")
    for f in out["features"]:
        print(f"  {f['feature']:16s} PSI {f['psi']:.3f}  {f['status']}")


check(sample, "Normal customers")
check(sample.assign(recency=12, channel="Phone"), "Shifted customers (all recency=12, all Phone)")