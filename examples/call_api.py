"""
Calls the API like another program would.

Local server (uvicorn api.main:app --reload):
    python examples/call_api.py

Live server (PowerShell):
    $env:API_BASE = "https://your-service.onrender.com"
    python examples/call_api.py
"""
import os

import requests

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
# The first request to a sleeping free service can take a minute
TIMEOUT = 120

print("Calling", BASE)
print(requests.get(f"{BASE}/health", timeout=TIMEOUT).json())

guide = requests.get(f"{BASE}/guide", params={"cost_per_email": 0.10}, timeout=TIMEOUT).json()
print("\nGuide:", guide["action"])
print(guide["message"])

customers = [
    {"recency": 10, "history_segment": "2) $100 - $200", "history": 142.44,
     "mens": 1, "womens": 0, "zip_code": "Suburban", "newbie": 0, "channel": "Phone"},
    {"recency": 2, "history_segment": "6) $750 - $1,000", "history": 910.0,
     "mens": 1, "womens": 1, "zip_code": "Urban", "newbie": 0, "channel": "Multichannel"},
]
resp = requests.post(
    f"{BASE}/score", json={"customers": customers, "cost_per_email": 0.10}, timeout=TIMEOUT
)
resp.raise_for_status()
out = resp.json()
print("\nScores:")
for row in out["results"]:
    print(row)
print("Counts:", out["counts"])
print("Note:", out["note"])