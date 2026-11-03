"""
Calls the running API like another program would.
Start the server first (uvicorn api.main:app --reload), then:
python examples/call_api.py
"""
import requests

BASE = "http://127.0.0.1:8000"

print(requests.get(f"{BASE}/health").json())

guide = requests.get(f"{BASE}/guide", params={"cost_per_email": 0.10}).json()
print("\nGuide:", guide["action"])
print(guide["message"])

customers = [
    {"recency": 10, "history_segment": "2) $100 - $200", "history": 142.44,
     "mens": 1, "womens": 0, "zip_code": "Suburban", "newbie": 0, "channel": "Phone"},
    {"recency": 2, "history_segment": "6) $750 - $1,000", "history": 910.0,
     "mens": 1, "womens": 1, "zip_code": "Urban", "newbie": 0, "channel": "Multichannel"},
]
resp = requests.post(f"{BASE}/score", json={"customers": customers, "cost_per_email": 0.10})
resp.raise_for_status()
out = resp.json()
print("\nScores:")
for row in out["results"]:
    print(row)
print("Counts:", out["counts"])
print("Note:", out["note"])