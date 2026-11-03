"""
Checks for the drift code and the /drift endpoint.
Run from the project root:  python -m pytest -v
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from api.main import app  # noqa: E402
from drift import build_reference, compare  # noqa: E402

CUSTOMER_COLS = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(ROOT / "data" / "hillstrom.csv")


@pytest.fixture(scope="module")
def sample(raw):
    return raw.sample(500, random_state=0)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def as_customers(df):
    """Turns a table into the plain list of dicts that the API expects."""
    return json.loads(df[CUSTOMER_COLS].to_json(orient="records"))


def psi_of(result, feature):
    return next(r["psi"] for r in result["features"] if r["feature"] == feature)


# ---------- drift code ----------
def test_same_distribution_is_stable(raw, sample):
    result = compare(sample, build_reference(raw))
    assert result["overall_status"] == "stable"


def test_shifted_recency_is_detected(raw, sample):
    result = compare(sample.assign(recency=12), build_reference(raw))
    assert psi_of(result, "recency") > 0.25
    assert result["overall_status"] == "major_shift"


def test_unseen_category_is_detected(raw, sample):
    result = compare(sample.assign(channel="Fax"), build_reference(raw))
    assert psi_of(result, "channel") > 0.25


# ---------- /drift endpoint ----------
def test_api_drift_stable(client, sample):
    r = client.post("/drift", json={"customers": as_customers(sample)})
    assert r.status_code == 200
    data = r.json()
    assert data["n_customers"] == 500
    assert data["overall_status"] == "stable"
    assert len(data["features"]) == 8


def test_api_drift_major_shift(client, sample):
    shifted = sample.assign(recency=12, channel="Phone")
    r = client.post("/drift", json={"customers": as_customers(shifted)})
    assert r.status_code == 200
    assert r.json()["overall_status"] == "major_shift"


def test_api_drift_needs_enough_customers(client, sample):
    r = client.post("/drift", json={"customers": as_customers(sample.head(10))})
    assert r.status_code == 400