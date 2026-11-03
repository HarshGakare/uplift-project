"""
Automatic checks for the API.
Run from the project root:  python -m pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_CUSTOMER = {
    "recency": 10,
    "history_segment": "2) $100 - $200",
    "history": 142.44,
    "mens": 1,
    "womens": 0,
    "zip_code": "Suburban",
    "newbie": 0,
    "channel": "Phone",
}


@pytest.fixture(scope="module")
def client():
    # `with` makes the server startup (model loading) run
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert len(r.json()["models_loaded"]) == 3


def test_guide_default_cost_recommends_mens_email(client):
    r = client.get("/guide")
    assert r.status_code == 200
    assert r.json()["action"] == "mens_email_to_everyone"


def test_guide_high_cost_recommends_nothing(client):
    r = client.get("/guide", params={"cost_per_email": 0.30})
    assert r.json()["action"] == "send_nothing"


def test_guide_near_break_even_flag(client):
    r = client.get("/guide", params={"cost_per_email": 0.20})
    assert r.json()["near_break_even"] is True


def test_guide_rejects_bad_cost(client):
    r = client.get("/guide", params={"cost_per_email": -1})
    assert r.status_code == 422


def test_score_one_result_per_customer(client):
    body = {"customers": [VALID_CUSTOMER] * 3}
    r = client.post("/score", json=body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    assert sum(data["counts"].values()) == 3
    assert data["results"][0]["experimental_plan"] in {"Men's email", "Women's email", "Do not email"}


def test_score_is_repeatable(client):
    body = {"customers": [VALID_CUSTOMER]}
    a = client.post("/score", json=body).json()["results"][0]
    b = client.post("/score", json=body).json()["results"][0]
    assert a == b


@pytest.mark.parametrize(
    "field,value",
    [("recency", 99), ("mens", 2), ("channel", "Fax"), ("history", -5), ("history_segment", "bad")],
)
def test_score_rejects_invalid_customer(client, field, value):
    bad = {**VALID_CUSTOMER, field: value}
    r = client.post("/score", json={"customers": [bad]})
    assert r.status_code == 422


def test_score_rejects_empty_list(client):
    r = client.post("/score", json={"customers": []})
    assert r.status_code == 422


def test_score_rejects_too_many_customers(client):
    r = client.post("/score", json={"customers": [VALID_CUSTOMER] * 5001})
    assert r.status_code == 422