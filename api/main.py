"""
Email decision API.
Run from the project root:  uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from drift import MIN_CUSTOMERS, compare  # noqa: E402
from features import build_features  # noqa: E402
from policy import experimental_plan, shrink_scores, simple_recommendation  # noqa: E402

from api.schemas import (  # noqa: E402
    CustomerScore,
    DriftRequest,
    DriftResponse,
    GuideResponse,
    ScoreRequest,
    ScoreResponse,
)

# Filled once at startup, then reused for every request
state = {}

GUIDE_NOTE = (
    "Recommendation comes from averages in a randomized experiment on 64,000 customers. "
    "Customer-level targeting was not shown to beat this simple rule."
)
SCORE_NOTE = (
    "Experimental. In validation, raw model scores were about 1.8x too high and customer-level "
    "targeting did not beat the simple plan with confidence. Scores here are shrunk toward the "
    "average effect. Use them to prepare an A/B test, not as proven advice."
)
DRIFT_NOTE = (
    "PSI compares the mix of new customers with the training customers: below 0.10 stable, "
    "0.10 to 0.25 moderate shift, above 0.25 major shift (rules of thumb). Input drift does "
    "not prove the model got worse, and it cannot show a change in how customers respond to "
    "the email; only a new experiment can. If the shift is major, treat /score output with "
    "extra caution and consider a fresh A/B test."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before `yield`) and once when it stops."""
    cfg_path = ROOT / "outputs" / "policy_config.json"
    if not cfg_path.exists():
        raise RuntimeError("outputs/policy_config.json not found. Run src/step10_cost_and_shrinkage.py first.")
    ref_path = ROOT / "outputs" / "reference_stats.json"
    if not ref_path.exists():
        raise RuntimeError("outputs/reference_stats.json not found. Run src/make_reference.py first.")

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(ref_path) as f:
        state["reference"] = json.load(f)
    with open(ROOT / "data" / "processed" / "feature_cols.json") as f:
        feature_cols = json.load(f)

    state["cfg"] = cfg
    state["feature_cols"] = feature_cols
    state["models"] = {k: joblib.load(ROOT / v) for k, v in cfg["model_files"].items()}
    yield
    state.clear()


app = FastAPI(
    title="Email Decision API",
    version="1.1.0",
    description="Break-even guidance, experimental customer-level uplift scores, and a data-drift check for an email campaign.",
    lifespan=lifespan,
)


def resolve_settings(cost, margin, avg_spend):
    """Use the caller's values, or the defaults saved in policy_config.json."""
    d = state["cfg"]["defaults"]
    cost = cost if cost is not None else d["cost_per_email"]
    margin = margin if margin is not None else d["profit_margin"]
    avg_spend = avg_spend if avg_spend is not None else d["avg_spend_of_buyers"]
    return float(cost), float(margin), float(avg_spend)


# Plain `def` (not `async def`) on purpose: model prediction is heavy computation, and
# FastAPI runs plain functions in a separate thread so the server stays responsive.
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(state["models"].keys())}


@app.get("/guide", response_model=GuideResponse)
def guide(
    cost_per_email: Optional[float] = Query(None, gt=0, le=10),
    profit_margin: Optional[float] = Query(None, gt=0, le=1),
    avg_spend_of_buyers: Optional[float] = Query(None, gt=0),
):
    cost, margin, avg_spend = resolve_settings(cost_per_email, profit_margin, avg_spend_of_buyers)
    g = state["cfg"]["global_uplift"]
    result = simple_recommendation(cost, g["mens"], g["womens"], avg_spend * margin)
    return {**result, "note": GUIDE_NOTE}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    cost, margin, avg_spend = resolve_settings(
        req.cost_per_email, req.profit_margin, req.avg_spend_of_buyers
    )
    threshold = cost / (avg_spend * margin)

    raw = pd.DataFrame([c.model_dump() for c in req.customers])
    try:
        X = build_features(raw, state["feature_cols"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not prepare the data: {e}")

    m = state["models"]
    p0 = m["none"].predict_proba(X)[:, 1]
    up_m = m["mens"].predict_proba(X)[:, 1] - p0
    up_w = m["womens"].predict_proba(X)[:, 1] - p0

    s_m, s_w = shrink_scores(up_m, up_w, state["cfg"])
    plan = experimental_plan(s_m, s_w, threshold)

    results = [
        CustomerScore(
            uplift_mens_email=float(a),
            uplift_womens_email=float(b),
            experimental_plan=str(c),
        )
        for a, b, c in zip(s_m, s_w, plan)
    ]
    counts = {k: int((plan == k).sum()) for k in ("Men's email", "Women's email", "Do not email")}

    return ScoreResponse(
        cost_per_email=cost,
        break_even_uplift=threshold,
        results=results,
        counts=counts,
        note=SCORE_NOTE,
    )


@app.post("/drift", response_model=DriftResponse)
def drift(req: DriftRequest):
    if len(req.customers) < MIN_CUSTOMERS:
        raise HTTPException(
            status_code=400,
            detail=f"Send at least {MIN_CUSTOMERS} customers: PSI is too noisy on smaller samples.",
        )
    raw = pd.DataFrame([c.model_dump() for c in req.customers])
    result = compare(raw, state["reference"])
    return DriftResponse(
        n_customers=len(raw),
        overall_status=result["overall_status"],
        features=result["features"],
        note=DRIFT_NOTE,
    )