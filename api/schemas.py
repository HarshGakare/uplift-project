"""Describes what valid input and output look like (Pydantic models)."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Customer(BaseModel):
    recency: int = Field(..., ge=1, le=12, description="Months since last purchase (1 to 12)")
    history_segment: str = Field(
        ..., pattern=r"^[1-7]\)", description="Spend band, starts with 1) to 7), e.g. '2) $100 - $200'"
    )
    history: float = Field(..., ge=0, description="Dollars spent in the last year")
    mens: Literal[0, 1] = Field(..., description="1 if bought men's products")
    womens: Literal[0, 1] = Field(..., description="1 if bought women's products")
    zip_code: Literal["Rural", "Suburban", "Urban", "Surburban"]
    newbie: Literal[0, 1] = Field(..., description="1 if new customer")
    channel: Literal["Phone", "Web", "Multichannel"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "recency": 10,
                "history_segment": "2) $100 - $200",
                "history": 142.44,
                "mens": 1,
                "womens": 0,
                "zip_code": "Suburban",
                "newbie": 0,
                "channel": "Phone",
            }
        }
    }


class ScoreRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=5000)
    cost_per_email: Optional[float] = Field(None, gt=0, le=10)
    profit_margin: Optional[float] = Field(None, gt=0, le=1)
    avg_spend_of_buyers: Optional[float] = Field(None, gt=0)


class CustomerScore(BaseModel):
    uplift_mens_email: float
    uplift_womens_email: float
    experimental_plan: str


class ScoreResponse(BaseModel):
    cost_per_email: float
    break_even_uplift: float
    results: List[CustomerScore]
    counts: Dict[str, int]
    note: str


class GuideResponse(BaseModel):
    action: str
    message: str
    cost_per_email: float
    profit_per_extra_buyer: float
    break_even_cost_mens: float
    break_even_cost_womens: float
    womens_email_pays_off: bool
    near_break_even: bool
    estimated_profit_per_100k_mens_to_all: float
    estimated_profit_per_100k_womens_to_all: float
    note: str


class DriftRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=20000)


class FeatureDrift(BaseModel):
    feature: str
    psi: float
    status: str


class DriftResponse(BaseModel):
    n_customers: int
    overall_status: str
    features: List[FeatureDrift]
    note: str