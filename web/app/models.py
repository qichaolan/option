"""Pydantic models for LEAPS Ranker API."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re


class LEAPSRequest(BaseModel):
    """Request model for LEAPS ranking."""

    symbol: str = Field(..., description="Ticker symbol (e.g., SPY, QQQ)")
    target_pct: float = Field(
        default=0.5,
        ge=0.01,
        le=2.0,
        description="Target percentage move (0.5 = 50%)"
    )
    mode: Literal["high_prob", "high_convexity"] = Field(
        default="high_prob",
        description="Scoring mode: high_prob or high_convexity"
    )
    top_n: int = Field(default=20, ge=1, le=50, description="Number of results")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol is alphanumeric and reasonable length."""
        v = v.strip().upper()
        if not re.match(r"^[A-Z]{1,5}$", v):
            raise ValueError("Symbol must be 1-5 uppercase letters")
        return v


class LEAPSContract(BaseModel):
    """Single LEAPS contract data."""

    contract_symbol: str
    expiration: str
    strike: float
    target_price: float
    premium: float
    cost: float
    payoff_target: float
    roi_target: float
    ease_score: float
    roi_score: float
    score: float
    implied_volatility: Optional[float] = None
    open_interest: Optional[int] = None


class LEAPSResponse(BaseModel):
    """Response model for LEAPS ranking."""

    symbol: str
    underlying_price: float
    target_price: float
    target_pct: float
    mode: str
    contracts: List[LEAPSContract]
    timestamp: str


class ROISimulatorRequest(BaseModel):
    """Request model for ROI simulation."""

    strike: float = Field(..., gt=0, description="Option strike price")
    premium: float = Field(..., gt=0, description="Option premium per share")
    underlying_price: float = Field(..., gt=0, description="Current underlying price")
    target_prices: List[float] = Field(
        ...,
        min_length=1,
        description="List of target prices to simulate"
    )
    contract_size: int = Field(default=100, description="Contract multiplier")


class ROISimulatorResult(BaseModel):
    """Single simulation result."""

    target_price: float
    price_change_pct: float
    intrinsic_value: float
    payoff: float
    profit: float
    roi_pct: float


class ROISimulatorResponse(BaseModel):
    """Response model for ROI simulation."""

    strike: float
    premium: float
    cost: float
    underlying_price: float
    results: List[ROISimulatorResult]


class TickerInfo(BaseModel):
    """Basic ticker information."""

    symbol: str
    name: str
    default_target_pct: float


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
