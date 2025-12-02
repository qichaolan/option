"""Pydantic models for LEAPS Ranker API."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re

# Strict symbol pattern: only uppercase letters, 1-5 chars
# This prevents injection attacks via symbol field
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def validate_ticker_symbol(v: str) -> str:
    """
    Validate and sanitize ticker symbol.

    Ensures the symbol is alphanumeric uppercase only (1-5 characters).
    Prevents potential injection via symbol field.
    """
    if not isinstance(v, str):
        raise ValueError("Symbol must be a string")

    # Strip whitespace and convert to uppercase
    v = v.strip().upper()

    # Check length first (fast fail)
    if len(v) == 0 or len(v) > 5:
        raise ValueError("Symbol must be 1-5 characters")

    # Validate pattern
    if not SYMBOL_PATTERN.match(v):
        raise ValueError("Symbol must contain only uppercase letters (A-Z)")

    return v


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
        return validate_ticker_symbol(v)


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


# Credit Spreads Models

class CreditSpreadRequest(BaseModel):
    """Request model for credit spread screening."""

    symbol: str = Field(..., description="Ticker symbol (e.g., SPY, QQQ)")
    min_dte: int = Field(default=14, ge=7, le=60, description="Minimum days to expiration")
    max_dte: int = Field(default=30, ge=7, le=90, description="Maximum days to expiration")
    min_delta: float = Field(default=0.08, ge=0.05, le=0.40, description="Minimum short leg delta")
    max_delta: float = Field(default=0.35, ge=0.10, le=0.50, description="Maximum short leg delta")
    max_width: float = Field(default=10.0, ge=1.0, le=50.0, description="Maximum spread width in dollars")
    min_roc: float = Field(default=0.20, ge=0.05, le=1.0, description="Minimum return on capital")
    spread_type: Optional[Literal["PCS", "CCS", "ALL"]] = Field(
        default="ALL",
        description="Spread type filter: PCS (Put Credit Spread), CCS (Call Credit Spread), or ALL"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol using shared validation function."""
        return validate_ticker_symbol(v)


class CreditSpreadResult(BaseModel):
    """Single credit spread result."""

    symbol: str
    spread_type: str  # "PCS" or "CCS"
    expiration: str
    dte: int
    short_strike: float
    long_strike: float
    width: float
    credit: float
    max_loss: float
    roc: float  # Return on capital as decimal (0.25 = 25%)
    short_delta: float
    delta_estimated: bool
    prob_profit: float  # Probability of profit as decimal
    iv: float
    ivp: float  # IV percentile
    underlying_price: float
    break_even: float
    break_even_distance_pct: float
    liquidity_score: float
    slippage_score: float
    total_score: float


class CreditSpreadResponse(BaseModel):
    """Response model for credit spread screening."""

    symbol: str
    underlying_price: float
    ivp: float
    spread_type_filter: str
    total_pcs: int
    total_ccs: int
    spreads: List[CreditSpreadResult]
    timestamp: str


# Credit Spread Simulator Models

class CreditSpreadSimulatorRequest(BaseModel):
    """Request model for credit spread P/L simulation."""

    symbol: str = Field(..., description="Ticker symbol (e.g., SPY, QQQ)")
    spread_type: Literal["PCS", "CCS"] = Field(
        ..., description="Spread type: PCS (Put Credit Spread) or CCS (Call Credit Spread)"
    )
    expiration: str = Field(..., description="Expiration date (e.g., 2025-12-19)")
    short_strike: float = Field(..., gt=0, description="Short leg strike price")
    long_strike: float = Field(..., gt=0, description="Long leg strike price")
    net_credit: float = Field(..., gt=0, description="Net credit received per share")
    underlying_price_now: float = Field(..., gt=0, description="Current underlying price")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol using shared validation function."""
        return validate_ticker_symbol(v)


class CreditSpreadSimulatorPoint(BaseModel):
    """Single simulation point for credit spread P/L."""

    pct_move: float  # Percentage move (e.g., -5.0, 0.0, 5.0)
    underlying_price: float  # Price at expiration
    pl_per_spread: float  # P/L for one spread (100 shares)


class CreditSpreadSimulatorSummary(BaseModel):
    """Summary metrics for credit spread simulation."""

    max_gain: float  # Maximum gain (net credit * 100)
    max_loss: float  # Maximum loss
    breakeven_price: float  # Breakeven price at expiration
    breakeven_pct: float  # Breakeven as % move from current price


class CreditSpreadSimulatorResponse(BaseModel):
    """Response model for credit spread P/L simulation."""

    symbol: str
    spread_type: str
    expiration: str
    short_strike: float
    long_strike: float
    net_credit: float
    underlying_price_now: float
    summary: CreditSpreadSimulatorSummary
    points: List[CreditSpreadSimulatorPoint]
