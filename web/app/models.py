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


# =============================================================================
# AI Explainer Models
# =============================================================================

# Whitelisted page IDs for AI Explainer
VALID_PAGE_IDS = {"leaps_ranker", "credit_spread_screener", "iron_condor_screener"}

# Whitelisted context types for AI Explainer
VALID_CONTEXT_TYPES = {"roi_simulator", "spread_simulator", "options_analysis"}

# Maximum metadata JSON size in bytes (10KB)
MAX_METADATA_SIZE = 10 * 1024


class AiExplainerRequest(BaseModel):
    """Request model for AI Explainer endpoint."""

    pageId: str = Field(
        ...,
        description="Page identifier (e.g., 'leaps_ranker')",
        min_length=1,
        max_length=50,
    )
    contextType: str = Field(
        ...,
        description="Context type (e.g., 'roi_simulator')",
        min_length=1,
        max_length=50,
    )
    timestamp: str = Field(
        ...,
        description="Client timestamp in ISO format",
        min_length=1,
        max_length=50,
    )
    metadata: dict = Field(
        ...,
        description="Domain-specific data for AI analysis",
    )

    @field_validator("pageId")
    @classmethod
    def validate_page_id(cls, v: str) -> str:
        """Validate pageId is whitelisted."""
        v = v.strip().lower()
        if v not in VALID_PAGE_IDS:
            raise ValueError(f"Invalid pageId. Must be one of: {VALID_PAGE_IDS}")
        return v

    @field_validator("contextType")
    @classmethod
    def validate_context_type(cls, v: str) -> str:
        """Validate contextType is whitelisted."""
        v = v.strip().lower()
        if v not in VALID_CONTEXT_TYPES:
            raise ValueError(f"Invalid contextType. Must be one of: {VALID_CONTEXT_TYPES}")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_size(cls, v: dict) -> dict:
        """Validate metadata size doesn't exceed limit."""
        import json
        metadata_json = json.dumps(v)
        if len(metadata_json) > MAX_METADATA_SIZE:
            raise ValueError(f"Metadata exceeds maximum size of {MAX_METADATA_SIZE} bytes")
        return v


class AiExplainerKeyInsight(BaseModel):
    """Single key insight from AI explanation."""

    title: str = Field(..., description="Short title for the insight")
    description: str = Field(..., description="Detailed description")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        default="neutral",
        description="Sentiment indicator for the insight",
    )


class AiExplainerRisk(BaseModel):
    """Risk item from AI explanation."""

    risk: str = Field(..., description="Description of the risk")
    severity: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Severity level of the risk",
    )


class AiExplainerWatchItem(BaseModel):
    """Watch item from AI explanation."""

    item: str = Field(..., description="What to watch for")
    trigger: Optional[str] = Field(None, description="Trigger condition")


class AiExplainerScenario(BaseModel):
    """Historical scenario analysis with narrative format."""

    min_annual_return: str = Field(..., description="Minimum annual return for header (e.g., +16.00%)")
    projected_price_target: str = Field(..., description="Narrative about the compounded price target")
    payoff_realism: str = Field(..., description="Narrative about historical likelihood and reasonableness")
    option_payoff: str = Field(..., description="Narrative about expected ROI and profit")


class AiExplainerScenarios(BaseModel):
    """Historical scenarios for medium and strong increases."""

    medium_increase: Optional[AiExplainerScenario] = Field(
        None, description="50-70th percentile scenario"
    )
    strong_increase: Optional[AiExplainerScenario] = Field(
        None, description="70-100th percentile scenario"
    )


class AiExplainerTradeMechanics(BaseModel):
    """Trade mechanics for credit spread or iron condor."""

    structure: str = Field(..., description="Trade structure description")
    credit_received: str = Field(..., description="Credit received per share and total")
    margin_requirement: str = Field(..., description="Margin/capital requirement")
    breakeven: Optional[str] = Field(None, description="Breakeven price (credit spread)")
    breakevens: Optional[str] = Field(None, description="Breakeven prices (iron condor)")


class AiExplainerKeyMetricValue(BaseModel):
    """Key metric with value and condition."""

    value: str = Field(..., description="Metric value")
    condition: Optional[str] = Field(None, description="Condition for this value")


class AiExplainerKeyMetrics(BaseModel):
    """Key metrics for credit spread or iron condor."""

    max_profit: Optional[AiExplainerKeyMetricValue] = Field(None, description="Maximum profit")
    max_loss: Optional[AiExplainerKeyMetricValue] = Field(None, description="Maximum loss")
    risk_reward_ratio: Optional[str] = Field(None, description="Risk/reward ratio")
    probability_of_profit: Optional[str] = Field(None, description="Probability of profit")


class AiExplainerVisualization(BaseModel):
    """Profit/loss zone visualization."""

    profit_zone: Optional[str] = Field(None, description="Profit zone description")
    loss_zone: Optional[str] = Field(None, description="Loss zone (credit spread)")
    lower_loss_zone: Optional[str] = Field(None, description="Lower loss zone (iron condor)")
    upper_loss_zone: Optional[str] = Field(None, description="Upper loss zone (iron condor)")
    transition_zone: Optional[str] = Field(None, description="Transition zone (credit spread)")
    transition_zones: Optional[str] = Field(None, description="Transition zones (iron condor)")


class AiExplainerStrategyOutcome(BaseModel):
    """Strategy outcome for a market scenario."""

    scenario: str = Field(..., description="Market scenario description")
    result: str = Field(..., description="Expected result")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        default="neutral", description="Outcome sentiment"
    )


class AiExplainerStrategyAnalysis(BaseModel):
    """Strategy analysis for different market scenarios."""

    bullish_outcome: Optional[AiExplainerStrategyOutcome] = Field(None, description="Bullish outcome")
    neutral_outcome: Optional[AiExplainerStrategyOutcome] = Field(None, description="Neutral outcome")
    bearish_outcome: Optional[AiExplainerStrategyOutcome] = Field(None, description="Bearish outcome")
    extreme_move_outcome: Optional[AiExplainerStrategyOutcome] = Field(
        None, description="Extreme move outcome (iron condor)"
    )


class AiExplainerRiskManagement(BaseModel):
    """Risk management guidelines."""

    early_exit_trigger: Optional[str] = Field(None, description="When to exit early")
    adjustment_options: Optional[str] = Field(None, description="Adjustment strategies")
    worst_case: Optional[str] = Field(None, description="Worst case scenario")


class AiExplainerContent(BaseModel):
    """Structured content from AI explanation."""

    summary: str = Field(..., description="Brief summary of the analysis")
    key_insights: List[AiExplainerKeyInsight] = Field(
        default_factory=list,
        description="Key insights from the analysis (3-5 items)",
    )
    scenarios: Optional[AiExplainerScenarios] = Field(
        None,
        description="Historical scenario analysis for medium and strong increases",
    )
    risks: List[AiExplainerRisk] = Field(
        default_factory=list,
        description="Risk factors to consider",
    )
    watch_items: List[AiExplainerWatchItem] = Field(
        default_factory=list,
        description="Items to watch going forward",
    )
    disclaimer: str = Field(
        default="This analysis is for educational purposes only and should not be considered financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.",
        description="Legal disclaimer",
    )
    # Credit Spread and Iron Condor specific fields
    strategy_name: Optional[str] = Field(None, description="Strategy name (e.g., 'Put Credit Spread on SPY')")
    trade_mechanics: Optional[AiExplainerTradeMechanics] = Field(
        None, description="Trade mechanics details"
    )
    key_metrics: Optional[AiExplainerKeyMetrics] = Field(
        None, description="Key metrics (max profit, max loss, etc.)"
    )
    visualization: Optional[AiExplainerVisualization] = Field(
        None, description="Profit/loss zone visualization"
    )
    strategy_analysis: Optional[AiExplainerStrategyAnalysis] = Field(
        None, description="Strategy analysis for different market scenarios"
    )
    risk_management: Optional[AiExplainerRiskManagement] = Field(
        None, description="Risk management guidelines"
    )


class AiExplainerResponse(BaseModel):
    """Response model for AI Explainer endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    pageId: str = Field(..., description="Echo of the page identifier")
    contextType: str = Field(..., description="Echo of the context type")
    content: Optional[AiExplainerContent] = Field(
        None,
        description="Structured explanation content",
    )
    cached: bool = Field(
        default=False,
        description="Whether this response was served from cache",
    )
    cachedAt: Optional[str] = Field(
        None,
        description="Timestamp when the response was cached (ISO format)",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if success is False",
    )
    timestamp: str = Field(..., description="Server timestamp")
