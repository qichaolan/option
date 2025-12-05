"""
Configuration classes for the broken-wing condor screener.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ScoringWeights:
    """
    Configurable weights for the multi-factor scoring model.

    All weights should sum to 1.0 for normalized final scores.
    """
    risk_weight: float = 0.25
    credit_weight: float = 0.20
    skew_weight: float = 0.20
    call_weight: float = 0.10
    rrr_weight: float = 0.10
    ev_weight: float = 0.10
    pop_weight: float = 0.05

    def __post_init__(self) -> None:
        total = (
            self.risk_weight
            + self.credit_weight
            + self.skew_weight
            + self.call_weight
            + self.rrr_weight
            + self.ev_weight
            + self.pop_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.4f}")

    def validate(self) -> bool:
        """Validate all weights are non-negative."""
        return all(
            w >= 0
            for w in [
                self.risk_weight,
                self.credit_weight,
                self.skew_weight,
                self.call_weight,
                self.rrr_weight,
                self.ev_weight,
                self.pop_weight,
            ]
        )


DirectionType = Literal["auto", "bullish", "neutral", "convexity"]


@dataclass
class CondorConfig:
    """
    Configuration for the broken-wing condor screener.

    Attributes:
        symbol: Underlying ticker symbol (default: "SPY")
        min_dte: Minimum days to expiration (default: 3)
        max_dte: Maximum days to expiration (default: 14)
        max_call_cost: Maximum cost of call spread in dollars (default: 0.05)
        min_put_credit_pct: Minimum put credit as % of spread width (default: 0.90)
        top_n: Number of top trades to return (default: 20)
        direction: Strategy direction preference (default: "auto")
        put_spread_width_min: Minimum put spread width in points (default: 5)
        put_spread_width_max: Maximum put spread width in points (default: 15)
        call_spread_width: Fixed call spread width in points (default: 10)
        max_loss_per_contract: Maximum acceptable loss per contract (default: 50)
        scoring_weights: Custom scoring weights (default: standard weights)
    """
    symbol: str = "SPY"
    min_dte: int = 3
    max_dte: int = 14
    max_call_cost: float = 0.05
    min_put_credit_pct: float = 0.90
    top_n: int = 20
    direction: DirectionType = "auto"
    put_spread_width_min: int = 5
    put_spread_width_max: int = 15
    call_spread_width: int = 10
    max_loss_per_contract: float = 50.0
    safety_margin_pct: float = 0.03  # Long put must be at least this % below spot
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.min_dte < 0:
            raise ValueError("min_dte must be non-negative")
        if self.max_dte < self.min_dte:
            raise ValueError("max_dte must be >= min_dte")
        if self.max_call_cost < 0:
            raise ValueError("max_call_cost must be non-negative")
        if not 0 <= self.min_put_credit_pct <= 1:
            raise ValueError("min_put_credit_pct must be between 0 and 1")
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if self.put_spread_width_min < 1:
            raise ValueError("put_spread_width_min must be at least 1")
        if self.put_spread_width_max < self.put_spread_width_min:
            raise ValueError("put_spread_width_max must be >= put_spread_width_min")
        if self.call_spread_width < 1:
            raise ValueError("call_spread_width must be at least 1")
        if self.direction not in ("auto", "bullish", "neutral", "convexity"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if not 0 <= self.safety_margin_pct <= 0.20:
            raise ValueError("safety_margin_pct must be between 0 and 0.20 (20%)")
