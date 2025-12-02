"""
Iron Condor Module

Generates, scores, and ranks Iron Condor candidates from compatible PCS + CCS combinations.
Uses existing credit spread outputs as building blocks without repricing individual options.

An Iron Condor consists of:
- 1 Put Credit Spread (PCS) - bullish, lower side
- 1 Call Credit Spread (CCS) - bearish, upper side
- Same underlying and expiration

Author: Option Chain Analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# ROC (Return on Capital) scoring
ROC_TARGET_FOR_FULL_SCORE = 0.30  # 30% ROC → score 1.0

# Width zone scoring
WIDTH_PCT_MAX_FOR_SCORE = 0.10  # 10% of underlying → score 1.0

# Liquidity scoring thresholds
MAX_SPREAD_PCT_FOR_FULL_LIQUIDITY = 0.01  # 1% bid-ask spread
MIN_VOLUME_FOR_FULL_LIQUIDITY = 1000
MIN_OI_FOR_FULL_LIQUIDITY = 1000

# Tail risk scoring
DELTA_MAX_FOR_TAILRISK = 0.40  # Delta 0.40 → tail risk 1.0

# Total score weights
WEIGHT_POP = 0.30
WEIGHT_ROC = 0.25
WEIGHT_WIDTH = 0.20
WEIGHT_LIQUIDITY = 0.15
WEIGHT_TAILRISK = 0.10  # Subtracted (penalty)

# Contract multiplier
CONTRACT_MULTIPLIER = 100


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max bounds."""
    return max(min_val, min(value, max_val))


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class CreditSpread:
    """
    Represents a single credit spread (PCS or CCS).

    For PCS (Put Credit Spread):
        - short_strike > long_strike
        - width = short_strike - long_strike

    For CCS (Call Credit Spread):
        - short_strike < long_strike
        - width = long_strike - short_strike
    """
    underlying: str
    expiration: str
    spread_type: Literal["PCS", "CCS"]
    short_strike: float
    long_strike: float
    credit: float  # Net credit per share
    short_delta: float  # Delta of the short leg (absolute value stored)
    bid_ask_spread: float  # Combined bid-ask spread of the spread
    volume: int
    open_interest: int

    @property
    def width(self) -> float:
        """Calculate the width of the spread."""
        if self.spread_type == "PCS":
            return self.short_strike - self.long_strike
        else:  # CCS
            return self.long_strike - self.short_strike


@dataclass
class IronCondorLeg:
    """
    Wraps a CreditSpread as a leg of an Iron Condor.

    Attributes:
        spread: The underlying credit spread
        side: "put" for PCS leg, "call" for CCS leg
    """
    spread: CreditSpread
    side: Literal["put", "call"]

    def __post_init__(self) -> None:
        # Validate that PCS is put side and CCS is call side
        if self.side == "put" and self.spread.spread_type != "PCS":
            raise ValueError("Put leg must use a PCS spread")
        if self.side == "call" and self.spread.spread_type != "CCS":
            raise ValueError("Call leg must use a CCS spread")


@dataclass
class IronCondor:
    """
    Represents an Iron Condor position.

    An Iron Condor consists of:
    - A put credit spread (lower/bullish side)
    - A call credit spread (upper/bearish side)
    - Same underlying and expiration

    All derived fields are computed in __post_init__.
    """
    put_leg: IronCondorLeg
    call_leg: IronCondorLeg
    underlying_price: float
    days_to_expiration: int

    # Derived fields - initialized after construction
    total_credit: float = field(init=False, default=0.0)
    max_loss_per_share: float = field(init=False, default=0.0)
    max_profit_dollars: float = field(init=False, default=0.0)
    max_loss_dollars: float = field(init=False, default=0.0)

    # Scoring fields
    roc_raw: float = field(init=False, default=0.0)
    roc_score: float = field(init=False, default=0.0)
    pop: float = field(init=False, default=0.0)
    width_zone_pct: float = field(init=False, default=0.0)
    width_score: float = field(init=False, default=0.0)
    liquidity_score: float = field(init=False, default=0.0)
    tail_risk: float = field(init=False, default=0.0)
    total_score: float = field(init=False, default=0.0)

    # Breakeven fields
    breakeven_low: float = field(init=False, default=0.0)
    breakeven_high: float = field(init=False, default=0.0)
    distance_to_BE_low_pct: float = field(init=False, default=0.0)
    distance_to_BE_high_pct: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Compute all derived fields after initialization."""
        # Validate same underlying and expiration
        if self.put_leg.spread.underlying != self.call_leg.spread.underlying:
            raise ValueError("Put and call legs must have the same underlying")
        if self.put_leg.spread.expiration != self.call_leg.spread.expiration:
            raise ValueError("Put and call legs must have the same expiration")

        # Validate proper condor shape: short_put < short_call
        if self.short_put_strike >= self.short_call_strike:
            raise ValueError(
                f"Invalid condor shape: short_put ({self.short_put_strike}) "
                f"must be < short_call ({self.short_call_strike})"
            )

        # Compute all metrics
        compute_max_loss_and_credit(self)
        compute_breakevens(self)
        compute_roc_score(self)
        compute_pop(self)
        compute_width_score(self)
        compute_liquidity_score(self)
        compute_tail_risk(self)
        compute_total_score(self)

    # Convenience properties for accessing strikes
    @property
    def short_put_strike(self) -> float:
        return self.put_leg.spread.short_strike

    @property
    def long_put_strike(self) -> float:
        return self.put_leg.spread.long_strike

    @property
    def short_call_strike(self) -> float:
        return self.call_leg.spread.short_strike

    @property
    def long_call_strike(self) -> float:
        return self.call_leg.spread.long_strike

    @property
    def credit_pcs(self) -> float:
        return self.put_leg.spread.credit

    @property
    def credit_ccs(self) -> float:
        return self.call_leg.spread.credit

    @property
    def underlying(self) -> str:
        return self.put_leg.spread.underlying

    @property
    def expiration(self) -> str:
        return self.put_leg.spread.expiration

    @property
    def put_width(self) -> float:
        return self.put_leg.spread.width

    @property
    def call_width(self) -> float:
        return self.call_leg.spread.width


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================


def compute_max_loss_and_credit(condor: IronCondor) -> None:
    """
    Compute max loss and total credit for an Iron Condor.

    Per share:
        total_credit = credit_PCS + credit_CCS
        W = max(put_width, call_width)
        max_loss_per_share = max(W - total_credit, 0)

    Per contract (100 shares):
        max_profit_dollars = total_credit * 100
        max_loss_dollars = max_loss_per_share * 100
    """
    condor.total_credit = condor.credit_pcs + condor.credit_ccs

    # Margin width is the maximum of the two wing widths
    margin_width = max(condor.put_width, condor.call_width)

    condor.max_loss_per_share = max(margin_width - condor.total_credit, 0.0)
    condor.max_profit_dollars = condor.total_credit * CONTRACT_MULTIPLIER
    condor.max_loss_dollars = condor.max_loss_per_share * CONTRACT_MULTIPLIER


def compute_breakevens(condor: IronCondor) -> None:
    """
    Compute breakeven prices for an Iron Condor.

    Lower breakeven: short_put - total_credit
    Upper breakeven: short_call + total_credit

    Also computes distance to breakevens as percentage of current price.
    """
    condor.breakeven_low = condor.short_put_strike - condor.total_credit
    condor.breakeven_high = condor.short_call_strike + condor.total_credit

    if condor.underlying_price > 0:
        condor.distance_to_BE_low_pct = (
            (condor.underlying_price - condor.breakeven_low) / condor.underlying_price
        )
        condor.distance_to_BE_high_pct = (
            (condor.breakeven_high - condor.underlying_price) / condor.underlying_price
        )
    else:
        condor.distance_to_BE_low_pct = 0.0
        condor.distance_to_BE_high_pct = 0.0


def compute_roc_score(condor: IronCondor) -> None:
    """
    Compute Return on Capital score.

    roc_raw = total_credit / max_loss_per_share (if max_loss_per_share > 0)
    roc_score = min(roc_raw / ROC_TARGET_FOR_FULL_SCORE, 1.0)
    """
    if condor.max_loss_per_share > 0:
        condor.roc_raw = condor.total_credit / condor.max_loss_per_share
    else:
        condor.roc_raw = 0.0

    condor.roc_score = clamp(condor.roc_raw / ROC_TARGET_FOR_FULL_SCORE, 0.0, 1.0)


def compute_pop(condor: IronCondor) -> None:
    """
    Compute Probability of Profit using delta-based approximation.

    POP ≈ 1 - (|delta_put| + |delta_call|)

    This approximates that POP is high when both short legs are
    far out-of-the-money (low deltas).
    """
    delta_put = abs(condor.put_leg.spread.short_delta)
    delta_call = abs(condor.call_leg.spread.short_delta)

    raw_pop = 1.0 - (delta_put + delta_call)
    condor.pop = clamp(raw_pop, 0.0, 1.0)


def compute_width_score(condor: IronCondor) -> None:
    """
    Compute width zone score based on profit zone width.

    Profit zone spans from short_put to short_call.
    Zone width as percentage of underlying:
        Z_pct = (short_call - short_put) / underlying_price

    Score is normalized from [0, WIDTH_PCT_MAX_FOR_SCORE] → [0, 1]
    """
    zone_width = condor.short_call_strike - condor.short_put_strike

    if condor.underlying_price > 0:
        condor.width_zone_pct = zone_width / condor.underlying_price
    else:
        condor.width_zone_pct = 0.0

    if condor.width_zone_pct <= 0:
        condor.width_score = 0.0
    else:
        condor.width_score = clamp(
            condor.width_zone_pct / WIDTH_PCT_MAX_FOR_SCORE, 0.0, 1.0
        )


def _compute_leg_liquidity(spread: CreditSpread) -> float:
    """
    Compute liquidity score for a single credit spread leg.

    Components:
    - Spread score: based on bid-ask spread as % of approx mid price
    - Volume score: based on volume vs threshold
    - OI score: based on open interest vs threshold

    Returns average of the three component scores.
    """
    # Approximate mid price to avoid division by zero
    approx_mid = max(spread.short_strike * 0.01, 0.01)

    # Spread percentage score
    spread_pct = spread.bid_ask_spread / approx_mid
    spread_score = 1.0 - clamp(spread_pct / MAX_SPREAD_PCT_FOR_FULL_LIQUIDITY, 0.0, 1.0)

    # Volume score
    volume_score = clamp(spread.volume / MIN_VOLUME_FOR_FULL_LIQUIDITY, 0.0, 1.0)

    # Open interest score
    oi_score = clamp(spread.open_interest / MIN_OI_FOR_FULL_LIQUIDITY, 0.0, 1.0)

    return (spread_score + volume_score + oi_score) / 3.0


def compute_liquidity_score(condor: IronCondor) -> None:
    """
    Compute overall liquidity score for the Iron Condor.

    Average of put leg and call leg liquidity scores.
    """
    put_liquidity = _compute_leg_liquidity(condor.put_leg.spread)
    call_liquidity = _compute_leg_liquidity(condor.call_leg.spread)

    condor.liquidity_score = (put_liquidity + call_liquidity) / 2.0


def compute_tail_risk(condor: IronCondor) -> None:
    """
    Compute tail risk score based on average short-leg delta.

    Higher delta = closer to ATM = higher risk of breach.

    tail_risk = avg(|delta_put|, |delta_call|) / DELTA_MAX_FOR_TAILRISK
    """
    delta_put = abs(condor.put_leg.spread.short_delta)
    delta_call = abs(condor.call_leg.spread.short_delta)

    delta_avg = (delta_put + delta_call) / 2.0
    condor.tail_risk = clamp(delta_avg / DELTA_MAX_FOR_TAILRISK, 0.0, 1.0)


def compute_total_score(condor: IronCondor) -> None:
    """
    Compute total score for ranking Iron Condors.

    TotalScore = 0.30 * POP
               + 0.25 * ROC_score
               + 0.20 * WidthScore
               + 0.15 * Liquidity
               - 0.10 * TailRisk

    Clamped to [0, 1].
    """
    raw_score = (
        WEIGHT_POP * condor.pop
        + WEIGHT_ROC * condor.roc_score
        + WEIGHT_WIDTH * condor.width_score
        + WEIGHT_LIQUIDITY * condor.liquidity_score
        - WEIGHT_TAILRISK * condor.tail_risk
    )

    condor.total_score = clamp(raw_score, 0.0, 1.0)


# =============================================================================
# PAYOFF AND ROI FUNCTIONS
# =============================================================================


def payoff_per_contract(condor: IronCondor, s_t: float) -> float:
    """
    Calculate Iron Condor payoff at expiration price S_T.

    Per contract (100 shares):

    PCS payoff:
        payoff_put = credit_PCS * 100
                   - max(0, K_short_put - S_T) * 100
                   + max(0, K_long_put - S_T) * 100

    CCS payoff:
        payoff_call = credit_CCS * 100
                    - max(0, S_T - K_short_call) * 100
                    + max(0, S_T - K_long_call) * 100

    Iron Condor payoff:
        payoff_IC = payoff_put + payoff_call

    Args:
        condor: The Iron Condor position
        s_t: Underlying price at expiration

    Returns:
        Payoff in dollars per contract
    """
    # PCS (Put Credit Spread) payoff
    payoff_put = (
        condor.credit_pcs * CONTRACT_MULTIPLIER
        - max(0.0, condor.short_put_strike - s_t) * CONTRACT_MULTIPLIER
        + max(0.0, condor.long_put_strike - s_t) * CONTRACT_MULTIPLIER
    )

    # CCS (Call Credit Spread) payoff
    payoff_call = (
        condor.credit_ccs * CONTRACT_MULTIPLIER
        - max(0.0, s_t - condor.short_call_strike) * CONTRACT_MULTIPLIER
        + max(0.0, s_t - condor.long_call_strike) * CONTRACT_MULTIPLIER
    )

    return payoff_put + payoff_call


def roi_at_price(condor: IronCondor, s_t: float) -> float:
    """
    Calculate ROI at a specific expiration price.

    ROI(S_T) = payoff_IC(S_T) / max_loss_dollars

    Returns 0 if max_loss_dollars <= 0.

    Args:
        condor: The Iron Condor position
        s_t: Underlying price at expiration

    Returns:
        ROI as a decimal (e.g., 0.25 = 25%)
    """
    if condor.max_loss_dollars <= 0:
        return 0.0

    payoff = payoff_per_contract(condor, s_t)
    return payoff / condor.max_loss_dollars


def payoff_roi_curve(
    condor: IronCondor,
    move_low_pct: float = -0.05,
    move_high_pct: float = 0.05,
    step_pct: float = 0.01,
) -> list[dict]:
    """
    Generate payoff and ROI curve for a range of price moves.

    Args:
        condor: The Iron Condor position
        move_low_pct: Lower bound of price move (e.g., -0.05 = -5%)
        move_high_pct: Upper bound of price move (e.g., 0.05 = +5%)
        step_pct: Step size (e.g., 0.01 = 1%)

    Returns:
        List of dicts with keys: move_pct, price, payoff, roi
    """
    results: list[dict] = []

    # Generate move percentages
    current_move = move_low_pct
    while current_move <= move_high_pct + step_pct / 2:  # Include endpoint
        price = condor.underlying_price * (1.0 + current_move)
        payoff = payoff_per_contract(condor, price)
        roi = roi_at_price(condor, price)

        results.append({
            "move_pct": round(current_move, 4),
            "price": round(price, 2),
            "payoff": round(payoff, 2),
            "roi": round(roi, 4),
        })

        current_move += step_pct

    return results


# =============================================================================
# IRON CONDOR BUILDING AND RANKING
# =============================================================================


def build_iron_condors(
    put_spreads: list[CreditSpread],
    call_spreads: list[CreditSpread],
    underlying_price: float,
    days_to_expiration: int,
) -> list[IronCondor]:
    """
    Build all valid Iron Condor candidates from PCS and CCS combinations.

    Validity requirements:
    - Same underlying and expiration (already assumed by input)
    - PCS spread_type == "PCS", CCS spread_type == "CCS"
    - short_put_strike < short_call_strike (proper condor shape)

    Args:
        put_spreads: List of Put Credit Spreads
        call_spreads: List of Call Credit Spreads
        underlying_price: Current underlying price
        days_to_expiration: Days until expiration

    Returns:
        List of valid IronCondor objects with all metrics computed
    """
    condors: list[IronCondor] = []

    for pcs in put_spreads:
        if pcs.spread_type != "PCS":
            continue

        for ccs in call_spreads:
            if ccs.spread_type != "CCS":
                continue

            # Check same underlying and expiration
            if pcs.underlying != ccs.underlying:
                continue
            if pcs.expiration != ccs.expiration:
                continue

            # Check proper condor shape
            if pcs.short_strike >= ccs.short_strike:
                continue

            try:
                put_leg = IronCondorLeg(spread=pcs, side="put")
                call_leg = IronCondorLeg(spread=ccs, side="call")

                condor = IronCondor(
                    put_leg=put_leg,
                    call_leg=call_leg,
                    underlying_price=underlying_price,
                    days_to_expiration=days_to_expiration,
                )
                condors.append(condor)
            except ValueError:
                # Skip invalid combinations
                continue

    return condors


def rank_iron_condors(
    put_spreads: list[CreditSpread],
    call_spreads: list[CreditSpread],
    underlying_price: float,
    days_to_expiration: int,
    top_n: int = 20,
) -> list[IronCondor]:
    """
    Build, score, and rank Iron Condor candidates.

    Args:
        put_spreads: List of Put Credit Spreads
        call_spreads: List of Call Credit Spreads
        underlying_price: Current underlying price
        days_to_expiration: Days until expiration
        top_n: Number of top candidates to return

    Returns:
        Top N Iron Condors sorted by total_score descending
    """
    condors = build_iron_condors(
        put_spreads=put_spreads,
        call_spreads=call_spreads,
        underlying_price=underlying_price,
        days_to_expiration=days_to_expiration,
    )

    # Sort by total_score descending
    condors.sort(key=lambda c: c.total_score, reverse=True)

    return condors[:top_n]


# =============================================================================
# TESTS
# =============================================================================


def test_max_profit_and_loss() -> None:
    """
    Test max profit and max loss calculation.

    Build a symmetric Iron Condor with known parameters:
    - Underlying: 100
    - Put: short 95, long 90, credit 1.0 (width = 5)
    - Call: short 105, long 110, credit 1.0 (width = 5)

    Expected:
    - total_credit = 2.0
    - max_loss_per_share = 5 - 2 = 3
    - max_profit_dollars = 200
    - max_loss_dollars = 300
    """
    pcs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="PCS",
        short_strike=95.0,
        long_strike=90.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    ccs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="CCS",
        short_strike=105.0,
        long_strike=110.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    put_leg = IronCondorLeg(spread=pcs, side="put")
    call_leg = IronCondorLeg(spread=ccs, side="call")

    condor = IronCondor(
        put_leg=put_leg,
        call_leg=call_leg,
        underlying_price=100.0,
        days_to_expiration=30,
    )

    assert condor.total_credit == 2.0, f"Expected total_credit=2.0, got {condor.total_credit}"
    assert condor.max_loss_per_share == 3.0, f"Expected max_loss_per_share=3.0, got {condor.max_loss_per_share}"
    assert condor.max_profit_dollars == 200.0, f"Expected max_profit_dollars=200.0, got {condor.max_profit_dollars}"
    assert condor.max_loss_dollars == 300.0, f"Expected max_loss_dollars=300.0, got {condor.max_loss_dollars}"

    print("✓ test_max_profit_and_loss passed")


def test_deep_itm_downside_payoff() -> None:
    """
    Test payoff when underlying crashes (deep ITM downside).

    Using the same symmetric condor at S_T = 50:
    - Put spread: short 95 ITM by 45, long 90 ITM by 40
    - Call spread: both legs OTM (expire worthless)

    PCS payoff = 1.0*100 - max(0, 95-50)*100 + max(0, 90-50)*100
               = 100 - 4500 + 4000 = -400

    CCS payoff = 1.0*100 - max(0, 50-105)*100 + max(0, 50-110)*100
               = 100 - 0 + 0 = 100

    Total = -400 + 100 = -300 (max loss per contract)

    Wait, let me recalculate. With S_T=50, both put strikes are ITM:
    - Short put at 95: buyer exercises, we pay (95-50)*100 = 4500
    - Long put at 90: we exercise, we get (90-50)*100 = 4000
    - Net PCS loss = 4500 - 4000 = 500, but we collected 100 credit
    - PCS payoff = 100 - 500 = -400 (per contract)

    Actually with the formula:
    payoff_put = credit*100 - max(0, K_short - S_T)*100 + max(0, K_long - S_T)*100
               = 100 - 4500 + 4000 = -400

    CCS payoff = 100 - 0 + 0 = 100

    Total IC payoff = -400 + 100 = -300

    This equals -max_loss_dollars = -300. Correct!
    """
    pcs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="PCS",
        short_strike=95.0,
        long_strike=90.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    ccs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="CCS",
        short_strike=105.0,
        long_strike=110.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    put_leg = IronCondorLeg(spread=pcs, side="put")
    call_leg = IronCondorLeg(spread=ccs, side="call")

    condor = IronCondor(
        put_leg=put_leg,
        call_leg=call_leg,
        underlying_price=100.0,
        days_to_expiration=30,
    )

    payoff = payoff_per_contract(condor, s_t=50.0)

    # Max loss is -300 (we lose the max_loss_dollars)
    expected_payoff = -300.0
    assert abs(payoff - expected_payoff) < 0.01, f"Expected payoff={expected_payoff}, got {payoff}"

    print("✓ test_deep_itm_downside_payoff passed")


def test_deep_itm_upside_payoff() -> None:
    """
    Test payoff when underlying rallies hard (deep ITM upside).

    Using the same symmetric condor at S_T = 200:
    - Put spread: both legs OTM (expire worthless)
    - Call spread: short 105 ITM by 95, long 110 ITM by 90

    PCS payoff = 1.0*100 - max(0, 95-200)*100 + max(0, 90-200)*100
               = 100 - 0 + 0 = 100

    CCS payoff = 1.0*100 - max(0, 200-105)*100 + max(0, 200-110)*100
               = 100 - 9500 + 9000 = -400

    Total IC payoff = 100 + (-400) = -300 (max loss per contract)
    """
    pcs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="PCS",
        short_strike=95.0,
        long_strike=90.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    ccs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="CCS",
        short_strike=105.0,
        long_strike=110.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    put_leg = IronCondorLeg(spread=pcs, side="put")
    call_leg = IronCondorLeg(spread=ccs, side="call")

    condor = IronCondor(
        put_leg=put_leg,
        call_leg=call_leg,
        underlying_price=100.0,
        days_to_expiration=30,
    )

    payoff = payoff_per_contract(condor, s_t=200.0)

    # Max loss is -300 (we lose the max_loss_dollars)
    expected_payoff = -300.0
    assert abs(payoff - expected_payoff) < 0.01, f"Expected payoff={expected_payoff}, got {payoff}"

    print("✓ test_deep_itm_upside_payoff passed")


def test_max_profit_in_zone() -> None:
    """
    Test payoff when underlying expires between short strikes (max profit zone).

    At S_T = 100 (between 95 and 105):
    - Both spreads expire worthless
    - We keep the full credit

    PCS payoff = 100 - 0 + 0 = 100
    CCS payoff = 100 - 0 + 0 = 100
    Total = 200 (max_profit_dollars)
    """
    pcs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="PCS",
        short_strike=95.0,
        long_strike=90.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    ccs = CreditSpread(
        underlying="TEST",
        expiration="2025-12-19",
        spread_type="CCS",
        short_strike=105.0,
        long_strike=110.0,
        credit=1.0,
        short_delta=0.15,
        bid_ask_spread=0.05,
        volume=500,
        open_interest=1000,
    )

    put_leg = IronCondorLeg(spread=pcs, side="put")
    call_leg = IronCondorLeg(spread=ccs, side="call")

    condor = IronCondor(
        put_leg=put_leg,
        call_leg=call_leg,
        underlying_price=100.0,
        days_to_expiration=30,
    )

    payoff = payoff_per_contract(condor, s_t=100.0)

    expected_payoff = 200.0  # max_profit_dollars
    assert abs(payoff - expected_payoff) < 0.01, f"Expected payoff={expected_payoff}, got {payoff}"

    print("✓ test_max_profit_in_zone passed")


def run_tests() -> None:
    """Run all unit tests."""
    print("\nRunning Iron Condor tests...\n")

    test_max_profit_and_loss()
    test_deep_itm_downside_payoff()
    test_deep_itm_upside_payoff()
    test_max_profit_in_zone()

    print("\n✓ All tests passed.\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def main() -> None:
    """Example usage demonstrating Iron Condor ranking and payoff curves."""

    print("=" * 80)
    print("IRON CONDOR RANKER - Example Usage")
    print("=" * 80)

    # Mock data for QQQ 2025-12-19
    underlying = "QQQ"
    expiration = "2025-12-19"
    underlying_price = 520.0
    days_to_expiration = 45

    # Create mock PCS spreads (put credit spreads - lower side)
    put_spreads = [
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="PCS",
            short_strike=490.0,
            long_strike=485.0,
            credit=0.85,
            short_delta=0.12,
            bid_ask_spread=0.08,
            volume=1200,
            open_interest=5500,
        ),
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="PCS",
            short_strike=495.0,
            long_strike=490.0,
            credit=1.10,
            short_delta=0.15,
            bid_ask_spread=0.10,
            volume=1500,
            open_interest=6200,
        ),
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="PCS",
            short_strike=500.0,
            long_strike=495.0,
            credit=1.35,
            short_delta=0.18,
            bid_ask_spread=0.12,
            volume=1800,
            open_interest=7000,
        ),
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="PCS",
            short_strike=505.0,
            long_strike=500.0,
            credit=1.65,
            short_delta=0.22,
            bid_ask_spread=0.15,
            volume=2000,
            open_interest=8000,
        ),
    ]

    # Create mock CCS spreads (call credit spreads - upper side)
    call_spreads = [
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="CCS",
            short_strike=540.0,
            long_strike=545.0,
            credit=0.90,
            short_delta=0.12,
            bid_ask_spread=0.08,
            volume=1100,
            open_interest=5000,
        ),
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="CCS",
            short_strike=545.0,
            long_strike=550.0,
            credit=0.75,
            short_delta=0.10,
            bid_ask_spread=0.06,
            volume=900,
            open_interest=4500,
        ),
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="CCS",
            short_strike=535.0,
            long_strike=540.0,
            credit=1.15,
            short_delta=0.15,
            bid_ask_spread=0.10,
            volume=1300,
            open_interest=5500,
        ),
        CreditSpread(
            underlying=underlying,
            expiration=expiration,
            spread_type="CCS",
            short_strike=530.0,
            long_strike=535.0,
            credit=1.45,
            short_delta=0.20,
            bid_ask_spread=0.12,
            volume=1600,
            open_interest=6500,
        ),
    ]

    # Rank Iron Condors
    print(f"\nUnderlying: {underlying} @ ${underlying_price:.2f}")
    print(f"Expiration: {expiration} ({days_to_expiration} DTE)")
    print(f"\nBuilding Iron Condors from {len(put_spreads)} PCS x {len(call_spreads)} CCS...\n")

    top_condors = rank_iron_condors(
        put_spreads=put_spreads,
        call_spreads=call_spreads,
        underlying_price=underlying_price,
        days_to_expiration=days_to_expiration,
        top_n=5,
    )

    print(f"Found {len(top_condors)} valid Iron Condors. Top 5:\n")

    # Print header
    header = (
        f"{'#':>2} | {'ShortP':>6} {'LongP':>6} | {'ShortC':>6} {'LongC':>6} | "
        f"{'Credit':>6} | {'MaxLoss':>7} | {'ROC':>5} | {'POP':>4} | "
        f"{'Zone%':>5} | {'WScore':>5} | {'Liq':>4} | {'Tail':>4} | "
        f"{'SCORE':>5} | {'BE_Lo':>6} {'BE_Hi':>6} | {'Dist%L':>6} {'Dist%H':>6}"
    )
    print(header)
    print("-" * len(header))

    for i, c in enumerate(top_condors, 1):
        row = (
            f"{i:>2} | "
            f"{c.short_put_strike:>6.0f} {c.long_put_strike:>6.0f} | "
            f"{c.short_call_strike:>6.0f} {c.long_call_strike:>6.0f} | "
            f"${c.total_credit:>5.2f} | "
            f"${c.max_loss_dollars:>6.0f} | "
            f"{c.roc_raw:>5.1%} | "
            f"{c.pop:>4.0%} | "
            f"{c.width_zone_pct:>5.1%} | "
            f"{c.width_score:>5.2f} | "
            f"{c.liquidity_score:>4.2f} | "
            f"{c.tail_risk:>4.2f} | "
            f"{c.total_score:>5.2f} | "
            f"{c.breakeven_low:>6.2f} {c.breakeven_high:>6.2f} | "
            f"{c.distance_to_BE_low_pct:>6.1%} {c.distance_to_BE_high_pct:>6.1%}"
        )
        print(row)

    # Print payoff/ROI curve for the best condor
    if top_condors:
        best = top_condors[0]
        print(f"\n{'=' * 80}")
        print(f"PAYOFF/ROI CURVE - Best Iron Condor")
        print(f"{'=' * 80}")
        print(f"\nPut Spread: Short {best.short_put_strike:.0f} / Long {best.long_put_strike:.0f}")
        print(f"Call Spread: Short {best.short_call_strike:.0f} / Long {best.long_call_strike:.0f}")
        print(f"Total Credit: ${best.total_credit:.2f}/share (${best.max_profit_dollars:.0f}/contract)")
        print(f"Max Loss: ${best.max_loss_dollars:.0f}/contract")
        print(f"Breakevens: ${best.breakeven_low:.2f} - ${best.breakeven_high:.2f}")
        print()

        curve = payoff_roi_curve(best, move_low_pct=-0.05, move_high_pct=0.05, step_pct=0.01)

        print(f"{'Move':>6} | {'Price':>8} | {'Payoff':>10} | {'ROI':>8}")
        print("-" * 42)

        for point in curve:
            move_str = f"{point['move_pct']:+.0%}"
            payoff_str = f"${point['payoff']:+,.0f}"
            roi_str = f"{point['roi']:+.1%}" if point['roi'] != 0 else "0.0%"
            print(f"{move_str:>6} | ${point['price']:>7.2f} | {payoff_str:>10} | {roi_str:>8}")

    print()


if __name__ == "__main__":
    # Run tests first
    run_tests()

    # Then run example
    main()
