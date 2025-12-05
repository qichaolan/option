"""
Scoring framework for broken-wing condors.

Multi-factor quantitative scoring model that evaluates trades
based on risk, credit quality, skew, and expected value.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from broken_wing_condor.config import CondorConfig, ScoringWeights
from broken_wing_condor.models import BrokenWingCondor, CondorScore, PayoffScenario

logger = logging.getLogger(__name__)


def calculate_risk_score(condor: BrokenWingCondor, config: CondorConfig) -> float:
    """
    Calculate risk score based on max loss relative to spread width.

    Lower max loss relative to spread width = higher score.

    Args:
        condor: The condor to score
        config: Configuration

    Returns:
        Normalized score 0-1 (higher is better)
    """
    # Max loss as percentage of put spread width (in dollars)
    put_width_dollars = condor.put_spread_width * 100

    if put_width_dollars <= 0:
        return 0.0

    # Risk ratio: max_loss / put_spread_width
    # Lower ratio means less risk relative to spread
    risk_ratio = condor.max_loss / put_width_dollars

    # Normalize: risk_ratio of 0.5 (50% of spread) = 0.5 score
    # risk_ratio of 0.2 (20% of spread) = 0.8 score
    # risk_ratio of 0.8 (80% of spread) = 0.2 score
    score = max(0.0, min(1.0, 1.0 - risk_ratio))

    return score


def calculate_credit_score(condor: BrokenWingCondor, config: CondorConfig) -> float:
    """
    Calculate credit score based on put spread credit capture.

    Higher credit capture percentage = higher score.

    Args:
        condor: The condor to score
        config: Configuration

    Returns:
        Normalized score 0-1 (higher is better)
    """
    # Credit capture: how much of the put spread width we collect as credit
    credit_pct = condor.credit_capture_pct

    # Scale: 0% capture = 0 score, 100% capture = 1 score
    # Typical range is 30-70% capture
    score = max(0.0, min(1.0, credit_pct))

    return score


def calculate_skew_score(condor: BrokenWingCondor) -> float:
    """
    Calculate skew score based on put vs call spread width asymmetry.

    Wider call spread relative to put spread = higher upside potential.

    Args:
        condor: The condor to score

    Returns:
        Normalized score 0-1 (higher is better)
    """
    if condor.put_spread_width <= 0:
        return 0.0

    # Skew ratio: call_width / put_width
    # Ratio of 1.0 = symmetrical iron condor
    # Ratio > 1.0 = more upside potential (broken-wing)
    # Ratio of 2.0 = call spread is 2x wider than put spread
    skew_ratio = condor.call_spread_width / condor.put_spread_width

    # Normalize: ratio of 0.5 = 0 score, ratio of 2.0 = 1 score
    # Linear interpolation in this range
    score = max(0.0, min(1.0, (skew_ratio - 0.5) / 1.5))

    return score


def calculate_call_score(condor: BrokenWingCondor, config: CondorConfig) -> float:
    """
    Calculate call score based on how free the call spread is.

    Lower call spread debit = higher score (closer to free).

    Args:
        condor: The condor to score
        config: Configuration

    Returns:
        Normalized score 0-1 (higher is better)
    """
    # call_spread_debit is in dollars per share
    debit = condor.call_spread_debit
    max_cost = config.max_call_cost

    if max_cost <= 0:
        return 1.0 if debit <= 0 else 0.0

    # Score: 0 debit = 1.0, max_cost debit = 0.0
    # Negative debit (credit) also gets 1.0
    if debit <= 0:
        return 1.0

    score = max(0.0, min(1.0, 1.0 - (debit / max_cost)))

    return score


def calculate_rrr_score(condor: BrokenWingCondor) -> float:
    """
    Calculate risk/reward ratio score.

    Higher max profit relative to max loss = higher score.

    Args:
        condor: The condor to score

    Returns:
        Normalized score 0-1 (higher is better)
    """
    if condor.max_loss <= 0:
        return 1.0

    # Use the higher of the two max profit scenarios
    max_profit = max(condor.max_profit_credit_only, condor.max_profit_with_calls)

    # Risk/Reward ratio
    rrr = max_profit / condor.max_loss

    # Normalize: RRR of 0 = 0 score, RRR of 2.0 = 1 score
    # Most condors have RRR < 1, so this scale rewards higher RRR
    score = max(0.0, min(1.0, rrr / 2.0))

    return score


def estimate_probability_of_profit(
    condor: BrokenWingCondor,
    underlying_price: float,
    days_to_expiration: int,
    annual_volatility: float = 0.20,
) -> float:
    """
    Estimate probability of profit using simplified model.

    Uses a normal distribution approximation based on expected move.

    Args:
        condor: The condor to score
        underlying_price: Current underlying price
        days_to_expiration: Days until expiration
        annual_volatility: Annualized volatility (default 20%)

    Returns:
        Estimated probability of profit (0-1)
    """
    if days_to_expiration <= 0 or underlying_price <= 0:
        return 0.5

    # Calculate expected move using vol
    time_factor = math.sqrt(days_to_expiration / 365)
    expected_move_pct = annual_volatility * time_factor
    expected_move = underlying_price * expected_move_pct

    # Standard deviation of price at expiration
    std_dev = expected_move

    if std_dev <= 0:
        return 0.5

    # For a broken-wing condor, profit zone is between short strikes
    # Simplified: probability of staying above short put strike
    short_put_strike = condor.short_put.strike

    # Z-score for short put strike
    z_score = (short_put_strike - underlying_price) / std_dev

    # Probability of price > short_put_strike (profit on put spread)
    # Using simplified normal CDF approximation
    prob_above_put = _normal_cdf(-z_score)

    return max(0.0, min(1.0, prob_above_put))


def _normal_cdf(x: float) -> float:
    """
    Approximate standard normal CDF using error function.

    Args:
        x: Z-score

    Returns:
        Cumulative probability
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def calculate_ev_score(
    condor: BrokenWingCondor,
    underlying_price: float,
    days_to_expiration: int,
    annual_volatility: float = 0.20,
) -> float:
    """
    Calculate expected value score.

    EV = (prob_profit * max_profit) - (prob_loss * max_loss)

    Args:
        condor: The condor to score
        underlying_price: Current underlying price
        days_to_expiration: Days until expiration
        annual_volatility: Annualized volatility

    Returns:
        Normalized score 0-1 (higher is better)
    """
    prob_profit = estimate_probability_of_profit(
        condor, underlying_price, days_to_expiration, annual_volatility
    )
    prob_loss = 1 - prob_profit

    max_profit = max(condor.max_profit_credit_only, condor.max_profit_with_calls)
    max_loss = condor.max_loss

    # Expected value
    ev = (prob_profit * max_profit) - (prob_loss * max_loss)

    # Normalize: EV relative to max_loss
    # EV of 0 = 0.5 score
    # Positive EV = higher score, Negative EV = lower score
    if max_loss <= 0:
        return 0.5

    ev_ratio = ev / max_loss

    # Scale: ev_ratio of -1 = 0 score, ev_ratio of +1 = 1 score
    score = max(0.0, min(1.0, (ev_ratio + 1) / 2))

    return score


def calculate_pop_score(
    condor: BrokenWingCondor,
    underlying_price: float,
    days_to_expiration: int,
    annual_volatility: float = 0.20,
) -> float:
    """
    Calculate probability of profit score.

    Args:
        condor: The condor to score
        underlying_price: Current underlying price
        days_to_expiration: Days until expiration
        annual_volatility: Annualized volatility

    Returns:
        Normalized score 0-1 (higher is better)
    """
    pop = estimate_probability_of_profit(
        condor, underlying_price, days_to_expiration, annual_volatility
    )

    return pop


def calculate_payoff_scenarios(
    condor: BrokenWingCondor,
    underlying_price: float,
) -> list[PayoffScenario]:
    """
    Calculate P/L at various price scenarios.

    Args:
        condor: The condor to evaluate
        underlying_price: Current underlying price

    Returns:
        List of PayoffScenario objects
    """
    scenarios = []

    # Price moves from -30% to +50%
    price_moves = [-0.30, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    for move in price_moves:
        price_at_exp = underlying_price * (1 + move)
        pl = calculate_pl_at_expiration(condor, price_at_exp)

        scenarios.append(PayoffScenario(
            price_at_expiration=price_at_exp,
            price_move_pct=move * 100,
            profit_loss=pl,
        ))

    return scenarios


def calculate_pl_at_expiration(condor: BrokenWingCondor, price: float) -> float:
    """
    Calculate P/L at expiration for a given underlying price.

    Args:
        condor: The condor position
        price: Underlying price at expiration

    Returns:
        Profit/Loss in dollars (per contract = 100 shares)
    """
    # Initial credit received (per share)
    net_credit = condor.net_credit

    # Put spread P/L
    put_pl = calculate_put_spread_pl(
        condor.long_put.strike,
        condor.short_put.strike,
        price,
    )

    # Call spread P/L (we're long the call spread)
    call_pl = calculate_call_spread_pl(
        condor.short_call.strike,
        condor.long_call.strike,
        price,
    )

    # Total P/L (in dollars per share, then multiply by 100)
    total_pl = (net_credit + put_pl + call_pl) * 100

    return total_pl


def calculate_put_spread_pl(long_strike: float, short_strike: float, price: float) -> float:
    """
    Calculate put spread P/L at expiration (short put spread).

    We're short the higher strike put, long the lower strike put.

    Args:
        long_strike: Long put strike (lower)
        short_strike: Short put strike (higher)
        price: Underlying price at expiration

    Returns:
        P/L per share (excluding premium)
    """
    # Short put P/L (positive when OTM)
    short_put_value = max(0, short_strike - price)

    # Long put P/L (positive when ITM)
    long_put_value = max(0, long_strike - price)

    # Net: we're short the spread, so we want both to expire worthless
    # Loss occurs when price < short_strike
    return long_put_value - short_put_value


def calculate_call_spread_pl(short_strike: float, long_strike: float, price: float) -> float:
    """
    Calculate call spread P/L at expiration (bull call spread for upside participation).

    For a BWC with upside convexity:
    - We're LONG the lower strike call (short_strike parameter) - gives us upside
    - We're SHORT the higher strike call (long_strike parameter) - caps upside, reduces cost

    Args:
        short_strike: Lower strike call (we're LONG this)
        long_strike: Higher strike call (we're SHORT this)
        price: Underlying price at expiration

    Returns:
        P/L per share (excluding premium)
    """
    # Long call value (we own the lower strike call)
    long_call_value = max(0, price - short_strike)

    # Short call liability (we owe on the higher strike call)
    short_call_liability = max(0, price - long_strike)

    # Net: profit = what we gain - what we owe
    # Max profit = spread width when price >= long_strike
    return long_call_value - short_call_liability


def score_condor(
    condor: BrokenWingCondor,
    underlying_price: float,
    days_to_expiration: int,
    config: CondorConfig,
    weights: Optional[ScoringWeights] = None,
    annual_volatility: float = 0.20,
) -> CondorScore:
    """
    Calculate all scores for a condor and combine into final score.

    Args:
        condor: The condor to score
        underlying_price: Current underlying price
        days_to_expiration: Days until expiration
        config: Configuration
        weights: Scoring weights (uses defaults if None)
        annual_volatility: Annualized volatility for probability calcs

    Returns:
        CondorScore with all component scores and final weighted score
    """
    if weights is None:
        weights = ScoringWeights()

    # Calculate individual scores
    risk_score = calculate_risk_score(condor, config)
    credit_score = calculate_credit_score(condor, config)
    skew_score = calculate_skew_score(condor)
    call_score = calculate_call_score(condor, config)
    rrr_score = calculate_rrr_score(condor)

    pop = estimate_probability_of_profit(
        condor, underlying_price, days_to_expiration, annual_volatility
    )
    pop_score = pop  # PoP is already 0-1

    # Calculate expected value
    max_profit = max(condor.max_profit_credit_only, condor.max_profit_with_calls)
    ev = (pop * max_profit) - ((1 - pop) * condor.max_loss)

    ev_score = calculate_ev_score(
        condor, underlying_price, days_to_expiration, annual_volatility
    )

    # Calculate weighted final score
    final_score = (
        weights.risk_weight * risk_score +
        weights.credit_weight * credit_score +
        weights.skew_weight * skew_score +
        weights.call_weight * call_score +
        weights.rrr_weight * rrr_score +
        weights.ev_weight * ev_score +
        weights.pop_weight * pop_score
    )

    # If POP < 25%, trade is too risky - set final_score to 0
    if pop < 0.25:
        final_score = 0.0

    # Calculate risk/reward ratio
    reward_to_risk = max_profit / condor.max_loss if condor.max_loss > 0 else 0.0

    # Calculate IV skew (simplified - use difference in IVs if available)
    iv_skew = 0.0
    if condor.short_put.iv and condor.short_call.iv:
        iv_skew = condor.short_put.iv - condor.short_call.iv

    return CondorScore(
        risk_score=risk_score,
        credit_score=credit_score,
        skew_score=skew_score,
        call_score=call_score,
        rrr_score=rrr_score,
        ev_score=ev_score,
        pop_score=pop_score,
        final_score=final_score,
        max_risk=condor.max_loss,
        reward_to_risk=reward_to_risk,
        put_credit_pct=condor.credit_capture_pct,
        call_spread_cost=condor.call_spread_debit,
        iv_skew=iv_skew,
        pop=pop,
        expected_value=ev,
    )
