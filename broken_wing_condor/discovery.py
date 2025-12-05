"""
Strategy discovery module for broken-wing condors.

Discovers and constructs valid broken-wing condor structures
with near-free call spreads funded by deep put spread credit.
"""

import logging
from dataclasses import dataclass
from datetime import date
from itertools import product
from typing import Iterator, Optional

from broken_wing_condor.config import CondorConfig
from broken_wing_condor.models import BrokenWingCondor, OptionLeg

logger = logging.getLogger(__name__)


@dataclass
class StrikeSelection:
    """Selection criteria for strike prices."""
    underlying_price: float
    long_put_strike: float
    short_put_strike: float
    short_call_strike: float
    long_call_strike: float


def find_atm_strike(
    options: list[OptionLeg], underlying_price: float
) -> Optional[float]:
    """
    Find the at-the-money strike closest to underlying price.

    Args:
        options: List of option legs
        underlying_price: Current underlying price

    Returns:
        ATM strike or None if no options available
    """
    if not options:
        return None

    strikes = sorted(set(o.strike for o in options))
    if not strikes:
        return None

    return min(strikes, key=lambda s: abs(s - underlying_price))


def get_strike_ladder(
    options: list[OptionLeg], min_strike: float, max_strike: float
) -> list[float]:
    """
    Get sorted list of available strikes within range.

    Args:
        options: List of option legs
        min_strike: Minimum strike to include
        max_strike: Maximum strike to include

    Returns:
        Sorted list of strikes
    """
    strikes = sorted(set(
        o.strike for o in options
        if min_strike <= o.strike <= max_strike
    ))
    return strikes


def get_option_at_strike(
    options: list[OptionLeg], strike: float
) -> Optional[OptionLeg]:
    """
    Get option leg at specific strike.

    Args:
        options: List of option legs
        strike: Target strike price

    Returns:
        OptionLeg at strike or None
    """
    for opt in options:
        if abs(opt.strike - strike) < 0.01:
            return opt
    return None


def generate_put_spread_candidates(
    puts: list[OptionLeg],
    underlying_price: float,
    config: CondorConfig,
) -> Iterator[tuple[OptionLeg, OptionLeg]]:
    """
    Generate valid put spread candidates (short put, long put).

    Rules:
    - Short put at ATM or slightly ITM
    - Long put 5-15 points below short put
    - Put credit >= min_put_credit_pct of spread width

    Args:
        puts: List of put options
        underlying_price: Current underlying price
        config: Configuration

    Yields:
        Tuples of (short_put, long_put)
    """
    if not puts:
        return

    # Find ATM strike
    atm_strike = find_atm_strike(puts, underlying_price)
    if atm_strike is None:
        return

    # Short put candidates: ATM to slightly ITM (ATM to ATM+5%)
    max_short_put = underlying_price * 1.05
    short_put_strikes = get_strike_ladder(
        puts,
        underlying_price * 0.95,  # Allow slightly OTM too
        max_short_put,
    )

    for short_strike in short_put_strikes:
        short_put = get_option_at_strike(puts, short_strike)
        if short_put is None or short_put.mid <= 0:
            continue

        # Long put candidates: 5-15 points below short
        for width in range(config.put_spread_width_min, config.put_spread_width_max + 1):
            long_strike = short_strike - width
            long_put = get_option_at_strike(puts, long_strike)

            if long_put is None:
                continue

            # Calculate credit
            credit = short_put.mid - long_put.mid

            # Check credit capture ratio
            if width > 0:
                credit_pct = credit / width
                if credit_pct >= config.min_put_credit_pct:
                    yield (short_put, long_put)


def generate_call_spread_candidates(
    calls: list[OptionLeg],
    underlying_price: float,
    config: CondorConfig,
) -> Iterator[tuple[OptionLeg, OptionLeg]]:
    """
    Generate valid call spread candidates (short call, long call).

    Rules:
    - Short call at ATM + 10 to 20 points
    - Long call = short call + call_spread_width (fixed 10 points)
    - Call spread should be free or cost <= max_call_cost

    Args:
        calls: List of call options
        underlying_price: Current underlying price
        config: Configuration

    Yields:
        Tuples of (short_call, long_call)
    """
    if not calls:
        return

    # Short call candidates: ATM + 10 to ATM + 20
    min_short_call = underlying_price + 10
    max_short_call = underlying_price + 30  # Allow wider range for discovery

    short_call_strikes = get_strike_ladder(calls, min_short_call, max_short_call)

    for short_strike in short_call_strikes:
        short_call = get_option_at_strike(calls, short_strike)
        if short_call is None:
            continue

        # Long call at short + call_spread_width
        long_strike = short_strike + config.call_spread_width
        long_call = get_option_at_strike(calls, long_strike)

        if long_call is None:
            continue

        # Calculate debit (positive when short_call premium > long_call premium)
        # Debit is positive when we pay, negative when we receive credit
        debit = short_call.mid - long_call.mid

        # Check if call spread is cheap enough (accept credits and small debits)
        if debit <= config.max_call_cost:
            yield (short_call, long_call)


def construct_condor(
    long_put: OptionLeg,
    short_put: OptionLeg,
    short_call: OptionLeg,
    long_call: OptionLeg,
) -> BrokenWingCondor:
    """
    Construct a BrokenWingCondor from four legs.

    Args:
        long_put: Long put leg (lowest strike)
        short_put: Short put leg
        short_call: Short call leg
        long_call: Long call leg (highest strike)

    Returns:
        BrokenWingCondor object
    """
    # Calculate premiums
    put_spread_credit = short_put.mid - long_put.mid
    # Debit is positive when we pay for the call spread (short_call premium > long_call premium)
    call_spread_debit = short_call.mid - long_call.mid
    net_credit = put_spread_credit - call_spread_debit

    # Calculate widths
    put_spread_width = short_put.strike - long_put.strike
    call_spread_width = long_call.strike - short_call.strike

    # Risk/reward calculations
    # Max loss = put spread width - put credit (in dollars per contract)
    max_loss = (put_spread_width - put_spread_credit) * 100

    # Max profit if staying between strikes = net credit * 100
    max_profit_credit_only = net_credit * 100

    # Max profit if above long call = net credit + (call spread width - call debit)
    # We keep the net credit AND get the call spread profit (width minus what we paid)
    max_profit_with_calls = (net_credit + (call_spread_width - call_spread_debit)) * 100

    return BrokenWingCondor(
        long_put=long_put,
        short_put=short_put,
        short_call=short_call,
        long_call=long_call,
        put_spread_credit=put_spread_credit,
        call_spread_debit=call_spread_debit,
        net_credit=net_credit,
        put_spread_width=put_spread_width,
        call_spread_width=call_spread_width,
        max_loss=max_loss,
        max_profit_credit_only=max_profit_credit_only,
        max_profit_with_calls=max_profit_with_calls,
    )


def validate_condor(
    condor: BrokenWingCondor,
    config: CondorConfig,
    underlying_price: float = 0.0,
) -> bool:
    """
    Validate that a condor meets all strategy requirements.

    Args:
        condor: The condor to validate
        config: Configuration
        underlying_price: Current underlying price for safety margin check

    Returns:
        True if valid, False otherwise
    """
    # Check strike ordering
    if not (
        condor.long_put.strike
        < condor.short_put.strike
        < condor.short_call.strike
        < condor.long_call.strike
    ):
        return False

    # Net credit must be positive (put credit covers call debit)
    if condor.net_credit < 0:
        return False

    # Max loss must be within limits
    if condor.max_loss > config.max_loss_per_contract * 100:
        return False

    # Put credit capture must meet minimum
    if condor.credit_capture_pct < config.min_put_credit_pct:
        return False

    # Call spread must be cheap
    if condor.call_spread_debit > config.max_call_cost:
        return False

    # Safety margin: long put must be at least X% below spot price
    # This ensures adequate downside protection buffer
    if underlying_price > 0 and config.safety_margin_pct > 0:
        max_long_put_strike = underlying_price * (1 - config.safety_margin_pct)
        if condor.long_put.strike > max_long_put_strike:
            return False

    return True


def discover_condors(
    calls: list[OptionLeg],
    puts: list[OptionLeg],
    underlying_price: float,
    config: CondorConfig,
) -> list[BrokenWingCondor]:
    """
    Discover all valid broken-wing condor candidates.

    Args:
        calls: List of call options
        puts: List of put options
        underlying_price: Current underlying price
        config: Configuration

    Returns:
        List of valid BrokenWingCondor objects
    """
    condors = []

    # Generate all put spread candidates
    put_spreads = list(generate_put_spread_candidates(puts, underlying_price, config))
    logger.debug(f"Found {len(put_spreads)} put spread candidates")

    # Generate all call spread candidates
    call_spreads = list(generate_call_spread_candidates(calls, underlying_price, config))
    logger.debug(f"Found {len(call_spreads)} call spread candidates")

    # Combine into condors
    for (short_put, long_put), (short_call, long_call) in product(put_spreads, call_spreads):
        try:
            condor = construct_condor(long_put, short_put, short_call, long_call)

            if validate_condor(condor, config, underlying_price):
                condors.append(condor)

        except Exception as e:
            logger.warning(f"Failed to construct condor: {e}")
            continue

    logger.info(f"Discovered {len(condors)} valid condor candidates")
    return condors


def discover_condors_for_expiration(
    calls: list[OptionLeg],
    puts: list[OptionLeg],
    underlying_price: float,
    expiration: date,
    config: CondorConfig,
) -> list[BrokenWingCondor]:
    """
    Discover condors for a specific expiration date.

    This is a convenience wrapper that filters options to the specific expiration.

    Args:
        calls: List of call options (may contain multiple expirations)
        puts: List of put options (may contain multiple expirations)
        underlying_price: Current underlying price
        expiration: Target expiration date
        config: Configuration

    Returns:
        List of valid BrokenWingCondor objects for this expiration
    """
    # Filter to expiration
    exp_calls = [c for c in calls if c.expiration == expiration]
    exp_puts = [p for p in puts if p.expiration == expiration]

    logger.debug(
        f"Expiration {expiration}: {len(exp_calls)} calls, {len(exp_puts)} puts"
    )

    return discover_condors(exp_calls, exp_puts, underlying_price, config)
