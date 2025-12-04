"""
AI Score service layer.

Orchestrates the daily scorer and GCS cache to provide AI scores for symbols.
"""

import logging
import math
import os
from datetime import datetime, date
from typing import Optional

from backtest.daily_scorer.scorer import DailyScorer
from backtest.daily_scorer.exceptions import DailyScorerError
from app.services import gcs_cache

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_STRATEGY = os.environ.get("DEFAULT_STRATEGY", "backtest/strategies/s.yaml")
SCORE_LOOKBACK_DAYS = int(os.environ.get("SCORE_LOOKBACK_DAYS", "365"))
SCORE_NORMALIZATION = os.environ.get("SCORE_NORMALIZATION", "zscore")

# Maximum days to consider a cached score as "fresh" on weekends/holidays
MAX_CACHE_STALENESS_DAYS = 3


def _validate_score(score_0_1: float) -> float:
    """
    Validate and clamp score to valid range.

    Args:
        score_0_1: Score value to validate

    Returns:
        Clamped score between 0 and 1

    Raises:
        ValueError: If score is NaN or infinite
    """
    if math.isnan(score_0_1) or math.isinf(score_0_1):
        raise ValueError(f"Invalid score value: {score_0_1}")

    # Clamp to [0, 1] range
    return max(0.0, min(1.0, score_0_1))


def _get_ai_rating(score_0_1: float) -> str:
    """
    Map a 0-1 score to an AI rating string.

    Args:
        score_0_1: Normalized score between 0 and 1

    Returns:
        AI rating string
    """
    # Ensure score is in valid range
    score_0_1 = _validate_score(score_0_1)

    if score_0_1 > 0.9:
        return "Strong Buy"
    elif score_0_1 > 0.7:
        return "Buy"
    elif score_0_1 >= 0.3:
        return "Hold"
    elif score_0_1 >= 0.1:
        return "Sell"
    else:
        return "Must Sell"


def _is_market_closed_today() -> bool:
    """
    Check if the market is closed today.

    Currently checks weekends only. For production, consider using
    a market calendar library like `exchange_calendars` or `pandas_market_calendars`.
    """
    today = date.today()
    return today.weekday() >= 5  # Saturday = 5, Sunday = 6


def _build_score_response(
    symbol: str,
    score_date: date,
    signal_raw: float,
    signal_0_1: float,
) -> dict:
    """
    Build a standardized score response dict.

    Args:
        symbol: Stock ticker symbol
        score_date: Date of the score
        signal_raw: Raw signal value
        signal_0_1: Normalized 0-1 signal value

    Returns:
        Standardized response dict
    """
    validated_score = _validate_score(signal_0_1)

    return {
        "symbol": symbol.upper(),
        "date": score_date.isoformat(),
        "score_raw": float(signal_raw),
        "score_0_1": validated_score,
        "ai_rating": _get_ai_rating(validated_score),
    }


def _extract_date(cached_date) -> date:
    """Extract date object from cached date value."""
    if isinstance(cached_date, datetime):
        return cached_date.date()
    return cached_date


def get_ai_score(symbol: str, force_refresh: bool = False) -> dict:
    """
    Get AI score for a symbol.

    This is the core function used by the API and potential future jobs.

    Args:
        symbol: Stock ticker symbol (e.g., 'SPY')
        force_refresh: If True, always refresh from source data

    Returns:
        Dict with keys:
            - symbol: str
            - date: str (YYYY-MM-DD)
            - score_raw: float
            - score_0_1: float
            - ai_rating: str

    Raises:
        ValueError: If symbol is invalid or no data available
    """
    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError("Symbol is required")

    # Try to get from cache first (unless force refresh)
    if not force_refresh:
        cached = gcs_cache.get_latest_score(symbol)
        if cached:
            cached_date = _extract_date(cached["date"])
            today = date.today()

            # If cached score is from today, use it
            if cached_date == today:
                logger.debug(f"Using today's cached score for {symbol}")
                return _build_score_response(
                    symbol=symbol,
                    score_date=cached_date,
                    signal_raw=cached["signal_raw"],
                    signal_0_1=cached["signal_0_1"],
                )

            # If market is closed and we have a recent score, use it
            if _is_market_closed_today():
                days_diff = (today - cached_date).days
                if days_diff <= MAX_CACHE_STALENESS_DAYS:
                    logger.debug(f"Using cached score from {cached_date} for {symbol} (market closed)")
                    return _build_score_response(
                        symbol=symbol,
                        score_date=cached_date,
                        signal_raw=cached["signal_raw"],
                        signal_0_1=cached["signal_0_1"],
                    )

    # Need to refresh - compute new score
    logger.info(f"Refreshing score for {symbol}")
    try:
        scorer = DailyScorer(
            symbol=symbol,
            strategy_files=DEFAULT_STRATEGY,
            normalization=SCORE_NORMALIZATION,
            lookback_days=SCORE_LOOKBACK_DAYS,
        )

        result = scorer.refresh()

        if result is None:
            # Check if we have any cached score to fall back to
            logger.warning(f"Refresh returned None for {symbol}, checking cache fallback")
            cached = gcs_cache.get_latest_score(symbol)
            if cached:
                cached_date = _extract_date(cached["date"])
                return _build_score_response(
                    symbol=symbol,
                    score_date=cached_date,
                    signal_raw=cached["signal_raw"],
                    signal_0_1=cached["signal_0_1"],
                )

            raise ValueError(f"No score data available for {symbol}")

        # Cache the new score to GCS
        try:
            gcs_cache.add_score(
                symbol=symbol,
                date=result.date,
                signal_raw=result.signal_raw,
                signal_0_1=result.signal_0_1,
            )
        except Exception as e:
            # Log but don't fail if caching fails
            logger.warning(f"Failed to cache score for {symbol}: {e}")

        score_date = _extract_date(result.date)

        return _build_score_response(
            symbol=symbol,
            score_date=score_date,
            signal_raw=result.signal_raw,
            signal_0_1=result.signal_0_1,
        )

    except DailyScorerError as e:
        logger.error(f"DailyScorerError for {symbol}: {e}")
        raise ValueError(f"Error computing score for {symbol}: {e}")


def get_ai_scores_batch(symbols: list[str], force_refresh: bool = False) -> list[dict]:
    """
    Get AI scores for multiple symbols.

    Args:
        symbols: List of stock ticker symbols
        force_refresh: If True, always refresh from source data

    Returns:
        List of score dicts (same format as get_ai_score)
        Errors are returned as dicts with 'error' key
    """
    results = []

    for symbol in symbols:
        try:
            score = get_ai_score(symbol, force_refresh=force_refresh)
            results.append(score)
        except ValueError as e:
            results.append({
                "symbol": symbol.upper(),
                "error": str(e),
            })

    return results
