"""
Rule evaluation engine for backtesting.

This module evaluates trading rules against indicator data to generate
trading signals (BUY, SELL, HOLD).

Supports lag indicators with pattern: indicator_lag_N
Example: ema_21_lag_1 means ema_21 shifted by 1 period (previous day's value)
"""

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine.constants import (
    Action,
    CombineMode,
    Signal,
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_SELL_THRESHOLD,
)
from backtest.engine.exceptions import MissingIndicatorError
from backtest.engine.strategy_loader import Rule, Strategy, get_all_required_indicators


# Backwards compatibility - expose thresholds at module level
BUY_THRESHOLD = DEFAULT_BUY_THRESHOLD
SELL_THRESHOLD = DEFAULT_SELL_THRESHOLD

# Pattern to match lag indicators: indicator_lag_N
LAG_PATTERN = re.compile(r"^(.+)_lag_(\d+)$", re.IGNORECASE)


def parse_lag_indicator(indicator_name: str) -> Tuple[str, Optional[int]]:
    """
    Parse an indicator name to extract base indicator and lag period.

    Args:
        indicator_name: Indicator name, possibly with _lag_N suffix.

    Returns:
        Tuple of (base_indicator, lag_periods) where lag_periods is None
        if no lag suffix found.

    Examples:
        "ema_21_lag_1" -> ("ema_21", 1)
        "rsi_14" -> ("rsi_14", None)
        "close_lag_5" -> ("close", 5)
    """
    match = LAG_PATTERN.match(indicator_name)
    if match:
        base_indicator = match.group(1)
        lag_periods = int(match.group(2))
        return base_indicator, lag_periods
    return indicator_name, None


def get_base_indicators(indicators: List[str]) -> List[str]:
    """
    Get base indicator names from a list that may include lag indicators.

    Args:
        indicators: List of indicator names.

    Returns:
        List of unique base indicator names (without _lag_N suffix).
    """
    base_indicators = set()
    for ind in indicators:
        base, _ = parse_lag_indicator(ind)
        base_indicators.add(base)
    return list(base_indicators)


def validate_indicators_present(
    df: pd.DataFrame, strategies: List[Strategy]
) -> None:
    """
    Validate that all required indicators are present in the DataFrame.

    Supports lag indicators: indicator_lag_N is valid if indicator exists.
    Also warns if any indicator column is entirely NaN.

    Args:
        df: DataFrame with indicator columns.
        strategies: List of strategies to check.

    Raises:
        MissingIndicatorError: If any required indicator is missing.
    """
    import warnings

    required = get_all_required_indicators(strategies)
    # Get base indicators (strip _lag_N suffix)
    base_required = get_base_indicators(required)

    # Normalize column names for comparison
    df_columns_lower = {col.lower(): col for col in df.columns}

    missing = []
    all_nan_indicators = []

    for ind in base_required:
        ind_lower = ind.lower()
        if ind_lower not in df_columns_lower:
            missing.append(ind)
        else:
            # Check if the indicator column is entirely NaN
            actual_col = df_columns_lower[ind_lower]
            if df[actual_col].isna().all():
                all_nan_indicators.append(ind)

    if missing:
        raise MissingIndicatorError(missing)

    # Warn about all-NaN indicators (rules using them will never trigger)
    if all_nan_indicators:
        warnings.warn(
            f"The following indicators contain only NaN values and rules using them "
            f"will never trigger: {', '.join(all_nan_indicators)}",
            UserWarning,
        )


def get_column_case_insensitive(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Get a column from DataFrame case-insensitively.

    Supports lag indicators: indicator_lag_N returns indicator shifted by N periods.
    Example: ema_21_lag_1 returns ema_21.shift(1)

    Args:
        df: DataFrame to search.
        column_name: Column name (case-insensitive), may include _lag_N suffix.

    Returns:
        Series for the matched column, shifted if lag indicator.

    Raises:
        KeyError: If column not found.
    """
    # Check for lag indicator pattern
    base_indicator, lag_periods = parse_lag_indicator(column_name)

    # Find the base column case-insensitively
    col_lower = base_indicator.lower()
    matched_col = None
    for col in df.columns:
        if col.lower() == col_lower:
            matched_col = col
            break

    if matched_col is None:
        raise KeyError(f"Column '{base_indicator}' not found")

    # Get the column values (copy to avoid modifying original)
    values = df[matched_col].copy()

    # Apply lag if specified
    if lag_periods is not None:
        values = values.shift(lag_periods)

    return values


def evaluate_rule(
    df: pd.DataFrame, rule: Rule
) -> Tuple[pd.Series, pd.Series]:
    """
    Evaluate a single rule against data.

    Handles NaN values by treating them as non-triggering (False).

    Args:
        df: DataFrame with indicator columns.
        rule: Rule to evaluate.

    Returns:
        Tuple of (triggered_mask, scores):
            - triggered_mask: Boolean Series indicating where rule triggers
            - scores: Series of scores (+strength for buy, -strength for sell)
    """
    indicator_values = get_column_case_insensitive(df, rule.indicator)

    # Get comparison value (either static or from another indicator)
    if rule.value is not None:
        comparison_values = rule.value
        # Create valid mask (only indicator NaN matters for static comparison)
        valid_mask = ~indicator_values.isna()
    else:
        comparison_values = get_column_case_insensitive(df, rule.value_indicator)
        # Both indicators must be valid
        valid_mask = ~indicator_values.isna() & ~comparison_values.isna()

    # Evaluate comparison (NaN comparisons yield False which is correct)
    if rule.operator == "<":
        triggered = indicator_values < comparison_values
    elif rule.operator == "<=":
        triggered = indicator_values <= comparison_values
    elif rule.operator == ">":
        triggered = indicator_values > comparison_values
    elif rule.operator == ">=":
        triggered = indicator_values >= comparison_values
    elif rule.operator == "==":
        triggered = indicator_values == comparison_values
    elif rule.operator == "!=":
        triggered = indicator_values != comparison_values
    else:
        triggered = pd.Series(False, index=df.index)

    # Explicitly mask NaN rows to prevent any edge case issues
    triggered = triggered & valid_mask

    # Calculate score based on action
    base_score = rule.strength if rule.action == Action.BUY else -rule.strength
    scores = pd.Series(0.0, index=df.index)
    scores[triggered] = base_score

    return triggered, scores


def _vectorized_nonzero_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calculate row-wise mean of non-zero values efficiently.

    Uses vectorized operations instead of row-wise apply for performance.

    Args:
        df: DataFrame to process.

    Returns:
        Series with mean of non-zero values per row, 0 where all values are zero.
    """
    # Replace zeros with NaN, then use built-in mean which skips NaN
    masked = df.replace(0, np.nan)
    return masked.mean(axis=1).fillna(0)


def evaluate_strategy(
    df: pd.DataFrame, strategy: Strategy
) -> pd.Series:
    """
    Evaluate a single strategy against data.

    Args:
        df: DataFrame with indicator columns.
        strategy: Strategy to evaluate.

    Returns:
        Series of strategy scores for each row.
    """
    if not strategy.rules:
        return pd.Series(0.0, index=df.index)

    # Separate buy and sell rules
    buy_rules = [r for r in strategy.rules if r.action == Action.BUY]
    sell_rules = [r for r in strategy.rules if r.action == Action.SELL]

    # Evaluate all rules
    buy_triggered_list = []
    buy_scores_list = []
    sell_triggered_list = []
    sell_scores_list = []

    for rule in buy_rules:
        triggered, scores = evaluate_rule(df, rule)
        buy_triggered_list.append(triggered)
        buy_scores_list.append(scores)

    for rule in sell_rules:
        triggered, scores = evaluate_rule(df, rule)
        sell_triggered_list.append(triggered)
        sell_scores_list.append(scores)

    # Combine based on strategy mode
    if strategy.combine == CombineMode.ALL:
        # All rules of the same type must trigger
        if buy_triggered_list:
            buy_all_triggered = pd.concat(buy_triggered_list, axis=1).all(axis=1)
            buy_score = pd.concat(buy_scores_list, axis=1).mean(axis=1)
            buy_score = buy_score.where(buy_all_triggered, 0.0)
        else:
            buy_score = pd.Series(0.0, index=df.index)

        if sell_triggered_list:
            sell_all_triggered = pd.concat(sell_triggered_list, axis=1).all(axis=1)
            sell_score = pd.concat(sell_scores_list, axis=1).mean(axis=1)
            sell_score = sell_score.where(sell_all_triggered, 0.0)
        else:
            sell_score = pd.Series(0.0, index=df.index)

    else:  # "any" - use vectorized operation for performance
        if buy_scores_list:
            buy_df = pd.concat(buy_scores_list, axis=1)
            buy_score = _vectorized_nonzero_mean(buy_df)
        else:
            buy_score = pd.Series(0.0, index=df.index)

        if sell_scores_list:
            sell_df = pd.concat(sell_scores_list, axis=1)
            sell_score = _vectorized_nonzero_mean(sell_df)
        else:
            sell_score = pd.Series(0.0, index=df.index)

    # Net score for the strategy (buy score is positive, sell score is negative)
    net_score = buy_score + sell_score

    return net_score


def evaluate_all_strategies(
    df: pd.DataFrame, strategies: List[Strategy]
) -> pd.Series:
    """
    Evaluate all strategies and compute weighted aggregate score.

    Args:
        df: DataFrame with indicator columns.
        strategies: List of Strategy objects with normalized weights.

    Returns:
        Series of aggregate scores for each row.
    """
    if not strategies:
        return pd.Series(0.0, index=df.index)

    aggregate_score = pd.Series(0.0, index=df.index)

    for strategy in strategies:
        strategy_score = evaluate_strategy(df, strategy)
        aggregate_score += strategy_score * strategy.weight

    return aggregate_score


def generate_signals(
    scores: pd.Series,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
) -> pd.Series:
    """
    Generate trading signals from scores.

    Args:
        scores: Series of aggregate scores.
        buy_threshold: Score threshold for BUY signal (default 0.3).
        sell_threshold: Score threshold for SELL signal (default -0.3).

    Returns:
        Series of signals ("BUY", "SELL", or "HOLD").
    """
    signals = pd.Series(Signal.HOLD, index=scores.index)
    signals[scores >= buy_threshold] = Signal.BUY
    signals[scores <= sell_threshold] = Signal.SELL
    return signals


def run_rule_engine(
    df: pd.DataFrame,
    strategies: List[Strategy],
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    """
    Run the full rule engine pipeline.

    Args:
        df: DataFrame with indicator columns.
        strategies: List of Strategy objects.
        buy_threshold: Score threshold for BUY signal (default 0.3).
        sell_threshold: Score threshold for SELL signal (default -0.3).

    Returns:
        Tuple of (signals, scores):
            - signals: Series of "BUY", "SELL", or "HOLD"
            - scores: Series of aggregate scores
    """
    # Validate indicators are present
    validate_indicators_present(df, strategies)

    # Evaluate all strategies
    scores = evaluate_all_strategies(df, strategies)

    # Generate signals from scores
    signals = generate_signals(scores, buy_threshold, sell_threshold)

    return signals, scores
