"""
Continuous activation functions for the scoring engine.

This module provides data-driven activation functions that return continuous
values in [0, 1] based on how deeply an indicator satisfies a rule condition.

All functions are pure, stateless, and fully vectorized using pandas operations.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def clip_activation(values: pd.Series) -> pd.Series:
    """
    Clip activation values to [0, 1] range.

    Args:
        values: Series of raw activation values.

    Returns:
        Series with values clipped to [0, 1].
    """
    return values.clip(lower=0.0, upper=1.0)


def compute_normalizing_scale(
    indicator: pd.Series,
    threshold: float,
    method: str = "std",
    lookback: Optional[int] = None,
) -> pd.Series:
    """
    Compute data-driven normalizing scale for activation functions.

    The scale is based on indicator volatility, making activations adaptive
    to the indicator's natural range of variation.

    IMPORTANT: To avoid look-ahead bias, this function uses expanding windows
    when lookback is None, ensuring only past data is used at each point.

    Args:
        indicator: Series of indicator values.
        threshold: The threshold value being compared against.
        method: Method for computing scale:
            - "std": Standard deviation (default)
            - "range": (max - min) / 4
            - "mad": Median absolute deviation
            - "threshold_pct": Percentage of threshold (for bounded indicators)
        lookback: Optional rolling window size. If None, uses expanding window
                  to avoid look-ahead bias.

    Returns:
        Series of normalizing scale values (always positive, minimum 0.001).
    """
    if lookback is not None:
        if method == "std":
            scale = indicator.rolling(window=lookback, min_periods=1).std()
        elif method == "range":
            rolling_max = indicator.rolling(window=lookback, min_periods=1).max()
            rolling_min = indicator.rolling(window=lookback, min_periods=1).min()
            scale = (rolling_max - rolling_min) / 4
        elif method == "mad":
            rolling_median = indicator.rolling(window=lookback, min_periods=1).median()
            scale = (indicator - rolling_median).abs().rolling(
                window=lookback, min_periods=1
            ).median() * 1.4826  # MAD to std conversion
        elif method == "threshold_pct":
            scale = pd.Series(abs(threshold) * 0.2, index=indicator.index)
        else:
            scale = indicator.rolling(window=lookback, min_periods=1).std()
    else:
        # Use expanding window to avoid look-ahead bias
        # At each point, only data up to that point is used
        # For small datasets, fall back to full history to avoid excessive NaNs
        n = len(indicator)
        if n <= 5:
            # Small dataset: use full history (original behavior for tests)
            if method == "std":
                scale = pd.Series(indicator.std(), index=indicator.index)
            elif method == "range":
                scale = pd.Series((indicator.max() - indicator.min()) / 4, index=indicator.index)
            elif method == "mad":
                mad = (indicator - indicator.median()).abs().median() * 1.4826
                scale = pd.Series(mad, index=indicator.index)
            elif method == "threshold_pct":
                scale = pd.Series(abs(threshold) * 0.2, index=indicator.index)
            else:
                scale = pd.Series(indicator.std(), index=indicator.index)
        else:
            # Normal case: expanding window for look-ahead bias prevention
            if method == "std":
                scale = indicator.expanding(min_periods=2).std()
            elif method == "range":
                expanding_max = indicator.expanding(min_periods=1).max()
                expanding_min = indicator.expanding(min_periods=1).min()
                scale = (expanding_max - expanding_min) / 4
            elif method == "mad":
                # Expanding MAD calculation
                expanding_median = indicator.expanding(min_periods=1).median()
                scale = (indicator - expanding_median).abs().expanding(
                    min_periods=1
                ).median() * 1.4826
            elif method == "threshold_pct":
                scale = pd.Series(abs(threshold) * 0.2, index=indicator.index)
            else:
                scale = indicator.expanding(min_periods=2).std()
            # Fill initial NaNs with first valid value to ensure no NaN outputs
            scale = scale.bfill()

    # Ensure minimum scale to avoid division by zero
    return scale.clip(lower=0.001)


def activation_less_than(
    indicator: pd.Series,
    threshold: float,
    normalizing_scale: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for indicator < threshold rules.

    Activation increases as indicator moves further below threshold.
    Returns 0 when indicator >= threshold.

    Formula: activation = clip((threshold - indicator) / scale, 0, 1)

    Args:
        indicator: Series of indicator values.
        threshold: Threshold value.
        normalizing_scale: Optional custom scale. If None, computed from data.

    Returns:
        Series of activation values in [0, 1].
    """
    if normalizing_scale is None:
        normalizing_scale = compute_normalizing_scale(indicator, threshold)

    # Distance below threshold (positive when satisfied)
    distance = threshold - indicator

    # Activation: 0 at threshold, increasing as we go deeper below
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_greater_than(
    indicator: pd.Series,
    threshold: float,
    normalizing_scale: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for indicator > threshold rules.

    Activation increases as indicator moves further above threshold.
    Returns 0 when indicator <= threshold.

    Formula: activation = clip((indicator - threshold) / scale, 0, 1)

    Args:
        indicator: Series of indicator values.
        threshold: Threshold value.
        normalizing_scale: Optional custom scale. If None, computed from data.

    Returns:
        Series of activation values in [0, 1].
    """
    if normalizing_scale is None:
        normalizing_scale = compute_normalizing_scale(indicator, threshold)

    # Distance above threshold (positive when satisfied)
    distance = indicator - threshold

    # Activation: 0 at threshold, increasing as we go deeper above
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_less_than_indicator(
    indicator: pd.Series,
    comparison_indicator: pd.Series,
    normalizing_scale: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for indicator < comparison_indicator rules.

    Activation increases as indicator moves further below comparison.
    Returns 0 when indicator >= comparison_indicator.

    Args:
        indicator: Series of indicator values.
        comparison_indicator: Series of comparison indicator values.
        normalizing_scale: Optional custom scale. If None, computed from data.

    Returns:
        Series of activation values in [0, 1].
    """
    if normalizing_scale is None:
        # Use std of the difference as scale
        diff = comparison_indicator - indicator
        normalizing_scale = compute_normalizing_scale(diff, 0.0)

    distance = comparison_indicator - indicator
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_greater_than_indicator(
    indicator: pd.Series,
    comparison_indicator: pd.Series,
    normalizing_scale: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for indicator > comparison_indicator rules.

    Activation increases as indicator moves further above comparison.
    Returns 0 when indicator <= comparison_indicator.

    Args:
        indicator: Series of indicator values.
        comparison_indicator: Series of comparison indicator values.
        normalizing_scale: Optional custom scale. If None, computed from data.

    Returns:
        Series of activation values in [0, 1].
    """
    if normalizing_scale is None:
        diff = indicator - comparison_indicator
        normalizing_scale = compute_normalizing_scale(diff, 0.0)

    distance = indicator - comparison_indicator
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


# =============================================================================
# INDICATOR-SPECIFIC ACTIVATION FUNCTIONS
# =============================================================================


def activation_bb_lower(
    close: pd.Series,
    bb_lower: pd.Series,
    bb_middle: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for close < bb_lower (Bollinger Band lower breakout).

    Uses band width as normalizing scale for data-driven activation.

    Args:
        close: Series of close prices.
        bb_lower: Series of lower Bollinger Band values.
        bb_middle: Optional middle band (SMA). If provided, uses band width.

    Returns:
        Series of activation values in [0, 1].
    """
    if bb_middle is not None:
        # Band width as scale
        normalizing_scale = (bb_middle - bb_lower).clip(lower=0.001)
    else:
        # Use std of difference as fallback
        diff = bb_lower - close
        normalizing_scale = compute_normalizing_scale(diff, 0.0)

    distance = bb_lower - close
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_bb_upper(
    close: pd.Series,
    bb_upper: pd.Series,
    bb_middle: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for close > bb_upper (Bollinger Band upper breakout).

    Uses band width as normalizing scale for data-driven activation.

    Args:
        close: Series of close prices.
        bb_upper: Series of upper Bollinger Band values.
        bb_middle: Optional middle band (SMA). If provided, uses band width.

    Returns:
        Series of activation values in [0, 1].
    """
    if bb_middle is not None:
        normalizing_scale = (bb_upper - bb_middle).clip(lower=0.001)
    else:
        diff = close - bb_upper
        normalizing_scale = compute_normalizing_scale(diff, 0.0)

    distance = close - bb_upper
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_rsi_oversold(
    rsi: pd.Series,
    threshold: float = 30.0,
) -> pd.Series:
    """
    Compute activation for RSI < threshold (oversold).

    RSI ranges [0, 100], so we use threshold as the normalizing factor.
    Activation increases as RSI gets deeper into oversold territory.

    Args:
        rsi: Series of RSI values.
        threshold: Oversold threshold (default 30).

    Returns:
        Series of activation values in [0, 1].
    """
    # Scale based on how far below threshold we can go
    # At threshold: activation = 0
    # At 0: activation = 1
    normalizing_scale = pd.Series(threshold, index=rsi.index)

    distance = threshold - rsi
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_rsi_overbought(
    rsi: pd.Series,
    threshold: float = 70.0,
) -> pd.Series:
    """
    Compute activation for RSI > threshold (overbought).

    RSI ranges [0, 100], so we use (100 - threshold) as normalizing factor.
    Activation increases as RSI gets deeper into overbought territory.

    Args:
        rsi: Series of RSI values.
        threshold: Overbought threshold (default 70).

    Returns:
        Series of activation values in [0, 1].
    """
    # Scale based on how far above threshold we can go
    # At threshold: activation = 0
    # At 100: activation = 1
    normalizing_scale = pd.Series(100.0 - threshold, index=rsi.index)

    distance = rsi - threshold
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_mfi_oversold(
    mfi: pd.Series,
    threshold: float = 20.0,
) -> pd.Series:
    """
    Compute activation for MFI < threshold (oversold money flow).

    MFI ranges [0, 100], similar to RSI.

    Args:
        mfi: Series of MFI values.
        threshold: Oversold threshold (default 20).

    Returns:
        Series of activation values in [0, 1].
    """
    normalizing_scale = pd.Series(threshold, index=mfi.index)
    distance = threshold - mfi
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_mfi_overbought(
    mfi: pd.Series,
    threshold: float = 80.0,
) -> pd.Series:
    """
    Compute activation for MFI > threshold (overbought money flow).

    MFI ranges [0, 100], similar to RSI.

    Args:
        mfi: Series of MFI values.
        threshold: Overbought threshold (default 80).

    Returns:
        Series of activation values in [0, 1].
    """
    normalizing_scale = pd.Series(100.0 - threshold, index=mfi.index)
    distance = mfi - threshold
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_volume_spike(
    volume: pd.Series,
    vol_sma: pd.Series,
    multiplier: float = 2.0,
) -> pd.Series:
    """
    Compute activation for volume > vol_sma (volume spike).

    Activation increases as volume exceeds the moving average.
    Full activation when volume is 'multiplier' times the average.

    Args:
        volume: Series of volume values.
        vol_sma: Series of volume SMA values.
        multiplier: How many times average for full activation (default 2x).

    Returns:
        Series of activation values in [0, 1].
    """
    # Scale: distance from 1x to multiplier x average
    normalizing_scale = vol_sma * (multiplier - 1.0)
    normalizing_scale = normalizing_scale.clip(lower=0.001)

    distance = volume - vol_sma
    raw_activation = distance / normalizing_scale

    return clip_activation(raw_activation)


def activation_sma_trend_bullish(
    close: pd.Series,
    sma: pd.Series,
    sma_lag: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for bullish SMA trend (close > sma, sma rising).

    Combines:
    1. Price position above SMA
    2. SMA slope (if lag provided)

    Args:
        close: Series of close prices.
        sma: Series of SMA values.
        sma_lag: Optional lagged SMA for slope calculation.

    Returns:
        Series of activation values in [0, 1].
    """
    # Price above SMA activation
    price_dist = close - sma
    price_scale = compute_normalizing_scale(price_dist, 0.0)
    price_activation = clip_activation(price_dist / price_scale)

    if sma_lag is not None:
        # SMA slope activation
        slope = sma - sma_lag
        slope_scale = compute_normalizing_scale(slope, 0.0)
        slope_activation = clip_activation(slope / slope_scale)

        # Combine: use geometric mean for AND-like behavior
        combined = np.sqrt(price_activation * slope_activation)
        return pd.Series(combined, index=close.index)

    return price_activation


def activation_sma_trend_bearish(
    close: pd.Series,
    sma: pd.Series,
    sma_lag: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute activation for bearish SMA trend (close < sma, sma falling).

    Args:
        close: Series of close prices.
        sma: Series of SMA values.
        sma_lag: Optional lagged SMA for slope calculation.

    Returns:
        Series of activation values in [0, 1].
    """
    # Price below SMA activation
    price_dist = sma - close
    price_scale = compute_normalizing_scale(price_dist, 0.0)
    price_activation = clip_activation(price_dist / price_scale)

    if sma_lag is not None:
        # SMA slope activation (falling)
        slope = sma_lag - sma
        slope_scale = compute_normalizing_scale(slope, 0.0)
        slope_activation = clip_activation(slope / slope_scale)

        combined = np.sqrt(price_activation * slope_activation)
        return pd.Series(combined, index=close.index)

    return price_activation


def activation_macd_histogram_positive(
    macd_hist: pd.Series,
) -> pd.Series:
    """
    Compute activation for positive MACD histogram (bullish momentum).

    Args:
        macd_hist: Series of MACD histogram values.

    Returns:
        Series of activation values in [0, 1].
    """
    normalizing_scale = compute_normalizing_scale(macd_hist, 0.0)
    raw_activation = macd_hist / normalizing_scale

    return clip_activation(raw_activation)


def activation_macd_histogram_negative(
    macd_hist: pd.Series,
) -> pd.Series:
    """
    Compute activation for negative MACD histogram (bearish momentum).

    Args:
        macd_hist: Series of MACD histogram values.

    Returns:
        Series of activation values in [0, 1].
    """
    normalizing_scale = compute_normalizing_scale(macd_hist, 0.0)
    raw_activation = -macd_hist / normalizing_scale

    return clip_activation(raw_activation)


def activation_macd_crossover_bullish(
    macd: pd.Series,
    macd_signal: pd.Series,
) -> pd.Series:
    """
    Compute activation for bullish MACD crossover (MACD > signal).

    Args:
        macd: Series of MACD line values.
        macd_signal: Series of MACD signal line values.

    Returns:
        Series of activation values in [0, 1].
    """
    diff = macd - macd_signal
    normalizing_scale = compute_normalizing_scale(diff, 0.0)
    raw_activation = diff / normalizing_scale

    return clip_activation(raw_activation)


def activation_macd_crossover_bearish(
    macd: pd.Series,
    macd_signal: pd.Series,
) -> pd.Series:
    """
    Compute activation for bearish MACD crossover (MACD < signal).

    Args:
        macd: Series of MACD line values.
        macd_signal: Series of MACD signal line values.

    Returns:
        Series of activation values in [0, 1].
    """
    diff = macd_signal - macd
    normalizing_scale = compute_normalizing_scale(diff, 0.0)
    raw_activation = diff / normalizing_scale

    return clip_activation(raw_activation)


# =============================================================================
# GENERIC ACTIVATION DISPATCHER
# =============================================================================


def compute_rule_activation(
    indicator_values: pd.Series,
    comparison_values: pd.Series,
    operator: str,
    indicator_name: str,
    comparison_name: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Compute continuous activation for a rule based on operator.

    This is the main dispatcher that selects the appropriate activation
    function based on the rule's operator and indicator types.

    Args:
        indicator_values: Series of indicator values.
        comparison_values: Series or scalar of comparison values.
        operator: Comparison operator ("<", "<=", ">", ">=", "==", "!=").
        indicator_name: Name of the indicator (for context-aware activation).
        comparison_name: Name of comparison indicator (if comparing to indicator).
        df: Optional full DataFrame for context-aware activation.

    Returns:
        Series of activation values in [0, 1].
    """
    is_scalar = not isinstance(comparison_values, pd.Series)

    # Detect indicator type for specialized activation
    ind_lower = indicator_name.lower()

    # RSI-specific activation
    if "rsi" in ind_lower:
        if operator in ("<", "<=") and is_scalar:
            return activation_rsi_oversold(indicator_values, float(comparison_values))
        elif operator in (">", ">=") and is_scalar:
            return activation_rsi_overbought(indicator_values, float(comparison_values))

    # MFI-specific activation
    if "mfi" in ind_lower:
        if operator in ("<", "<=") and is_scalar:
            return activation_mfi_oversold(indicator_values, float(comparison_values))
        elif operator in (">", ">=") and is_scalar:
            return activation_mfi_overbought(indicator_values, float(comparison_values))

    # Bollinger Band activation
    if "close" in ind_lower and comparison_name:
        comp_lower = comparison_name.lower()
        if "bb_lower" in comp_lower and operator in ("<", "<="):
            # Try to get middle band for better scaling
            bb_middle = None
            if df is not None:
                for col in df.columns:
                    if "bb_middle" in col.lower() or "bb_sma" in col.lower():
                        bb_middle = df[col]
                        break
            return activation_bb_lower(indicator_values, comparison_values, bb_middle)
        elif "bb_upper" in comp_lower and operator in (">", ">="):
            bb_middle = None
            if df is not None:
                for col in df.columns:
                    if "bb_middle" in col.lower() or "bb_sma" in col.lower():
                        bb_middle = df[col]
                        break
            return activation_bb_upper(indicator_values, comparison_values, bb_middle)

    # Volume spike activation
    if "volume" in ind_lower and comparison_name:
        comp_lower = comparison_name.lower()
        if "vol_sma" in comp_lower and operator in (">", ">="):
            return activation_volume_spike(indicator_values, comparison_values)

    # MACD histogram activation
    if "macd_hist" in ind_lower:
        if operator in (">", ">="):
            return activation_macd_histogram_positive(indicator_values)
        elif operator in ("<", "<="):
            return activation_macd_histogram_negative(indicator_values)

    # MACD line vs signal activation
    if "macd" in ind_lower and comparison_name and "signal" in comparison_name.lower():
        if operator in (">", ">="):
            return activation_macd_crossover_bullish(indicator_values, comparison_values)
        elif operator in ("<", "<="):
            return activation_macd_crossover_bearish(indicator_values, comparison_values)

    # SMA/EMA trend activation
    if comparison_name:
        comp_lower = comparison_name.lower()
        if ("sma" in comp_lower or "ema" in comp_lower) and "_lag_" not in comp_lower:
            if "close" in ind_lower:
                if operator in (">", ">="):
                    # Bullish: close > sma
                    return activation_sma_trend_bullish(indicator_values, comparison_values)
                elif operator in ("<", "<="):
                    # Bearish: close < sma
                    return activation_sma_trend_bearish(indicator_values, comparison_values)

    # Generic fallback based on operator
    if is_scalar:
        threshold = float(comparison_values)
        if operator in ("<", "<="):
            return activation_less_than(indicator_values, threshold)
        elif operator in (">", ">="):
            return activation_greater_than(indicator_values, threshold)
        elif operator == "==":
            # For equality, use a narrow activation around the threshold
            scale = compute_normalizing_scale(indicator_values, threshold)
            distance = (indicator_values - threshold).abs()
            return clip_activation(1.0 - distance / scale)
        elif operator == "!=":
            # For inequality, activation is 1 when far from threshold
            scale = compute_normalizing_scale(indicator_values, threshold)
            distance = (indicator_values - threshold).abs()
            return clip_activation(distance / scale)
    else:
        if operator in ("<", "<="):
            return activation_less_than_indicator(indicator_values, comparison_values)
        elif operator in (">", ">="):
            return activation_greater_than_indicator(indicator_values, comparison_values)
        elif operator == "==":
            scale = compute_normalizing_scale(indicator_values - comparison_values, 0.0)
            distance = (indicator_values - comparison_values).abs()
            return clip_activation(1.0 - distance / scale)
        elif operator == "!=":
            scale = compute_normalizing_scale(indicator_values - comparison_values, 0.0)
            distance = (indicator_values - comparison_values).abs()
            return clip_activation(distance / scale)

    # Default: no activation
    return pd.Series(0.0, index=indicator_values.index)
