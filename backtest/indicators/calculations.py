"""
Technical Indicator Calculations Module.

This module provides pure, vectorized functions for calculating
technical indicators. All functions operate on pandas Series/DataFrames
and return new data without modifying inputs.

Indicators implemented:
    - Trend/Momentum: SMA, EMA, MACD, RSI, MFI
    - Volatility: ATR, Historical Volatility
    - Volume: OBV, Volume SMA
    - Market Structure: Pivot Points, Bollinger Bands
"""

from typing import Tuple

import numpy as np
import pandas as pd

# Constants
TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Moving Averages
# =============================================================================


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        series: Price series.
        period: Lookback period.

    Returns:
        SMA series with NaN for insufficient lookback.
    """
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Uses span-based smoothing: alpha = 2 / (period + 1)

    Args:
        series: Price series.
        period: Lookback period (span).

    Returns:
        EMA series.
    """
    return series.ewm(span=period, adjust=False).mean()


# =============================================================================
# MACD (Moving Average Convergence Divergence)
# =============================================================================


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicator components.

    Args:
        close: Close price series.
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    ema_fast = calculate_ema(close, fast_period)
    ema_slow = calculate_ema(close, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =============================================================================
# RSI (Relative Strength Index)
# =============================================================================


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index using Wilder's smoothing.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Uses Wilder's smoothing (RMA) with alpha = 1/period.

    Args:
        close: Close price series.
        period: Lookback period (default 14).

    Returns:
        RSI series (0-100 scale).
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Calculate RSI with proper handling of edge cases
    # When avg_loss is 0 (all gains), RSI should be 100
    # When avg_gain is 0 (all losses), RSI should be 0
    rsi = pd.Series(index=close.index, dtype=float)

    # Normal case: avg_loss > 0
    normal_mask = avg_loss > 0
    rs = avg_gain[normal_mask] / avg_loss[normal_mask]
    rsi[normal_mask] = 100.0 - (100.0 / (1.0 + rs))

    # Edge case: avg_loss == 0 (all gains) -> RSI = 100
    all_gains_mask = (avg_loss == 0) & (avg_gain > 0)
    rsi[all_gains_mask] = 100.0

    # Edge case: both are 0 or NaN
    both_zero_mask = (avg_loss == 0) & (avg_gain == 0)
    rsi[both_zero_mask] = 50.0  # Neutral

    return rsi


# =============================================================================
# MFI (Money Flow Index)
# =============================================================================


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Money Flow Index.

    MFI is volume-weighted RSI, measuring buying/selling pressure.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: Lookback period (default 14).

    Returns:
        MFI series (0-100 scale).
    """
    # Typical Price
    typical_price = (high + low + close) / 3.0

    # Raw Money Flow
    raw_money_flow = typical_price * volume

    # Direction based on typical price change
    tp_diff = typical_price.diff()

    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)

    # Sum over period
    positive_sum = positive_flow.rolling(window=period, min_periods=period).sum()
    negative_sum = negative_flow.rolling(window=period, min_periods=period).sum()

    # Calculate MFI with proper handling of edge cases
    mfi = pd.Series(index=close.index, dtype=float)

    # Normal case: negative_sum > 0
    normal_mask = negative_sum > 0
    mf_ratio = positive_sum[normal_mask] / negative_sum[normal_mask]
    mfi[normal_mask] = 100.0 - (100.0 / (1.0 + mf_ratio))

    # Edge case: negative_sum == 0 (all positive flow) -> MFI = 100
    all_positive_mask = (negative_sum == 0) & (positive_sum > 0)
    mfi[all_positive_mask] = 100.0

    # Edge case: both are 0
    both_zero_mask = (negative_sum == 0) & (positive_sum == 0)
    mfi[both_zero_mask] = 50.0

    return mfi


# =============================================================================
# ATR (Average True Range)
# =============================================================================


def calculate_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Calculate True Range.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.

    Returns:
        True Range series.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range using Wilder's smoothing.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback period (default 14).

    Returns:
        ATR series.
    """
    true_range = calculate_true_range(high, low, close)

    # Wilder's smoothing
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return atr


# =============================================================================
# Historical Volatility
# =============================================================================


def calculate_historical_volatility(
    close: pd.Series, period: int = 20, annualize: bool = True
) -> pd.Series:
    """
    Calculate Historical Volatility (realized volatility of log returns).

    Args:
        close: Close price series.
        period: Lookback period (default 20).
        annualize: Whether to annualize the volatility (default True).

    Returns:
        Historical volatility series.
    """
    # Log returns
    log_returns = np.log(close / close.shift(1))

    # Rolling standard deviation
    hv = log_returns.rolling(window=period, min_periods=period).std()

    # Annualize if requested
    if annualize:
        hv = hv * np.sqrt(TRADING_DAYS_PER_YEAR)

    return hv


# =============================================================================
# OBV (On-Balance Volume)
# =============================================================================


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume.

    OBV adds volume on up days and subtracts on down days.

    Args:
        close: Close price series.
        volume: Volume series.

    Returns:
        OBV series.
    """
    # Direction: +1 if price up, -1 if down, 0 if unchanged
    price_change = close.diff()
    direction = np.sign(price_change)

    # First day has no prior price, set direction to +1 (add volume)
    direction.iloc[0] = 1.0

    # Signed volume
    signed_volume = direction * volume

    # Cumulative sum
    obv = signed_volume.cumsum()

    return obv


# =============================================================================
# Volume SMA
# =============================================================================


def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average of Volume.

    Args:
        volume: Volume series.
        period: Lookback period (default 20).

    Returns:
        Volume SMA series.
    """
    return calculate_sma(volume, period)


# =============================================================================
# Pivot Points
# =============================================================================


def calculate_pivot_high(
    high: pd.Series, left_bars: int = 3, right_bars: int = 3
) -> pd.Series:
    """
    Calculate Pivot High points (vectorized implementation).

    A pivot high is a bar where the high is greater than the highs
    of `left_bars` bars to the left and `right_bars` bars to the right.

    Args:
        high: High price series.
        left_bars: Number of bars to check on the left.
        right_bars: Number of bars to check on the right.

    Returns:
        Series with pivot high values, NaN elsewhere.
    """
    # Vectorized approach using rolling max
    # A point is a pivot high if it's greater than all surrounding points
    total_window = left_bars + 1 + right_bars

    # Rolling max of the window
    rolling_max = high.rolling(window=total_window, center=True).max()

    # A point is a pivot high if it equals the rolling max AND
    # it's strictly greater than both left and right rolling maxes
    left_max = high.shift(1).rolling(window=left_bars, min_periods=left_bars).max()
    right_max = high.shift(-right_bars).rolling(window=right_bars, min_periods=right_bars).max()

    # Point is pivot if it's strictly greater than both sides
    is_pivot = (high > left_max) & (high > right_max)

    result = pd.Series(np.nan, index=high.index)
    result[is_pivot] = high[is_pivot]

    return result


def calculate_pivot_low(
    low: pd.Series, left_bars: int = 3, right_bars: int = 3
) -> pd.Series:
    """
    Calculate Pivot Low points (vectorized implementation).

    A pivot low is a bar where the low is less than the lows
    of `left_bars` bars to the left and `right_bars` bars to the right.

    Args:
        low: Low price series.
        left_bars: Number of bars to check on the left.
        right_bars: Number of bars to check on the right.

    Returns:
        Series with pivot low values, NaN elsewhere.
    """
    # Vectorized approach using rolling min
    # A point is a pivot low if it's less than all surrounding points
    left_min = low.shift(1).rolling(window=left_bars, min_periods=left_bars).min()
    right_min = low.shift(-right_bars).rolling(window=right_bars, min_periods=right_bars).min()

    # Point is pivot if it's strictly less than both sides
    is_pivot = (low < left_min) & (low < right_min)

    result = pd.Series(np.nan, index=low.index)
    result[is_pivot] = low[is_pivot]

    return result


# =============================================================================
# Bollinger Bands
# =============================================================================


def calculate_bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        close: Close price series.
        period: SMA period (default 20).
        std_dev: Standard deviation multiplier (default 2).

    Returns:
        Tuple of (middle_band, upper_band, lower_band).
    """
    middle = calculate_sma(close, period)
    rolling_std = close.rolling(window=period, min_periods=period).std()

    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)

    return middle, upper, lower


# =============================================================================
# Main Indicator Builder
# =============================================================================


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add all indicators to the DataFrame.

    This is the main function that adds all required indicators
    to an OHLCV DataFrame.

    Args:
        df: DataFrame with Date, Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame with all indicators added.
    """
    df = df.copy()

    # Extract price and volume series
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- Trend / Momentum Indicators ---

    # SMAs
    df["sma_5"] = calculate_sma(close, 5)
    df["sma_9"] = calculate_sma(close, 9)
    df["sma_20"] = calculate_sma(close, 20)
    df["sma_50"] = calculate_sma(close, 50)
    df["sma_200"] = calculate_sma(close, 200)

    # EMAs
    df["ema_9"] = calculate_ema(close, 9)
    df["ema_21"] = calculate_ema(close, 21)
    df["ema_50"] = calculate_ema(close, 50)

    # MACD
    macd, signal, hist = calculate_macd(close, 12, 26, 9)
    df["macd_12_26_9"] = macd
    df["macd_signal_12_26_9"] = signal
    df["macd_hist_12_26_9"] = hist

    # RSI
    df["rsi_14"] = calculate_rsi(close, 14)

    # MFI
    df["mfi_14"] = calculate_mfi(high, low, close, volume, 14)

    # --- Volatility Indicators ---

    # ATR
    df["atr_14"] = calculate_atr(high, low, close, 14)

    # Historical Volatility
    df["hv_20"] = calculate_historical_volatility(close, 20, annualize=True)

    # --- Volume-Based Indicators ---

    # OBV
    df["obv"] = calculate_obv(close, volume)

    # Volume SMA
    df["vol_sma_20"] = calculate_volume_sma(volume, 20)

    # --- Market Structure ---

    # Pivot Points
    df["pivot_high_3"] = calculate_pivot_high(high, 3, 3)
    df["pivot_low_3"] = calculate_pivot_low(low, 3, 3)

    # Bollinger Bands
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    df["bb_mid_20_2"] = bb_mid
    df["bb_upper_20_2"] = bb_upper
    df["bb_lower_20_2"] = bb_lower

    return df
