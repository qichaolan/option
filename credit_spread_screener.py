#!/usr/bin/env python3
"""
Credit Spread Screener

Scans Put Credit Spreads (PCS) and Call Credit Spreads (CCS) candidates
for a list of underlying tickers using OpenBB.

Features:
- Pull option chains from OpenBB
- Build PCS and CCS candidates
- Compute key metrics (ROC, delta, probability, IV percentile, etc.)
- Score and rank spreads
- Filter and sort by multiple criteria
- Export results to CSV
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ScreenerConfig:
    """Configuration for the credit spread screener."""
    tickers: list[str]
    min_dte: int = 14
    max_dte: int = 30
    min_delta: float = 0.10
    max_delta: float = 0.25
    max_width: float = 10.0
    min_roc: float = 0.20
    min_ivp: int = 40
    output_csv: Optional[str] = None

    # Scoring weights
    prob_weight: float = 0.0       # filtered, not ranked
    roc_weight: float = 0.40
    convexity_weight: float = 0.30
    slippage_weight: float = 0.15
    liquidity_weight: float = 0.15
    ease_weight: float = 0.0       # removed from scoring

    # Filtering thresholds
    min_liquidity_score: float = 0.1
    min_slippage_score: float = 0.1


def fetch_current_price(symbol: str) -> float:
    """
    Fetch current underlying price using OpenBB.

    Parameters
    ----------
    symbol : str
        Ticker symbol.

    Returns
    -------
    float
        Current price of the underlying.

    Raises
    ------
    ValueError
        If price cannot be retrieved.
    """
    from openbb import obb

    try:
        result = obb.equity.price.quote(symbol=symbol.upper(), provider="yfinance")
        df = result.to_df()

        for col in ["last_price", "close", "price", "regularMarketPrice", "previousClose"]:
            if col in df.columns:
                price = df[col].iloc[0]
                if price is not None and not pd.isna(price):
                    return float(price)
    except Exception as exc:
        logger.debug(f"OpenBB quote failed for {symbol}: {exc}")

    # Fallback to yfinance directly
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")

        if price is None:
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]

        if price is not None:
            return float(price)
    except Exception as exc:
        logger.debug(f"yfinance fallback failed for {symbol}: {exc}")

    raise ValueError(f"Could not fetch price for {symbol}")


def fetch_option_chain(symbol: str, min_dte: int, max_dte: int) -> pd.DataFrame:
    """
    Fetch options chain for a symbol within the DTE range.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    min_dte : int
        Minimum days to expiration.
    max_dte : int
        Maximum days to expiration.

    Returns
    -------
    pd.DataFrame
        Options chain with columns: strike, bid, ask, volume, openInterest,
        impliedVolatility, delta, option_type, expiration, dte.
    """
    from openbb import obb

    try:
        result = obb.derivatives.options.chains(symbol=symbol.upper(), provider="yfinance")
        df = result.to_df()
    except Exception as exc:
        logger.warning(f"OpenBB chains failed for {symbol}: {exc}, trying yfinance directly")
        df = _fetch_chain_yfinance(symbol)

    if df.empty:
        return pd.DataFrame()

    # Standardize column names
    col_mapping = {
        "last_trade_price": "lastPrice",
        "open_interest": "openInterest",
        "implied_volatility": "impliedVolatility",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # Ensure expiration is datetime
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"])

    # Calculate DTE
    today = pd.Timestamp.now().normalize()
    df["dte"] = (df["expiration"] - today).dt.days

    # Filter by DTE range
    df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)].copy()

    # Calculate mid price
    if "mid" not in df.columns:
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2

    return df


def estimate_delta(
    strike: float,
    underlying_price: float,
    dte: int,
    iv: float,
    option_type: str,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Estimate option delta using Black-Scholes formula.

    Parameters
    ----------
    strike : float
        Strike price.
    underlying_price : float
        Current underlying price.
    dte : int
        Days to expiration.
    iv : float
        Implied volatility (annualized, e.g., 0.25 for 25%).
    option_type : str
        "call" or "put".
    risk_free_rate : float
        Risk-free rate (annualized).

    Returns
    -------
    float
        Estimated delta (absolute value for puts).
    """
    if dte <= 0 or iv <= 0 or underlying_price <= 0 or strike <= 0:
        return 0.0

    t = dte / 365.0
    sqrt_t = np.sqrt(t)

    d1 = (np.log(underlying_price / strike) + (risk_free_rate + 0.5 * iv ** 2) * t) / (iv * sqrt_t)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1  # Negative for puts

    return delta


def _fetch_chain_yfinance(symbol: str) -> pd.DataFrame:
    """Fallback to fetch option chain directly from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol.upper())
        expirations = stock.options

        all_chains = []
        for exp in expirations:
            try:
                opt = stock.option_chain(exp)
                calls = opt.calls.copy()
                calls["option_type"] = "call"
                puts = opt.puts.copy()
                puts["option_type"] = "put"
                calls["expiration"] = pd.to_datetime(exp)
                puts["expiration"] = pd.to_datetime(exp)
                all_chains.extend([calls, puts])
            except Exception:
                continue

        if all_chains:
            return pd.concat(all_chains, ignore_index=True)
    except Exception as exc:
        logger.error(f"yfinance chain fetch failed for {symbol}: {exc}")

    return pd.DataFrame()


def fetch_iv_percentile(symbol: str) -> float:
    """
    Fetch IV percentile for a symbol.

    Uses historical IV data to compute percentile. Falls back to a default
    if historical data is unavailable.

    Parameters
    ----------
    symbol : str
        Ticker symbol.

    Returns
    -------
    float
        IV percentile (0-100).
    """
    try:
        from openbb import obb

        # Attempt to get historical volatility data
        result = obb.equity.price.historical(symbol=symbol.upper(), period="1y")
        df = result.to_df()

        if df.empty:
            return 50.0

        # Calculate historical volatility (20-day rolling)
        df["returns"] = df["close"].pct_change()
        df["hv_20"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

        # Get current HV and compute percentile
        current_hv = df["hv_20"].iloc[-1]
        if pd.isna(current_hv):
            return 50.0

        hv_values = df["hv_20"].dropna()
        ivp = (hv_values < current_hv).sum() / len(hv_values) * 100

        return float(ivp)

    except Exception as exc:
        logger.debug(f"IV percentile calculation failed for {symbol}: {exc}")
        return 50.0


def build_credit_spreads_from_chain(
    chain: pd.DataFrame,
    underlying_price: float,
    config: ScreenerConfig,
    symbol: str,
    ivp: float,
) -> pd.DataFrame:
    """
    Build Put Credit Spreads (PCS) and Call Credit Spreads (CCS) from an options chain.

    Parameters
    ----------
    chain : pd.DataFrame
        Options chain data.
    underlying_price : float
        Current price of the underlying.
    config : ScreenerConfig
        Screener configuration.
    symbol : str
        Ticker symbol.
    ivp : float
        IV percentile for the symbol.

    Returns
    -------
    pd.DataFrame
        DataFrame of spread candidates with computed metrics.
    """
    spreads = []

    # Get unique expirations
    expirations = chain["expiration"].unique()

    for expiration in expirations:
        exp_chain = chain[chain["expiration"] == expiration]
        dte = exp_chain["dte"].iloc[0] if len(exp_chain) > 0 else 0

        # Build PCS (Put Credit Spreads)
        puts = exp_chain[exp_chain["option_type"] == "put"].copy()
        pcs_spreads = _build_put_credit_spreads(
            puts, underlying_price, config, symbol, expiration, dte, ivp
        )
        spreads.extend(pcs_spreads)

        # Build CCS (Call Credit Spreads)
        calls = exp_chain[exp_chain["option_type"] == "call"].copy()
        ccs_spreads = _build_call_credit_spreads(
            calls, underlying_price, config, symbol, expiration, dte, ivp
        )
        spreads.extend(ccs_spreads)

    if not spreads:
        return pd.DataFrame()

    return pd.DataFrame(spreads)


def _build_put_credit_spreads(
    puts: pd.DataFrame,
    underlying_price: float,
    config: ScreenerConfig,
    symbol: str,
    expiration: datetime,
    dte: int,
    ivp: float,
) -> list[dict]:
    """Build Put Credit Spreads from puts chain."""
    spreads = []

    # Sort puts by strike descending (higher strikes first for short leg)
    puts = puts.sort_values("strike", ascending=False).reset_index(drop=True)

    # Filter OTM puts (strike < underlying)
    otm_puts = puts[puts["strike"] < underlying_price].copy()

    if len(otm_puts) < 2:
        return spreads

    puts_array = otm_puts.to_dict("records")

    for i, short_put in enumerate(puts_array):
        short_strike = short_put["strike"]
        short_bid = short_put.get("bid", 0) or 0
        short_ask = short_put.get("ask", 0) or 0
        short_mid = (short_bid + short_ask) / 2 if short_bid > 0 and short_ask > 0 else 0
        short_iv = short_put.get("impliedVolatility") or short_put.get("iv") or 0.25
        short_oi = short_put.get("openInterest", 0) or 0
        short_volume = short_put.get("volume", 0) or 0

        # Get delta from chain or estimate it
        short_delta = short_put.get("delta")
        delta_estimated = False
        if short_delta is None or short_delta == 0:
            short_delta = estimate_delta(short_strike, underlying_price, dte, short_iv, "put")
            delta_estimated = True
        short_delta = abs(short_delta)

        # Check delta range for short leg
        if not (config.min_delta <= short_delta <= config.max_delta):
            continue

        # Find valid long puts (lower strikes)
        for long_put in puts_array[i + 1:]:
            long_strike = long_put["strike"]
            long_bid = long_put.get("bid", 0) or 0
            long_ask = long_put.get("ask", 0) or 0
            long_mid = (long_bid + long_ask) / 2 if long_bid > 0 and long_ask > 0 else 0
            long_oi = long_put.get("openInterest", 0) or 0
            long_volume = long_put.get("volume", 0) or 0

            # Check width constraint
            width = short_strike - long_strike
            if width <= 0 or width > config.max_width:
                continue

            # Calculate credit (sell short, buy long)
            # Credit = short_bid - long_ask (conservative)
            credit = short_bid - long_ask if short_bid > 0 and long_ask > 0 else 0

            if credit <= 0:
                continue

            # Calculate max loss
            max_loss = width - credit

            if max_loss <= 0:
                continue

            # Calculate ROC
            roc = credit / max_loss

            if roc < config.min_roc:
                continue

            # Break-even for PCS: short_strike - credit
            break_even = short_strike - credit

            # Break-even distance percentage
            break_even_distance_pct = (underlying_price - break_even) / underlying_price

            # Probability of profit (delta-based approximation)
            prob_profit = 1 - short_delta

            spread = {
                "symbol": symbol,
                "expiration": expiration.strftime("%Y-%m-%d") if hasattr(expiration, "strftime") else str(expiration)[:10],
                "dte": dte,
                "type": "PCS",
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": width,
                "mid_credit": (short_mid - long_mid) if short_mid > 0 and long_mid > 0 else credit,
                "credit": credit,
                "max_loss": max_loss,
                "roc": roc,
                "short_delta": short_delta,
                "delta_estimated": delta_estimated,
                "prob_profit": prob_profit,
                "iv": short_iv,
                "ivp": ivp,
                "underlying_price": underlying_price,
                "break_even": break_even,
                "break_even_distance_pct": break_even_distance_pct,
                "short_oi": short_oi,
                "long_oi": long_oi,
                "short_volume": short_volume,
                "long_volume": long_volume,
                "short_bid": short_bid,
                "short_ask": short_ask,
                "long_bid": long_bid,
                "long_ask": long_ask,
            }
            spreads.append(spread)

    return spreads


def _build_call_credit_spreads(
    calls: pd.DataFrame,
    underlying_price: float,
    config: ScreenerConfig,
    symbol: str,
    expiration: datetime,
    dte: int,
    ivp: float,
) -> list[dict]:
    """Build Call Credit Spreads from calls chain."""
    spreads = []

    # Sort calls by strike ascending (lower strikes first for short leg)
    calls = calls.sort_values("strike", ascending=True).reset_index(drop=True)

    # Filter OTM calls (strike > underlying)
    otm_calls = calls[calls["strike"] > underlying_price].copy()

    if len(otm_calls) < 2:
        return spreads

    calls_array = otm_calls.to_dict("records")

    for i, short_call in enumerate(calls_array):
        short_strike = short_call["strike"]
        short_bid = short_call.get("bid", 0) or 0
        short_ask = short_call.get("ask", 0) or 0
        short_mid = (short_bid + short_ask) / 2 if short_bid > 0 and short_ask > 0 else 0
        short_iv = short_call.get("impliedVolatility") or short_call.get("iv") or 0.25
        short_oi = short_call.get("openInterest", 0) or 0
        short_volume = short_call.get("volume", 0) or 0

        # Get delta from chain or estimate it
        short_delta = short_call.get("delta")
        delta_estimated = False
        if short_delta is None or short_delta == 0:
            short_delta = estimate_delta(short_strike, underlying_price, dte, short_iv, "call")
            delta_estimated = True
        short_delta = abs(short_delta)

        # Check delta range for short leg
        if not (config.min_delta <= short_delta <= config.max_delta):
            continue

        # Find valid long calls (higher strikes)
        for long_call in calls_array[i + 1:]:
            long_strike = long_call["strike"]
            long_bid = long_call.get("bid", 0) or 0
            long_ask = long_call.get("ask", 0) or 0
            long_mid = (long_bid + long_ask) / 2 if long_bid > 0 and long_ask > 0 else 0
            long_oi = long_call.get("openInterest", 0) or 0
            long_volume = long_call.get("volume", 0) or 0

            # Check width constraint
            width = long_strike - short_strike
            if width <= 0 or width > config.max_width:
                continue

            # Calculate credit (sell short, buy long)
            # Credit = short_bid - long_ask (conservative)
            credit = short_bid - long_ask if short_bid > 0 and long_ask > 0 else 0

            if credit <= 0:
                continue

            # Calculate max loss
            max_loss = width - credit

            if max_loss <= 0:
                continue

            # Calculate ROC
            roc = credit / max_loss

            if roc < config.min_roc:
                continue

            # Break-even for CCS: short_strike + credit
            break_even = short_strike + credit

            # Break-even distance percentage
            break_even_distance_pct = (break_even - underlying_price) / underlying_price

            # Probability of profit (delta-based approximation)
            prob_profit = 1 - short_delta

            spread = {
                "symbol": symbol,
                "expiration": expiration.strftime("%Y-%m-%d") if hasattr(expiration, "strftime") else str(expiration)[:10],
                "dte": dte,
                "type": "CCS",
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": width,
                "mid_credit": (short_mid - long_mid) if short_mid > 0 and long_mid > 0 else credit,
                "credit": credit,
                "max_loss": max_loss,
                "roc": roc,
                "short_delta": short_delta,
                "delta_estimated": delta_estimated,
                "prob_profit": prob_profit,
                "iv": short_iv,
                "ivp": ivp,
                "underlying_price": underlying_price,
                "break_even": break_even,
                "break_even_distance_pct": break_even_distance_pct,
                "short_oi": short_oi,
                "long_oi": long_oi,
                "short_volume": short_volume,
                "long_volume": long_volume,
                "short_bid": short_bid,
                "short_ask": short_ask,
                "long_bid": long_bid,
                "long_ask": long_ask,
            }
            spreads.append(spread)

    return spreads


def compute_spread_metrics(df: pd.DataFrame, config: ScreenerConfig) -> pd.DataFrame:
    """
    Compute additional metrics for spreads: liquidity_score, slippage_score,
    convexity_score, ease_score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of spread candidates.
    config : ScreenerConfig
        Screener configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional metric columns.
    """
    if df.empty:
        return df

    df = df.copy()

    # Liquidity score: based on min(OI, volume) of both legs
    liquidity_target = 100
    df["min_oi"] = df[["short_oi", "long_oi"]].min(axis=1)
    df["min_volume"] = df[["short_volume", "long_volume"]].min(axis=1)
    df["liquidity_raw"] = df[["min_oi", "min_volume"]].min(axis=1)
    df["liquidity_score"] = (df["liquidity_raw"] / liquidity_target).clip(0, 1)

    # Slippage score: based on bid-ask spread percentage
    def calc_slippage(row: pd.Series) -> float:
        short_mid = (row["short_bid"] + row["short_ask"]) / 2
        long_mid = (row["long_bid"] + row["long_ask"]) / 2

        if short_mid <= 0 or long_mid <= 0:
            return 0.5

        short_spread_pct = (row["short_ask"] - row["short_bid"]) / short_mid
        long_spread_pct = (row["long_ask"] - row["long_bid"]) / long_mid

        max_spread_pct = max(short_spread_pct, long_spread_pct)

        # Invert: lower spread = higher score
        # Normalize: 0% spread = 1.0, 10%+ spread = 0.0
        return max(0, 1 - max_spread_pct / 0.10)

    df["slippage_score"] = df.apply(calc_slippage, axis=1)

    # Convexity score: reward narrower spreads with good ROC and decent distance
    def calc_convexity(row: pd.Series) -> float:
        roc_component = min(row["roc"] / 0.5, 1.0)
        delta_component = 1.0 if config.min_delta <= row["short_delta"] <= config.max_delta else 0.5
        distance_component = min(abs(row["break_even_distance_pct"]) / 0.05, 1.0)
        width_component = 1 - min(row["width"] / config.max_width, 1.0)

        return 0.4 * roc_component + 0.3 * delta_component + 0.2 * distance_component + 0.1 * width_component

    df["convexity_score"] = df.apply(calc_convexity, axis=1)

    # Ease score: distance from underlying, DTE, liquidity, slippage
    def calc_ease(row: pd.Series) -> float:
        # Distance cushion (more = easier)
        distance_score = min(abs(row["break_even_distance_pct"]) / 0.10, 1.0)

        # DTE score (prefer 14-30 days, penalize extremes)
        dte = row["dte"]
        if 14 <= dte <= 30:
            dte_score = 1.0
        elif 7 <= dte < 14 or 30 < dte <= 45:
            dte_score = 0.7
        else:
            dte_score = 0.4

        # Combine with liquidity and slippage
        ease = (
            0.35 * distance_score +
            0.25 * dte_score +
            0.20 * row["liquidity_score"] +
            0.20 * row["slippage_score"]
        )
        return ease

    df["ease_score"] = df.apply(calc_ease, axis=1)

    return df


def score_spreads(df: pd.DataFrame, config: ScreenerConfig) -> pd.DataFrame:
    """
    Compute total composite score for each spread.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of spread candidates with metrics.
    config : ScreenerConfig
        Screener configuration with scoring weights.

    Returns
    -------
    pd.DataFrame
        DataFrame with total_score column added.
    """
    if df.empty:
        return df

    df = df.copy()

    def normalize_series_minmax(series: pd.Series) -> pd.Series:
        """Normalize a series to 0-1 range using min-max scaling."""
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return pd.Series([0.5] * len(series), index=series.index)
        return ((series - s_min) / (s_max - s_min)).clip(0, 1)

    # Normalize all raw scores to 0-1 range using min-max scaling
    df["normalized_roc"] = normalize_series_minmax(df["roc"])
    df["normalized_convexity"] = normalize_series_minmax(df["convexity_score"])
    df["normalized_slippage"] = normalize_series_minmax(df["slippage_score"])
    df["normalized_liquidity"] = normalize_series_minmax(df["liquidity_score"])
    df["normalized_prob"] = normalize_series_minmax(df["prob_profit"])
    df["normalized_ease"] = normalize_series_minmax(df["ease_score"])

    # Compute total score
    # Weights: roc=0.40, convexity=0.30, slippage=0.15, liquidity=0.15
    # prob_weight and ease_weight are 0 (filtered, not ranked)
    df["total_score"] = (
        config.prob_weight * df["normalized_prob"] +
        config.roc_weight * df["normalized_roc"] +
        config.convexity_weight * df["normalized_convexity"] +
        config.ease_weight * df["normalized_ease"] +
        config.slippage_weight * df["normalized_slippage"] +
        config.liquidity_weight * df["normalized_liquidity"]
    )

    return df


def filter_and_sort_spreads(df: pd.DataFrame, config: ScreenerConfig) -> pd.DataFrame:
    """
    Filter and sort spreads by total score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of scored spread candidates.
    config : ScreenerConfig
        Screener configuration.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()

    # Filter by IVP threshold
    df = df[df["ivp"] >= config.min_ivp]

    # Filter by liquidity score
    df = df[df["liquidity_score"] >= config.min_liquidity_score]

    # Filter by slippage score
    df = df[df["slippage_score"] >= config.min_slippage_score]

    # Sort by total score descending
    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)

    return df


def format_output_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the output table with key columns for display.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame of spread candidates.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with selected columns.
    """
    if df.empty:
        return df

    display_cols = [
        "symbol",
        "type",
        "expiration",
        "dte",
        "short_strike",
        "long_strike",
        "credit",
        "roc",
        "short_delta",
        "delta_est",
        "prob_profit",
        "ivp",
        "break_even",
        "break_even_distance_pct",
        "liquidity_score",
        "slippage_score",
        "total_score",
    ]

    result = df.copy()

    # Create delta_est column to indicate estimated delta (marked with *)
    if "delta_estimated" in result.columns:
        result["delta_est"] = result["delta_estimated"].apply(lambda x: "*" if x else "")
    else:
        result["delta_est"] = ""

    available_cols = [col for col in display_cols if col in result.columns]
    return result[available_cols].copy()


def run_screener(config: ScreenerConfig) -> pd.DataFrame:
    """
    Run the credit spread screener.

    Parameters
    ----------
    config : ScreenerConfig
        Screener configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame of filtered and scored spread candidates.
    """
    all_spreads = []

    for symbol in config.tickers:
        logger.info(f"Processing {symbol}...")

        try:
            # Fetch current price
            underlying_price = fetch_current_price(symbol)
            logger.info(f"  {symbol} price: ${underlying_price:.2f}")

            # Fetch IV percentile
            ivp = fetch_iv_percentile(symbol)
            logger.info(f"  {symbol} IVP: {ivp:.1f}%")

            # Fetch option chain
            chain = fetch_option_chain(symbol, config.min_dte, config.max_dte)

            if chain.empty:
                logger.warning(f"  No options found for {symbol} in DTE range")
                continue

            logger.info(f"  Found {len(chain)} options in DTE range [{config.min_dte}, {config.max_dte}]")

            # Build credit spreads
            spreads_df = build_credit_spreads_from_chain(
                chain, underlying_price, config, symbol, ivp
            )

            if spreads_df.empty:
                logger.warning(f"  No valid spreads found for {symbol}")
                continue

            logger.info(f"  Built {len(spreads_df)} spread candidates")
            all_spreads.append(spreads_df)

        except Exception as exc:
            logger.error(f"  Error processing {symbol}: {exc}")
            continue

    if not all_spreads:
        logger.warning("No spreads found for any ticker")
        return pd.DataFrame()

    # Combine all spreads
    df = pd.concat(all_spreads, ignore_index=True)
    logger.info(f"Total spreads before scoring: {len(df)}")

    # Compute metrics
    df = compute_spread_metrics(df, config)

    # Score spreads
    df = score_spreads(df, config)

    # Filter and sort
    df = filter_and_sort_spreads(df, config)
    logger.info(f"Total spreads after filtering: {len(df)}")

    return df


def print_results_table(df: pd.DataFrame) -> None:
    """Print formatted results table to console, with PCS and CCS ranked separately."""
    if df.empty:
        print("\nNo spreads found matching criteria.")
        return

    # Check if any deltas were estimated
    has_estimated_delta = False
    if "delta_estimated" in df.columns:
        has_estimated_delta = df["delta_estimated"].any()

    # Format numeric columns for display
    formatters = {
        "credit": "${:.2f}".format,
        "roc": "{:.1%}".format,
        "short_delta": "{:.2f}".format,
        "prob_profit": "{:.1%}".format,
        "ivp": "{:.0f}".format,
        "break_even": "${:.2f}".format,
        "break_even_distance_pct": "{:.1%}".format,
        "liquidity_score": "{:.2f}".format,
        "slippage_score": "{:.2f}".format,
        "total_score": "{:.3f}".format,
    }

    def format_df(input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply formatting to a DataFrame."""
        display_df = format_output_table(input_df)
        for col, fmt in formatters.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(fmt)
        return display_df

    # Split by spread type
    pcs_df = df[df["type"] == "PCS"].copy()
    ccs_df = df[df["type"] == "CCS"].copy()

    print("\n" + "=" * 140)
    print("CREDIT SPREAD SCREENER RESULTS")
    print("=" * 140)

    if has_estimated_delta:
        print("NOTE: Delta values marked with '*' are ESTIMATED using Black-Scholes (yfinance does not provide delta).")
        print("      For accurate delta, use a paid data provider (Intrinio, Tradier, CBOE).")
        print("-" * 140)

    # Print PCS results
    print("\n" + "-" * 140)
    print("PUT CREDIT SPREADS (PCS) - Bullish Strategy")
    print("-" * 140)
    if pcs_df.empty:
        print("No PCS spreads found matching criteria.")
    else:
        pcs_display = format_df(pcs_df)
        print(pcs_display.to_string(index=False))
        print(f"Total PCS: {len(pcs_df)}")

    # Print CCS results
    print("\n" + "-" * 140)
    print("CALL CREDIT SPREADS (CCS) - Bearish Strategy")
    print("-" * 140)
    if ccs_df.empty:
        print("No CCS spreads found matching criteria.")
    else:
        ccs_display = format_df(ccs_df)
        print(ccs_display.to_string(index=False))
        print(f"Total CCS: {len(ccs_df)}")

    print("\n" + "=" * 140)
    print(f"TOTAL SPREADS: {len(df)} (PCS: {len(pcs_df)}, CCS: {len(ccs_df)})")
    print("=" * 140)


def parse_args() -> ScreenerConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Spread Screener - Find optimal PCS and CCS candidates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=["SPY", "QQQ", "NVDA"],
        help="List of ticker symbols to scan",
    )
    parser.add_argument(
        "--min-dte",
        type=int,
        default=14,
        dest="min_dte",
        help="Minimum days to expiration",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=30,
        dest="max_dte",
        help="Maximum days to expiration",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.10,
        dest="min_delta",
        help="Minimum short leg delta",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.25,
        dest="max_delta",
        help="Maximum short leg delta",
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=10.0,
        dest="max_width",
        help="Maximum spread width in dollars",
    )
    parser.add_argument(
        "--min-roc",
        type=float,
        default=0.20,
        dest="min_roc",
        help="Minimum return on capital (ROC) threshold",
    )
    parser.add_argument(
        "--min-ivp",
        type=int,
        default=40,
        dest="min_ivp",
        help="Minimum IV percentile threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        dest="output_csv",
        help="Output CSV file path",
    )

    # Scoring weights
    parser.add_argument("--prob-weight", type=float, default=0.0, dest="prob_weight")
    parser.add_argument("--roc-weight", type=float, default=0.40, dest="roc_weight")
    parser.add_argument("--convexity-weight", type=float, default=0.30, dest="convexity_weight")
    parser.add_argument("--slippage-weight", type=float, default=0.15, dest="slippage_weight")
    parser.add_argument("--liquidity-weight", type=float, default=0.15, dest="liquidity_weight")
    parser.add_argument("--ease-weight", type=float, default=0.0, dest="ease_weight")

    # Filtering thresholds
    parser.add_argument("--min-liquidity-score", type=float, default=0.1, dest="min_liquidity_score")
    parser.add_argument("--min-slippage-score", type=float, default=0.1, dest="min_slippage_score")

    args = parser.parse_args()

    return ScreenerConfig(
        tickers=[t.upper() for t in args.tickers],
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        min_delta=args.min_delta,
        max_delta=args.max_delta,
        max_width=args.max_width,
        min_roc=args.min_roc,
        min_ivp=args.min_ivp,
        output_csv=args.output_csv,
        prob_weight=args.prob_weight,
        roc_weight=args.roc_weight,
        convexity_weight=args.convexity_weight,
        slippage_weight=args.slippage_weight,
        liquidity_weight=args.liquidity_weight,
        ease_weight=args.ease_weight,
        min_liquidity_score=args.min_liquidity_score,
        min_slippage_score=args.min_slippage_score,
    )


def main() -> None:
    """Main entry point."""
    config = parse_args()

    print("Credit Spread Screener")
    print("-" * 40)
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"DTE Range: {config.min_dte} - {config.max_dte} days")
    print(f"Delta Range: {config.min_delta} - {config.max_delta}")
    print(f"Max Width: ${config.max_width}")
    print(f"Min ROC: {config.min_roc:.0%}")
    print(f"Min IVP: {config.min_ivp}%")
    print("-" * 40)

    # Run screener
    results = run_screener(config)

    # Print results
    print_results_table(results)

    # Save to CSV if requested
    if config.output_csv and not results.empty:
        results.to_csv(config.output_csv, index=False)
        print(f"\nResults saved to: {config.output_csv}")


if __name__ == "__main__":
    main()
