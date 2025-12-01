#!/usr/bin/env python3
"""
LEAPS Ranker - Long-term Equity Anticipation Securities Ranking Tool

A CLI tool to automatically rank and find the best LEAPS (long-dated call options)
based on ROI potential and ease of reaching the target price.

Features:
- Pull options data via OpenBB Platform
- Filter to LEAPS based on days to expiration
- Compute ROI, ease scores, and combined rankings
- Support for High-Probability and High-Convexity modes
- Configuration-driven via YAML file

Usage:
    python leaps_ranker.py QQQ --config config/leaps_ranker.yaml
    python leaps_ranker.py NVDA --mode high_convexity --target-pct 0.75
    python leaps_ranker.py SPY --min-dte 180 --no-longest-only --top-n 30
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Module-level logger
logger = logging.getLogger(__name__)

# Global flag for OpenBB availability
_USE_OPENBB = False
_obb = None

try:
    from openbb import obb
    _obb = obb
    _USE_OPENBB = True
except ImportError:
    logger.warning("OpenBB not installed, will use yfinance directly")
    _USE_OPENBB = False


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_scoring_weights(config: Dict[str, Any], mode: str) -> Dict[str, float]:
    """
    Get scoring weights for the specified mode from configuration.

    Args:
        config: Configuration dictionary loaded from YAML.
        mode: Scoring mode name (e.g., 'high_prob', 'high_convexity').

    Returns:
        Dictionary with 'ease_weight' and 'roi_weight' keys.

    Raises:
        ValueError: If the mode is not found in configuration.
    """
    scoring_modes = config.get("scoring_modes", {})

    if mode not in scoring_modes:
        available_modes = list(scoring_modes.keys())
        raise ValueError(
            f"Unknown scoring mode: '{mode}'. Available modes: {available_modes}"
        )

    mode_config = scoring_modes[mode]
    ease_weight = mode_config.get("ease_weight", 0.5)
    roi_weight = mode_config.get("roi_weight", 0.5)

    # Validate weights sum to 1.0
    total = ease_weight + roi_weight
    if abs(total - 1.0) > 0.01:
        logger.warning(
            f"Scoring weights sum to {total:.2f}, not 1.0. Normalizing weights."
        )
        ease_weight /= total
        roi_weight /= total

    return {"ease_weight": ease_weight, "roi_weight": roi_weight}


# =============================================================================
# DATA FETCHING
# =============================================================================


def _get_yfinance_options_chain(
    symbol: str,
    option_type: str = "call",
) -> Tuple[pd.DataFrame, float]:
    """
    Fetch options chain directly from yfinance.

    Args:
        symbol: Underlying ticker symbol.
        option_type: Type of options to fetch ('call' or 'put').

    Returns:
        Tuple of (DataFrame with options data, underlying price).

    Raises:
        RuntimeError: If yfinance fails to fetch data.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance required. Install with: pip install yfinance")

    logger.info(f"Fetching options via yfinance for {symbol}")

    try:
        ticker = yf.Ticker(symbol.upper())

        # Get underlying price
        info = ticker.info
        underlying_price = info.get("regularMarketPrice") or info.get(
            "previousClose", 0
        )

        # Get all expiration dates
        expirations = ticker.options
        if not expirations:
            raise ValueError(f"No options expirations found for {symbol}")

        logger.info(f"Found {len(expirations)} expiration dates for {symbol}")

        # Fetch all chains and combine
        all_chains = []
        for exp in expirations:
            try:
                opt_chain = ticker.option_chain(exp)
                if option_type == "call":
                    chain = opt_chain.calls.copy()
                else:
                    chain = opt_chain.puts.copy()

                chain["expiration"] = exp
                all_chains.append(chain)
            except Exception as e:
                logger.debug(f"Failed to fetch chain for {exp}: {e}")
                continue

        if not all_chains:
            raise ValueError(f"No {option_type} options data found for {symbol}")

        df = pd.concat(all_chains, ignore_index=True)

        # Calculate DTE
        today = pd.Timestamp.now().normalize()
        df["expiration_dt"] = pd.to_datetime(df["expiration"])
        df["dte"] = (df["expiration_dt"] - today).dt.days

        # Standardize column names from yfinance format
        col_mapping = {
            "contractSymbol": "contract_symbol",
            "lastPrice": "last_trade_price",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_volatility",
        }
        df = df.rename(columns=col_mapping)

        # Add mark as mid price
        if "bid" in df.columns and "ask" in df.columns:
            df["mark"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2

        # Add underlying price
        df["underlying_price"] = underlying_price

        return df, underlying_price

    except Exception as e:
        raise RuntimeError(f"yfinance failed for {symbol}: {e}") from e


def fetch_options_chain(
    symbol: str,
    provider: str = "cboe",
    option_type: str = "call",
) -> Tuple[pd.DataFrame, float]:
    """
    Fetch options chain data for a given symbol.

    Tries OpenBB first, then falls back to direct yfinance.

    Args:
        symbol: Underlying ticker symbol (e.g., 'QQQ', 'NVDA').
        provider: OpenBB options data provider (e.g., 'cboe', 'yfinance').
        option_type: Type of options to fetch ('call' or 'put').

    Returns:
        Tuple of (DataFrame with options chain data, underlying price).

    Raises:
        ValueError: If no options data is found for the symbol.
        RuntimeError: If all API calls fail.
    """
    logger.info(f"Fetching options chain for {symbol}")

    underlying_price = 0.0
    df = None

    # Try OpenBB first
    if _USE_OPENBB and _obb is not None:
        for prov in [provider, "intrinio", "yfinance"]:
            try:
                logger.debug(f"Trying OpenBB provider: {prov}")
                result = _obb.derivatives.options.chains(
                    symbol=symbol.upper(),
                    provider=prov,
                )
                df = result.to_df()

                if not df.empty:
                    # Extract underlying price if available
                    if "underlying_price" in df.columns:
                        underlying_price = float(df["underlying_price"].iloc[0])

                    logger.info(f"OpenBB {prov} returned {len(df)} contracts")
                    break

            except Exception as e:
                logger.debug(f"OpenBB {prov} failed: {e}")
                continue

    # Fallback to direct yfinance
    if df is None or df.empty:
        logger.info("Falling back to direct yfinance")
        df, underlying_price = _get_yfinance_options_chain(symbol, option_type)

    if df.empty:
        raise ValueError(f"No options data found for {symbol}")

    # Filter by option type
    if "option_type" in df.columns:
        df = df[df["option_type"] == option_type].copy()
    elif "type" in df.columns:
        df = df[df["type"] == option_type].copy()

    if df.empty:
        raise ValueError(f"No {option_type} options found for {symbol}")

    # Standardize column names
    df = _standardize_columns(df)

    logger.info(f"Retrieved {len(df)} {option_type} options for {symbol}")
    return df, underlying_price


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names from various OpenBB providers.

    Args:
        df: Raw options DataFrame from OpenBB.

    Returns:
        DataFrame with standardized column names.
    """
    # Column mapping for common variations
    column_mapping = {
        # Contract identifier
        "contract_symbol": "contract_symbol",
        "contractSymbol": "contract_symbol",
        "symbol": "contract_symbol",
        # Expiration
        "expiration": "expiration",
        "expirationDate": "expiration",
        "expiry": "expiration",
        # Days to expiration
        "dte": "dte",
        "days_to_expiration": "dte",
        "daysToExpiration": "dte",
        # Strike
        "strike": "strike",
        "strikePrice": "strike",
        # Underlying price
        "underlying_price": "underlying_price",
        "underlyingPrice": "underlying_price",
        "spot_price": "underlying_price",
        # Premium/price columns
        "mark": "mark",
        "mid": "mark",
        "midPrice": "mark",
        "last_trade_price": "last_trade_price",
        "lastPrice": "last_trade_price",
        "last": "last_trade_price",
        "theoretical_price": "theoretical_price",
        "theoreticalPrice": "theoretical_price",
        "close": "close",
        # Bid/Ask
        "bid": "bid",
        "bidPrice": "bid",
        "ask": "ask",
        "askPrice": "ask",
        # Greeks
        "delta": "delta",
        "gamma": "gamma",
        "theta": "theta",
        "vega": "vega",
        "rho": "rho",
        # Implied Volatility
        "implied_volatility": "implied_volatility",
        "impliedVolatility": "implied_volatility",
        "iv": "implied_volatility",
        # Volume/Interest
        "open_interest": "open_interest",
        "openInterest": "open_interest",
        "volume": "volume",
    }

    # Rename columns that exist
    rename_map = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and old_name != new_name:
            if new_name not in df.columns:
                rename_map[old_name] = new_name

    df = df.rename(columns=rename_map)
    return df


def get_underlying_price(symbol: str, provider: str = "cboe") -> float:
    """
    Get the current underlying price for a symbol.

    Tries multiple methods to get the price:
    1. OpenBB equity.price.quote (if available)
    2. Direct yfinance Ticker.info

    Args:
        symbol: Underlying ticker symbol.
        provider: Data provider to use (hint for ordering).

    Returns:
        Current underlying price as a float.

    Raises:
        ValueError: If unable to determine the underlying price.
    """
    logger.info(f"Fetching underlying price for {symbol}")

    # Try various price columns
    price_columns = [
        "last_price",
        "close",
        "price",
        "regularMarketPrice",
        "previousClose",
        "last",
        "adj_close",
    ]

    # Method 1: Try OpenBB equity.price.quote with various providers
    if _USE_OPENBB and _obb is not None:
        quote_providers = ["fmp", "intrinio", "yfinance"]
        for qp in quote_providers:
            try:
                result = _obb.equity.price.quote(symbol=symbol.upper(), provider=qp)
                df = result.to_df()

                for col in price_columns:
                    if col in df.columns:
                        price = df[col].iloc[0]
                        if price is not None and not pd.isna(price) and price > 0:
                            logger.info(
                                f"Underlying price for {symbol}: ${price:.2f} "
                                f"(via quote/{qp})"
                            )
                            return float(price)
            except Exception as e:
                logger.debug(f"Quote via {qp} failed: {e}")

    # Method 2: Direct yfinance
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        for key in ["regularMarketPrice", "previousClose", "ask", "bid"]:
            price = info.get(key)
            if price is not None and price > 0:
                logger.info(
                    f"Underlying price for {symbol}: ${price:.2f} (via yfinance.{key})"
                )
                return float(price)

    except Exception as e:
        logger.debug(f"Direct yfinance failed: {e}")

    raise ValueError(f"Could not determine underlying price for {symbol}")


# =============================================================================
# LEAPS FILTERING
# =============================================================================


def filter_to_leaps(
    df: pd.DataFrame,
    min_dte: int = 365,
    longest_only: bool = True,
    max_target_distance_pct: float = 0.5,
    underlying_price: float = 0.0,
    target_price: float = 0.0,
) -> pd.DataFrame:
    """
    Filter options DataFrame to LEAPS contracts based on DTE and strike criteria.

    Args:
        df: Options DataFrame with 'dte' column.
        min_dte: Minimum days to expiration to qualify as a LEAP.
        longest_only: If True, keep only contracts at the longest available expiry.
        max_target_distance_pct: Maximum distance from target price to include.
        underlying_price: Current underlying price (for distance calculation).
        target_price: Target underlying price (for distance calculation).

    Returns:
        Filtered DataFrame containing only LEAPS contracts.

    Raises:
        ValueError: If no contracts satisfy the LEAPS filter criteria.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Convert DTE to numeric
    df = df.copy()
    df["dte"] = pd.to_numeric(df["dte"], errors="coerce")

    # Filter by minimum DTE
    df = df[df["dte"] >= min_dte].copy()

    if df.empty:
        raise ValueError(
            f"No contracts found with DTE >= {min_dte}. "
            "Consider lowering --min-dte or checking available expirations."
        )

    # Filter to longest expiry only if requested
    if longest_only:
        max_dte = df["dte"].max()
        df = df[df["dte"] == max_dte].copy()
        logger.info(f"Filtered to longest expiry: DTE = {max_dte}")

    # Filter by distance from target
    if target_price > 0 and max_target_distance_pct > 0:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        # Distance from target as a fraction of target price
        df["distance_from_target"] = (df["strike"] - target_price).abs() / target_price
        df = df[df["distance_from_target"] <= max_target_distance_pct].copy()
        df = df.drop(columns=["distance_from_target"])

        if df.empty:
            raise ValueError(
                f"No contracts within {max_target_distance_pct:.0%} of target price "
                f"(${target_price:.2f}). Consider increasing max_target_distance_pct."
            )

    logger.info(f"Filtered to {len(df)} LEAPS contracts")
    return df


# =============================================================================
# PREMIUM SELECTION
# =============================================================================


def select_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the premium (cost) for each option contract.

    Selects the first valid, positive value from a prioritized list of price columns.
    Falls back to mid-price calculation if primary columns are unavailable.

    Args:
        df: Options DataFrame with price columns.

    Returns:
        DataFrame with 'premium' column added.
    """
    df = df.copy()

    # Priority order for premium selection
    premium_columns = ["mark", "last_trade_price", "theoretical_price", "close"]

    def get_premium(row: pd.Series) -> float:
        """Extract premium from row using priority order."""
        # Try primary columns
        for col in premium_columns:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and val > 0:
                    return float(val)

        # Try mid-price from bid/ask
        bid = row.get("bid", np.nan)
        ask = row.get("ask", np.nan)

        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
            return float((bid + ask) / 2)
        elif pd.notna(bid) and bid > 0:
            return float(bid)
        elif pd.notna(ask) and ask > 0:
            return float(ask)

        return np.nan

    df["premium"] = df.apply(get_premium, axis=1)

    # Drop contracts with invalid premiums
    initial_count = len(df)
    df = df[df["premium"].notna() & (df["premium"] > 0)].copy()
    dropped_count = initial_count - len(df)

    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} contracts with invalid premiums")

    if df.empty:
        raise ValueError(
            "All contracts have invalid or missing premiums. "
            "Check data quality from the provider."
        )

    return df


# =============================================================================
# METRIC COMPUTATION
# =============================================================================


def compute_metrics(
    df: pd.DataFrame,
    underlying_price: float,
    target_pct: float,
    contract_size: int = 100,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute ROI, ease scores, and final ranking score for each contract.

    Args:
        df: Options DataFrame with 'strike' and 'premium' columns.
        underlying_price: Current price of the underlying asset.
        target_pct: Target percentage move (e.g., 0.5 for +50%).
        contract_size: Options contract multiplier (typically 100).
        weights: Dictionary with 'ease_weight' and 'roi_weight' for scoring.

    Returns:
        DataFrame with computed metrics and scores.
    """
    if weights is None:
        weights = {"ease_weight": 0.5, "roi_weight": 0.5}

    df = df.copy()

    # Set underlying and target prices
    target_price = underlying_price * (1.0 + target_pct)
    df["current_underlying_price"] = underlying_price
    df["target_price"] = target_price

    # Ensure strike is numeric
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    # Compute payoff at target
    df["intrinsic_at_target"] = np.maximum(target_price - df["strike"], 0)
    df["payoff_target"] = df["intrinsic_at_target"] * contract_size

    # Compute cost per contract
    df["cost"] = df["premium"] * contract_size

    # Compute ROI at target (as percentage)
    df["roi_target"] = ((df["payoff_target"] - df["cost"]) / df["cost"]) * 100

    # Handle infinite ROI values
    df["roi_target"] = df["roi_target"].replace([np.inf, -np.inf], np.nan)
    df["roi_target"] = df["roi_target"].fillna(df["roi_target"].min())

    # Compute ease score based on strike position relative to midpoint
    df = _compute_ease_score(df, underlying_price, target_price)

    # Normalize ROI to 0-1 scale
    df = _normalize_roi_score(df)

    # Compute final combined score
    ease_weight = weights["ease_weight"]
    roi_weight = weights["roi_weight"]
    df["score"] = (ease_weight * df["ease_score"]) + (roi_weight * df["roi_score"])

    # Ensure numeric columns for volume and open interest
    for col in ["open_interest", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Sort by final score descending
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


def _compute_ease_score(
    df: pd.DataFrame,
    underlying_price: float,
    target_price: float,
) -> pd.DataFrame:
    """
    Compute ease score based on how likely the option is to be profitable.

    Uses three anchor points with linear interpolation:
    - Deep ITM (strike << underlying): score = 1.00
    - ATM (strike = underlying): score = 0.90
    - Target price (strike = target): score = 0.85

    For strikes beyond target, deduct 0.03 for every 1% farther from target.

    Args:
        df: Options DataFrame with 'strike' column.
        underlying_price: Current underlying price.
        target_price: Target underlying price.

    Returns:
        DataFrame with 'ease_score' column added.
    """
    df = df.copy()

    # Anchor point scores
    DEEP_ITM_SCORE = 1.00
    ATM_SCORE = 0.90
    TARGET_SCORE = 0.85
    BEYOND_TARGET_PENALTY_PER_PCT = 0.03

    # Calculate position of each strike relative to underlying and target
    strikes = df["strike"]

    # Initialize ease_score column
    df["ease_score"] = 0.0

    # Case 1: Deep ITM (strike <= underlying) - linear interpolation from 1.0 to 0.9
    # We consider "deep ITM" as strikes that are 10% or more below underlying
    deep_itm_threshold = underlying_price * 0.90

    # Strikes at or below deep ITM threshold get 1.0
    mask_deep_itm = strikes <= deep_itm_threshold
    df.loc[mask_deep_itm, "ease_score"] = DEEP_ITM_SCORE

    # Strikes between deep ITM and ATM: linear interpolation 1.0 -> 0.9
    mask_itm_to_atm = (strikes > deep_itm_threshold) & (strikes <= underlying_price)
    if mask_itm_to_atm.any():
        # Distance from deep ITM threshold to underlying
        itm_range = underlying_price - deep_itm_threshold
        if itm_range > 0:
            # Position ratio: 0 at deep ITM threshold, 1 at underlying
            position_in_itm = (strikes[mask_itm_to_atm] - deep_itm_threshold) / itm_range
            # Linear interpolation: 1.0 -> 0.9
            df.loc[mask_itm_to_atm, "ease_score"] = DEEP_ITM_SCORE - (DEEP_ITM_SCORE - ATM_SCORE) * position_in_itm
        else:
            df.loc[mask_itm_to_atm, "ease_score"] = ATM_SCORE

    # Case 2: ATM to Target - linear interpolation from 0.9 to 0.85
    mask_atm_to_target = (strikes > underlying_price) & (strikes <= target_price)
    if mask_atm_to_target.any():
        # Distance from underlying to target
        otm_range = target_price - underlying_price
        if otm_range > 0:
            # Position ratio: 0 at underlying, 1 at target
            position_in_otm = (strikes[mask_atm_to_target] - underlying_price) / otm_range
            # Linear interpolation: 0.9 -> 0.85
            df.loc[mask_atm_to_target, "ease_score"] = ATM_SCORE - (ATM_SCORE - TARGET_SCORE) * position_in_otm
        else:
            df.loc[mask_atm_to_target, "ease_score"] = TARGET_SCORE

    # Case 3: Beyond target - start at 0.85 and deduct 0.03 per 1% beyond target
    mask_beyond_target = strikes > target_price
    if mask_beyond_target.any():
        # Calculate how many percent beyond target each strike is
        pct_beyond_target = ((strikes[mask_beyond_target] - target_price) / target_price) * 100
        # Deduct 0.03 for each 1% beyond target
        penalty = pct_beyond_target * BEYOND_TARGET_PENALTY_PER_PCT
        df.loc[mask_beyond_target, "ease_score"] = TARGET_SCORE - penalty

    # Clamp to [0, 1]
    df["ease_score"] = df["ease_score"].clip(0, 1)

    return df


def _normalize_roi_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ROI into a 0-1 score.

    Uses 50% ROI as the floor (score = 0) and normalizes linearly to
    the highest ROI in the dataset (score = 1).

    - ROI <= 50%: score = 0.0
    - ROI = max ROI: score = 1.0
    - ROI between 50% and max: linearly interpolated

    Args:
        df: Options DataFrame with 'roi_target' column.

    Returns:
        DataFrame with 'roi_score' column added.
    """
    df = df.copy()

    roi = df["roi_target"]

    # Floor at 50% ROI
    ROI_FLOOR = 50.0
    roi_max = roi.max()

    if roi_max <= ROI_FLOOR:
        # All ROIs are at or below floor, give all 0
        df["roi_score"] = 0.0
    else:
        # Normalize: 50% -> 0, max -> 1
        df["roi_score"] = (roi - ROI_FLOOR) / (roi_max - ROI_FLOOR)

    # Clamp to [0, 1]
    df["roi_score"] = df["roi_score"].clip(0, 1)

    return df


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


def format_output(
    df: pd.DataFrame,
    top_n: int = 20,
    output_columns: Optional[list] = None,
    price_decimals: int = 2,
    pct_decimals: int = 4,
) -> pd.DataFrame:
    """
    Format the DataFrame for output display.

    Args:
        df: DataFrame with computed metrics.
        top_n: Number of top contracts to return.
        output_columns: List of columns to include in output.
        price_decimals: Decimal places for price columns.
        pct_decimals: Decimal places for percentage/ratio columns.

    Returns:
        Formatted DataFrame with top N contracts.
    """
    # Default output columns
    if output_columns is None:
        output_columns = [
            "contract_symbol",
            "expiration",
            "dte",
            "strike",
            "current_underlying_price",
            "target_price",
            "premium",
            "cost",
            "payoff_target",
            "roi_target",
            "ease_score",
            "roi_score",
            "score",
            "delta",
            "implied_volatility",
            "open_interest",
            "volume",
        ]

    # Select top N
    result = df.head(top_n).copy()

    # Filter to available columns
    available_columns = [col for col in output_columns if col in result.columns]
    result = result[available_columns]

    # Format decimal places for specific columns
    price_cols = [
        "strike",
        "current_underlying_price",
        "target_price",
        "premium",
        "cost",
        "payoff_target",
        "mark",
    ]
    pct_cols = [
        "roi_target",
        "ease_score",
        "roi_score",
        "score",
        "delta",
        "implied_volatility",
    ]

    for col in price_cols:
        if col in result.columns:
            result[col] = result[col].round(price_decimals)

    for col in pct_cols:
        if col in result.columns:
            result[col] = result[col].round(pct_decimals)

    return result


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def rank_leaps(
    symbol: str,
    provider: str = "cboe",
    mode: str = "high_prob",
    target_pct: float = 0.5,
    min_dte: int = 365,
    longest_only: bool = True,
    top_n: int = 20,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Main workflow to rank LEAPS options for a given symbol.

    Args:
        symbol: Underlying ticker symbol.
        provider: OpenBB options data provider.
        mode: Scoring mode ('high_prob' or 'high_convexity').
        target_pct: Target percentage move in underlying.
        min_dte: Minimum days to expiration for LEAPS.
        longest_only: If True, only keep contracts at longest expiry.
        top_n: Number of top contracts to return.
        config: Configuration dictionary (optional).

    Returns:
        DataFrame with ranked LEAPS contracts.
    """
    # Use default config values if not provided
    if config is None:
        config = {
            "scoring_modes": {
                "high_prob": {"ease_weight": 0.75, "roi_weight": 0.25},
                "high_convexity": {"ease_weight": 0.25, "roi_weight": 0.75},
            },
            "filtering": {"max_target_distance_pct": 0.5},
            "contract": {"contract_size": 100},
            "display": {
                "output_columns": None,
                "price_decimals": 2,
                "pct_decimals": 4,
            },
        }

    # Get scoring weights
    weights = get_scoring_weights(config, mode)
    logger.info(f"Using mode '{mode}' with weights: {weights}")

    # Get filtering parameters
    filtering_config = config.get("filtering", {})
    max_target_distance_pct = filtering_config.get("max_target_distance_pct", 0.5)

    # Get contract parameters
    contract_config = config.get("contract", {})
    contract_size = contract_config.get("contract_size", 100)

    # Step 1: Fetch options chain (also returns underlying price if available)
    df, chain_underlying_price = fetch_options_chain(
        symbol, provider, option_type="call"
    )

    # Step 2: Get underlying price (prefer chain price, fallback to direct fetch)
    if chain_underlying_price and chain_underlying_price > 0:
        underlying_price = chain_underlying_price
        logger.info(f"Using underlying price from options chain: ${underlying_price:.2f}")
    else:
        underlying_price = get_underlying_price(symbol, provider)

    # Calculate target price for filtering
    target_price = underlying_price * (1.0 + target_pct)
    logger.info(
        f"Underlying: ${underlying_price:.2f}, "
        f"Target (+{target_pct:.0%}): ${target_price:.2f}"
    )

    # Step 3: Filter to LEAPS
    df = filter_to_leaps(
        df,
        min_dte=min_dte,
        longest_only=longest_only,
        max_target_distance_pct=max_target_distance_pct,
        underlying_price=underlying_price,
        target_price=target_price,
    )

    # Step 4: Select premium
    df = select_premium(df)

    # Step 5: Compute metrics
    df = compute_metrics(
        df,
        underlying_price=underlying_price,
        target_pct=target_pct,
        contract_size=contract_size,
        weights=weights,
    )

    # Step 6: Format output
    display_config = config.get("display", {})
    df = format_output(
        df,
        top_n=top_n,
        output_columns=display_config.get("output_columns"),
        price_decimals=display_config.get("price_decimals", 2),
        pct_decimals=display_config.get("pct_decimals", 4),
    )

    return df


# =============================================================================
# CLI INTERFACE
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LEAPS Ranker - Find and rank the best long-dated call options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s QQQ --config config/leaps_ranker.yaml
  %(prog)s NVDA --mode high_convexity --target-pct 0.75
  %(prog)s SPY --min-dte 180 --no-longest-only --top-n 30
  %(prog)s AAPL --provider yfinance --mode high_prob

Modes:
  high_prob       High-Probability mode - favors ease/delta (default)
  high_convexity  High-Convexity mode - favors ROI potential
        """,
    )

    # Required positional argument
    parser.add_argument(
        "symbol",
        type=str,
        help="Underlying ticker symbol (e.g., QQQ, NVDA, SPY)",
    )

    # Required config file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Optional arguments
    parser.add_argument(
        "--provider",
        type=str,
        default="cboe",
        help="OpenBB options data provider (default: cboe)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["high_prob", "high_convexity"],
        default=None,
        help="Scoring mode: high_prob (favor ease) or high_convexity (favor ROI)",
    )

    parser.add_argument(
        "--target-pct",
        type=float,
        default=None,
        help="Target upside as fraction (e.g., 0.5 for +50%% move)",
    )

    parser.add_argument(
        "--min-dte",
        type=int,
        default=None,
        help="Minimum days to expiration for LEAPS filter",
    )

    parser.add_argument(
        "--no-longest-only",
        action="store_true",
        help="Keep all contracts with DTE >= min-dte (not just longest expiry)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top contracts to display",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    return parser.parse_args()


def setup_logging(config: Dict[str, Any], verbose: bool = False) -> None:
    """
    Configure logging based on configuration and CLI flags.

    Args:
        config: Configuration dictionary.
        verbose: If True, override config to use DEBUG level.
    """
    logging_config = config.get("logging", {})

    level_str = logging_config.get("level", "INFO")
    if verbose:
        level_str = "DEBUG"

    level = getattr(logging, level_str.upper(), logging.INFO)
    log_format = logging_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def main() -> int:
    """
    Main entry point for the LEAPS Ranker CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_arguments()

    try:
        # Load configuration
        config = load_config(args.config)

        # Setup logging
        setup_logging(config, args.verbose)

        # Get default values from config, override with CLI args
        filtering_config = config.get("filtering", {})
        target_config = config.get("target", {})
        display_config = config.get("display", {})
        provider_config = config.get("provider", {})

        # Determine final parameter values (CLI overrides config)
        provider = args.provider or provider_config.get("default", "cboe")
        mode = args.mode or config.get("default_mode", "high_prob")
        target_pct = (
            args.target_pct
            if args.target_pct is not None
            else target_config.get("default_target_pct", 0.5)
        )
        min_dte = (
            args.min_dte
            if args.min_dte is not None
            else filtering_config.get("min_dte", 365)
        )
        longest_only = not args.no_longest_only and filtering_config.get(
            "longest_only", True
        )
        top_n = (
            args.top_n
            if args.top_n is not None
            else display_config.get("top_n", 20)
        )

        # Run the ranking
        logger.info(f"Ranking LEAPS for {args.symbol.upper()}")
        logger.info(
            f"Parameters: provider={provider}, mode={mode}, "
            f"target_pct={target_pct}, min_dte={min_dte}, "
            f"longest_only={longest_only}, top_n={top_n}"
        )

        result_df = rank_leaps(
            symbol=args.symbol,
            provider=provider,
            mode=mode,
            target_pct=target_pct,
            min_dte=min_dte,
            longest_only=longest_only,
            top_n=top_n,
            config=config,
        )

        # Print results
        print(f"\n{'='*80}")
        print(f"LEAPS Ranking for {args.symbol.upper()}")
        print(f"Mode: {mode} | Target: +{target_pct:.0%} | Min DTE: {min_dte}")
        print(f"{'='*80}\n")

        # Configure pandas display for wide output
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        print(result_df.to_string(index=False))

        print(f"\n{'='*80}")
        print(f"Displayed top {len(result_df)} contracts out of available LEAPS")
        print(f"{'='*80}\n")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    sys.exit(main())
