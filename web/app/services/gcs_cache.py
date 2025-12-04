"""
GCS cache module for Parquet score storage.

This module handles reading and writing daily scorer results to Google Cloud Storage
using Parquet format for efficient storage and retrieval.
"""

import io
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Optional

import pandas as pd
from google.cloud import storage
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)

# Configuration from environment variables
SCORE_CACHE_BUCKET = os.environ.get("SCORE_CACHE_BUCKET", "optchain_temp_date")
SCORE_CACHE_PREFIX = os.environ.get("SCORE_CACHE_PREFIX", "cache/")

# Valid symbol pattern (alphanumeric, 1-10 chars)
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,10}$")


@lru_cache(maxsize=1)
def _get_client() -> storage.Client:
    """Get cached GCS client using Application Default Credentials."""
    return storage.Client()


def _validate_symbol(symbol: str) -> str:
    """
    Validate and normalize symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Uppercase validated symbol

    Raises:
        ValueError: If symbol is invalid
    """
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol


def _get_cache_path(symbol: str) -> str:
    """
    Get the GCS path for a symbol's score cache.

    Args:
        symbol: Stock ticker symbol (e.g., 'SPY') - must be pre-validated

    Returns:
        GCS object path (without bucket)
    """
    return f"{SCORE_CACHE_PREFIX}daily_scores/{symbol}.parquet"


def read_scores(symbol: str) -> Optional[pd.DataFrame]:
    """
    Read cached scores for a symbol from GCS.

    Args:
        symbol: Stock ticker symbol

    Returns:
        DataFrame with columns [date, signal_raw, signal_0_1] or None if not found

    Raises:
        ValueError: If symbol format is invalid
    """
    symbol = _validate_symbol(symbol)

    try:
        client = _get_client()
        bucket = client.bucket(SCORE_CACHE_BUCKET)
        blob = bucket.blob(_get_cache_path(symbol))

        if not blob.exists():
            return None

        # Download to bytes and read with pandas
        data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(data))

        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df

    except NotFound:
        return None
    except Exception as e:
        logger.warning(f"Error reading cache for {symbol}: {e}")
        return None


def write_scores(symbol: str, df: pd.DataFrame) -> bool:
    """
    Write scores to GCS cache.

    Args:
        symbol: Stock ticker symbol
        df: DataFrame with columns [date, signal_raw, signal_0_1]

    Returns:
        True if successful, False otherwise

    Raises:
        ValueError: If symbol format is invalid
    """
    symbol = _validate_symbol(symbol)

    try:
        if df.empty:
            return True

        client = _get_client()
        bucket = client.bucket(SCORE_CACHE_BUCKET)
        blob = bucket.blob(_get_cache_path(symbol))

        # Convert to parquet bytes
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)

        # Upload to GCS
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        return True

    except Exception as e:
        logger.error(f"Error writing cache for {symbol}: {e}")
        return False


def get_latest_score(symbol: str) -> Optional[dict]:
    """
    Get the latest cached score for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with {date, signal_raw, signal_0_1} or None if no cache
    """
    df = read_scores(symbol)

    if df is None or df.empty:
        return None

    # Sort by date and get latest
    df = df.sort_values("date", ascending=False)
    latest = df.iloc[0]

    return {
        "date": latest["date"].to_pydatetime() if hasattr(latest["date"], "to_pydatetime") else latest["date"],
        "signal_raw": float(latest["signal_raw"]),
        "signal_0_1": float(latest["signal_0_1"]),
    }


def add_score(symbol: str, date: datetime, signal_raw: float, signal_0_1: float) -> bool:
    """
    Add a new score to the cache.

    Args:
        symbol: Stock ticker symbol
        date: Score date
        signal_raw: Raw signal value
        signal_0_1: Normalized 0-1 signal value (must be between 0 and 1)

    Returns:
        True if successful

    Raises:
        ValueError: If symbol is invalid or score_0_1 is out of bounds
    """
    # Validate inputs
    symbol = _validate_symbol(symbol)

    if not (0.0 <= signal_0_1 <= 1.0):
        raise ValueError(f"signal_0_1 must be between 0 and 1, got {signal_0_1}")

    # Read existing scores
    df = read_scores(symbol)

    # Create new row
    new_row = pd.DataFrame([{
        "date": date,
        "signal_raw": float(signal_raw),
        "signal_0_1": float(signal_0_1),
    }])

    if df is None or df.empty:
        df = new_row
    else:
        # Check if date already exists
        df["date"] = pd.to_datetime(df["date"])
        target_date = pd.to_datetime(date)
        date_mask = df["date"].dt.normalize() == target_date.normalize()
        if date_mask.any():
            # Update existing
            df.loc[date_mask, "signal_raw"] = float(signal_raw)
            df.loc[date_mask, "signal_0_1"] = float(signal_0_1)
        else:
            # Append new
            df = pd.concat([df, new_row], ignore_index=True)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return write_scores(symbol, df)


def clear_cache(symbol: str) -> bool:
    """
    Clear the cache for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        True if successful

    Raises:
        ValueError: If symbol format is invalid
    """
    symbol = _validate_symbol(symbol)

    try:
        client = _get_client()
        bucket = client.bucket(SCORE_CACHE_BUCKET)
        blob = bucket.blob(_get_cache_path(symbol))

        if blob.exists():
            blob.delete()

        return True

    except Exception as e:
        logger.error(f"Error clearing cache for {symbol}: {e}")
        return False


def has_score_for_date(symbol: str, date: datetime) -> bool:
    """
    Check if a score exists for a specific date.

    Args:
        symbol: Stock ticker symbol
        date: Date to check

    Returns:
        True if score exists for date
    """
    df = read_scores(symbol)

    if df is None or df.empty:
        return False

    df["date"] = pd.to_datetime(df["date"])
    target_date = pd.to_datetime(date).normalize()

    return target_date in df["date"].dt.normalize().values
