"""
Stock data downloader using yfinance.

Downloads daily OHLCV data for a given symbol.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from backtest.daily_scorer.exceptions import DownloadError


def download_stock_data(
    symbol: str,
    days: int = 365,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Download daily stock data for a symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "SPY", "AAPL").
        days: Number of days of historical data to download (default 365).
        end_date: End date for data. If None, uses today.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
        Date column is datetime type, sorted ascending.

    Raises:
        DownloadError: If download fails or returns no data.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise DownloadError(
            "yfinance is required for downloading stock data. "
            "Install with: pip install yfinance"
        ) from e

    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=days)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
    except Exception as e:
        raise DownloadError(f"Failed to download data for {symbol}: {e}") from e

    if df.empty:
        raise DownloadError(
            f"No data returned for {symbol} from {start_date.date()} to {end_date.date()}"
        )

    # Reset index to get Date as column
    df = df.reset_index()

    # Rename columns to match expected format
    df = df.rename(columns={
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    })

    # Select only required columns
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise DownloadError(
            f"Downloaded data missing required columns: {missing_cols}"
        )

    df = df[required_cols].copy()

    # Ensure Date is datetime and timezone-naive
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # Sort by date ascending
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def get_latest_trading_date(symbol: str) -> Optional[datetime]:
    """
    Get the most recent trading date for a symbol.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Most recent trading date, or None if unable to determine.
    """
    try:
        df = download_stock_data(symbol, days=7)
        if not df.empty:
            return df["Date"].iloc[-1].to_pydatetime()
    except DownloadError:
        pass
    return None
