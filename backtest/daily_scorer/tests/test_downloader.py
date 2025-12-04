"""
Tests for the stock data downloader.
"""

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtest.daily_scorer.downloader import (
    download_stock_data,
    get_latest_trading_date,
)
from backtest.daily_scorer.exceptions import DownloadError


@pytest.fixture
def mock_yf_data():
    """Create mock yfinance data with named index."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "Open": [100.0 + i for i in range(10)],
        "High": [101.0 + i for i in range(10)],
        "Low": [99.0 + i for i in range(10)],
        "Close": [100.5 + i for i in range(10)],
        "Volume": [1000000 + i * 10000 for i in range(10)],
    }, index=dates)
    df.index.name = "Date"  # yfinance returns index named "Date"
    return df


@pytest.fixture
def mock_yf_module(mock_yf_data):
    """Create a mock yfinance module."""
    mock_yf = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_yf_data
    mock_yf.Ticker.return_value = mock_ticker
    return mock_yf, mock_ticker


class TestDownloadStockData:
    """Tests for download_stock_data function."""

    def test_download_returns_dataframe(self, mock_yf_data, mock_yf_module):
        """Should return DataFrame with expected columns."""
        mock_yf, mock_ticker = mock_yf_module

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            df = download_stock_data("SPY", days=30)

            assert isinstance(df, pd.DataFrame)
            assert "Date" in df.columns
            assert "Open" in df.columns
            assert "High" in df.columns
            assert "Low" in df.columns
            assert "Close" in df.columns
            assert "Volume" in df.columns

    def test_download_removes_timezone(self, mock_yf_data, mock_yf_module):
        """Should return timezone-naive dates."""
        mock_yf, mock_ticker = mock_yf_module

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            df = download_stock_data("SPY", days=30)

            assert df["Date"].dt.tz is None

    def test_download_sorts_ascending(self, mock_yf_data, mock_yf_module):
        """Should sort data by date ascending."""
        mock_yf, mock_ticker = mock_yf_module
        # Reverse the data
        reversed_data = mock_yf_data.iloc[::-1].copy()
        reversed_data.index.name = "Date"
        mock_ticker.history.return_value = reversed_data

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            df = download_stock_data("SPY", days=30)

            # Check dates are ascending
            dates = df["Date"].tolist()
            assert dates == sorted(dates)

    def test_download_with_custom_end_date(self, mock_yf_data, mock_yf_module):
        """Should use custom end date."""
        mock_yf, mock_ticker = mock_yf_module
        end_date = datetime(2024, 6, 15)

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            download_stock_data("SPY", days=30, end_date=end_date)

            # Verify history was called with correct date range
            call_args = mock_ticker.history.call_args
            assert "2024-06-16" in call_args.kwargs.get("end", "")

    def test_download_raises_on_empty_data(self, mock_yf_module):
        """Should raise DownloadError for empty data."""
        mock_yf, mock_ticker = mock_yf_module
        mock_ticker.history.return_value = pd.DataFrame()

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            with pytest.raises(DownloadError, match="No data returned"):
                download_stock_data("INVALID_SYMBOL", days=30)

    def test_download_raises_on_api_error(self, mock_yf_module):
        """Should raise DownloadError on API failure."""
        mock_yf, mock_ticker = mock_yf_module
        mock_ticker.history.side_effect = Exception("API error")

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            with pytest.raises(DownloadError, match="Failed to download"):
                download_stock_data("SPY", days=30)

    def test_download_raises_on_missing_columns(self, mock_yf_module):
        """Should raise DownloadError for missing required columns."""
        mock_yf, mock_ticker = mock_yf_module
        incomplete_data = pd.DataFrame({
            "Open": [100.0],
            "High": [101.0],
            # Missing Low, Close, Volume
        }, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
        incomplete_data.index.name = "Date"
        mock_ticker.history.return_value = incomplete_data

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            with pytest.raises(DownloadError, match="missing required columns"):
                download_stock_data("SPY", days=30)

    def test_download_raises_on_missing_yfinance(self):
        """Should raise DownloadError if yfinance not installed."""
        # Remove yfinance from modules to simulate it not being installed
        original_modules = sys.modules.copy()

        # Remove yfinance if present
        if "yfinance" in sys.modules:
            del sys.modules["yfinance"]

        # Block the import
        with patch.dict(sys.modules, {"yfinance": None}):
            with pytest.raises(DownloadError, match="yfinance is required"):
                download_stock_data("SPY", days=30)


class TestGetLatestTradingDate:
    """Tests for get_latest_trading_date function."""

    def test_returns_latest_date(self, mock_yf_data, mock_yf_module):
        """Should return the most recent trading date."""
        mock_yf, mock_ticker = mock_yf_module

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = get_latest_trading_date("SPY")

            assert isinstance(result, datetime)

    def test_returns_none_on_error(self, mock_yf_module):
        """Should return None if download fails."""
        mock_yf, mock_ticker = mock_yf_module
        mock_ticker.history.return_value = pd.DataFrame()

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = get_latest_trading_date("INVALID")

            assert result is None
