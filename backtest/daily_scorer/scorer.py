"""
Main DailyScorer class.

Orchestrates downloading, indicator computation, scoring, and caching.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from backtest.daily_scorer.cache import ScoreCache
from backtest.daily_scorer.downloader import download_stock_data
from backtest.daily_scorer.exceptions import (
    ConfigurationError,
    DailyScorerError,
    InsufficientDataError,
)
from backtest.engine import run_scoring
from backtest.indicators.calculations import add_all_indicators


@dataclass
class ScoreResult:
    """Result from scoring operation."""

    date: datetime
    signal_raw: float
    signal_0_1: float
    symbol: str
    is_cached: bool = False

    def __str__(self) -> str:
        status = "cached" if self.is_cached else "new"
        return (
            f"ScoreResult({self.symbol} @ {self.date.strftime('%Y-%m-%d')}: "
            f"raw={self.signal_raw:.4f}, normalized={self.signal_0_1:.4f} [{status}])"
        )


class DailyScorer:
    """
    Daily stock scorer with caching.

    Downloads stock data, computes technical indicators, runs scoring engine,
    and caches results to avoid recomputation.

    Example:
        scorer = DailyScorer(
            symbol="SPY",
            strategy_files=["strategy.yaml"],
            cache_dir="./cache",
        )

        # Get the latest score (uses cache if available)
        result = scorer.get_latest_score()

        # Force refresh - download new data and score any unscored days
        result = scorer.refresh()
    """

    # Minimum days of data needed for indicator calculation
    MIN_DATA_DAYS = 250

    def __init__(
        self,
        symbol: str,
        strategy_files: Union[str, List[str]],
        cache_dir: str = "./cache",
        normalization: str = "zscore",
        lookback_days: int = 365,
    ):
        """
        Initialize daily scorer.

        Args:
            symbol: Stock ticker symbol (e.g., "SPY").
            strategy_files: Path(s) to strategy YAML file(s).
            cache_dir: Directory for cache files.
            normalization: Normalization method for scores ("none", "minmax", "zscore").
            lookback_days: Number of days of historical data to download.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        self._symbol = symbol.upper()
        self._strategy_files = (
            [strategy_files] if isinstance(strategy_files, str) else strategy_files
        )
        self._cache_dir = Path(cache_dir)
        self._normalization = normalization
        self._lookback_days = lookback_days

        # Validate strategy files exist
        for f in self._strategy_files:
            if not Path(f).exists():
                raise ConfigurationError(f"Strategy file not found: {f}")

        # Initialize cache
        cache_path = self._cache_dir / f"{self._symbol}_scores.csv"
        self._cache = ScoreCache(str(cache_path))

    @property
    def symbol(self) -> str:
        """Return the stock symbol."""
        return self._symbol

    @property
    def cache(self) -> ScoreCache:
        """Return the score cache."""
        return self._cache

    def _download_data(self) -> pd.DataFrame:
        """Download stock data."""
        return download_stock_data(
            symbol=self._symbol,
            days=self._lookback_days,
        )

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators on the data."""
        if len(df) < self.MIN_DATA_DAYS:
            raise InsufficientDataError(
                f"Need at least {self.MIN_DATA_DAYS} days of data for indicators, "
                f"got {len(df)}"
            )
        return add_all_indicators(df)

    def _run_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the scoring engine on indicator data."""
        result = run_scoring(
            data_file=df,
            strategy_files=self._strategy_files,
            normalization=self._normalization,
        )
        return result.scores_df

    def _filter_unscored_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to only include dates not yet scored.

        Args:
            df: DataFrame with Date column.

        Returns:
            Filtered DataFrame with only unscored dates.
        """
        scored_dates = self._cache.get_scored_dates()
        if not scored_dates:
            return df

        # Convert dates for comparison
        def normalize_date(d: datetime) -> datetime:
            return datetime(d.year, d.month, d.day)

        scored_normalized = {normalize_date(d) for d in scored_dates}

        mask = df["Date"].apply(
            lambda d: normalize_date(d.to_pydatetime()) not in scored_normalized
        )
        return df[mask].copy()

    def _has_valid_close(self, df: pd.DataFrame) -> pd.Series:
        """
        Check which rows have valid (non-null) Close prices.

        Args:
            df: DataFrame with Close column.

        Returns:
            Boolean Series indicating valid rows.
        """
        return df["Close"].notna()

    def refresh(self) -> Optional[ScoreResult]:
        """
        Refresh scores by downloading new data and scoring unscored days.

        Downloads latest data, computes indicators, runs scoring on any
        dates not already in cache, and returns the latest score.

        Returns:
            ScoreResult for the most recent date, or None if no data.

        Raises:
            DailyScorerError: If any step fails.
        """
        # Download data
        raw_df = self._download_data()

        # Filter to valid closes only
        valid_mask = self._has_valid_close(raw_df)
        raw_df = raw_df[valid_mask].copy()

        if raw_df.empty:
            return None

        # Compute indicators on full dataset
        df_with_indicators = self._compute_indicators(raw_df)

        # Run scoring on full dataset
        scores_df = self._run_scoring(df_with_indicators)

        # Filter to only unscored dates
        unscored_df = self._filter_unscored_dates(scores_df)

        if not unscored_df.empty:
            # Extract scores for caching
            dates = [d.to_pydatetime() for d in unscored_df["Date"]]
            signal_raw = unscored_df["signal_raw"].tolist()
            signal_0_1 = unscored_df["signal_0_1"].tolist()

            # Add to cache
            self._cache.add_scores(dates, signal_raw, signal_0_1)

        # Return latest score
        return self.get_latest_score()

    def get_latest_score(self) -> Optional[ScoreResult]:
        """
        Get the most recent score.

        Returns cached score if available, otherwise returns None.
        Call refresh() first to ensure cache is up to date.

        Returns:
            ScoreResult for the most recent date, or None if no scores.
        """
        latest = self._cache.get_latest_score()
        if latest is None:
            return None

        return ScoreResult(
            date=latest["date"],
            signal_raw=latest["signal_raw"],
            signal_0_1=latest["signal_0_1"],
            symbol=self._symbol,
            is_cached=True,
        )

    def get_score(self, date: datetime) -> Optional[ScoreResult]:
        """
        Get score for a specific date.

        Args:
            date: The date to look up.

        Returns:
            ScoreResult if found, None otherwise.
        """
        score = self._cache.get_score(date)
        if score is None:
            return None

        return ScoreResult(
            date=datetime(date.year, date.month, date.day),
            signal_raw=score["signal_raw"],
            signal_0_1=score["signal_0_1"],
            symbol=self._symbol,
            is_cached=True,
        )

    def get_all_scores(self) -> pd.DataFrame:
        """
        Get all cached scores as a DataFrame.

        Returns:
            DataFrame with columns: date, signal_raw, signal_0_1.
        """
        if len(self._cache) == 0:
            return pd.DataFrame(columns=["date", "signal_raw", "signal_0_1"])

        return self._cache._cache[["date", "signal_raw", "signal_0_1"]].copy()

    def clear_cache(self) -> None:
        """Clear all cached scores."""
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"DailyScorer(symbol={self._symbol}, "
            f"cached_scores={len(self._cache)}, "
            f"normalization={self._normalization})"
        )
