"""
Score caching system.

Caches computed scores to avoid recomputation for dates already scored.
Uses a simple CSV file for persistence.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from backtest.daily_scorer.exceptions import CacheError


class ScoreCache:
    """
    Cache for daily scores.

    Stores scores in a CSV file with columns:
    - date: Trading date (YYYY-MM-DD)
    - signal_raw: Raw score in [-1, +1]
    - signal_0_1: Normalized score in [0, 1]
    - scored_at: Timestamp when score was computed
    """

    COLUMNS = ["date", "signal_raw", "signal_0_1", "scored_at"]

    def __init__(self, cache_path: str):
        """
        Initialize score cache.

        Args:
            cache_path: Path to the cache CSV file.
        """
        self._cache_path = Path(cache_path)
        self._cache: pd.DataFrame = self._load_cache()

    @property
    def cache_path(self) -> Path:
        """Return the cache file path."""
        return self._cache_path

    def _load_cache(self) -> pd.DataFrame:
        """Load cache from disk, or create empty cache."""
        if self._cache_path.exists():
            try:
                df = pd.read_csv(self._cache_path, parse_dates=["date", "scored_at"])
                # Validate columns
                for col in self.COLUMNS:
                    if col not in df.columns:
                        raise CacheError(
                            f"Cache file missing required column: {col}"
                        )
                return df
            except Exception as e:
                if isinstance(e, CacheError):
                    raise
                raise CacheError(f"Failed to load cache: {e}") from e
        else:
            return pd.DataFrame(columns=self.COLUMNS)

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            # Ensure parent directory exists
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache.to_csv(self._cache_path, index=False)
        except Exception as e:
            raise CacheError(f"Failed to save cache: {e}") from e

    def get_scored_dates(self) -> Set[datetime]:
        """
        Get set of dates that have already been scored.

        Returns:
            Set of datetime objects for scored dates.
        """
        if self._cache.empty:
            return set()
        return set(self._cache["date"].dt.to_pydatetime())

    def get_score(self, date: datetime) -> Optional[Dict[str, float]]:
        """
        Get cached score for a specific date.

        Args:
            date: The date to look up.

        Returns:
            Dict with 'signal_raw' and 'signal_0_1', or None if not cached.
        """
        if self._cache.empty:
            return None

        # Normalize date to midnight
        date_normalized = datetime(date.year, date.month, date.day)

        mask = self._cache["date"].dt.normalize() == pd.Timestamp(date_normalized)
        matches = self._cache[mask]

        if matches.empty:
            return None

        row = matches.iloc[-1]  # Use most recent if duplicates
        return {
            "signal_raw": float(row["signal_raw"]),
            "signal_0_1": float(row["signal_0_1"]),
        }

    def get_latest_score(self) -> Optional[Dict]:
        """
        Get the most recent cached score.

        Returns:
            Dict with 'date', 'signal_raw', 'signal_0_1', or None if cache empty.
        """
        if self._cache.empty:
            return None

        # Sort by date descending and get first
        sorted_cache = self._cache.sort_values("date", ascending=False)
        row = sorted_cache.iloc[0]

        return {
            "date": row["date"].to_pydatetime(),
            "signal_raw": float(row["signal_raw"]),
            "signal_0_1": float(row["signal_0_1"]),
        }

    def add_scores(
        self,
        dates: List[datetime],
        signal_raw: List[float],
        signal_0_1: List[float],
    ) -> int:
        """
        Add new scores to the cache.

        Only adds scores for dates not already in cache.

        Args:
            dates: List of dates.
            signal_raw: List of raw scores.
            signal_0_1: List of normalized scores.

        Returns:
            Number of new scores added.
        """
        if len(dates) != len(signal_raw) or len(dates) != len(signal_0_1):
            raise ValueError("All input lists must have the same length")

        existing_dates = self.get_scored_dates()
        now = datetime.now()

        new_rows = []
        for date, raw, norm in zip(dates, signal_raw, signal_0_1):
            # Normalize date
            date_normalized = datetime(date.year, date.month, date.day)
            if date_normalized not in existing_dates:
                new_rows.append({
                    "date": date_normalized,
                    "signal_raw": raw,
                    "signal_0_1": norm,
                    "scored_at": now,
                })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self._cache = pd.concat([self._cache, new_df], ignore_index=True)
            self._save_cache()

        return len(new_rows)

    def clear(self) -> None:
        """Clear all cached scores."""
        self._cache = pd.DataFrame(columns=self.COLUMNS)
        if self._cache_path.exists():
            self._cache_path.unlink()

    def __len__(self) -> int:
        """Return number of cached scores."""
        return len(self._cache)

    def __contains__(self, date: datetime) -> bool:
        """Check if a date is in the cache."""
        return self.get_score(date) is not None
