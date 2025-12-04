"""
Daily Stock Scorer Module.

A production-quality module for downloading stock data, computing technical
indicators and scores, with intelligent caching to avoid recomputation.

Usage (Python API):
    from backtest.daily_scorer import DailyScorer

    scorer = DailyScorer(
        symbol="SPY",
        strategy_files=["strategy.yaml"],
        cache_dir="./cache",
    )

    # Get the most recent score
    result = scorer.get_latest_score()
    print(f"Date: {result.date}, Score: {result.score}")

    # Force refresh (re-download and score new data)
    result = scorer.refresh()

Usage (CLI):
    python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml
    python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml --refresh
"""

from backtest.daily_scorer.scorer import DailyScorer, ScoreResult
from backtest.daily_scorer.cache import ScoreCache
from backtest.daily_scorer.downloader import download_stock_data

__all__ = [
    "DailyScorer",
    "ScoreResult",
    "ScoreCache",
    "download_stock_data",
]
