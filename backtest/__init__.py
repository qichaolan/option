"""
Backtest Package.

A production-quality module for calculating technical indicators from OHLCV data
and running backtests on trading strategies.

Modules:
    - indicators: Technical indicator calculations
    - engine: Backtest engine for strategy evaluation

Main APIs:
    - build_indicators: Calculate technical indicators from OHLCV CSV
    - run_backtest: Run backtest on indicator data with strategies
"""

from backtest.indicators import build_indicators
from backtest.engine import run_backtest

__all__ = ["build_indicators", "run_backtest"]
__version__ = "1.0.0"
