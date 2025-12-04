"""
Stock Indicator Generation Module.

A production-quality module for calculating technical indicators from OHLCV data.

Modules:
    - loader: CSV loading and parsing
    - validators: Data validation functions
    - calculations: Technical indicator calculations
    - main: Main entry point with API and CLI
"""

from backtest.indicators.main import build_indicators

__all__ = ["build_indicators"]
__version__ = "1.0.0"
