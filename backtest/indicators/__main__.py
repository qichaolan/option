"""
CLI entry point for the indicators module.

Allows running as: python -m backtest.indicators
"""

import sys

from backtest.indicators.main import main

if __name__ == "__main__":
    sys.exit(main())
