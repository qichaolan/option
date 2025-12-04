"""
Parameter Tuning Module for Backtest Engine.

This module provides grid search optimization for strategy parameters.

Usage (Python API):
    from backtest.tuning import run_parameter_search

    results = run_parameter_search(
        data_file="SPY_train.csv",
        strategy_file="strategy.yaml",
        param_config_file="params.yaml",
    )

Usage (CLI):
    python -m backtest.tuning.optimizer --data SPY.csv --strategy strategy.yaml --param_config params.yaml
"""

from backtest.tuning.optimizer import run_parameter_search

__all__ = ["run_parameter_search"]
