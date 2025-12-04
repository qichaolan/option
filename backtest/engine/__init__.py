"""
Backtest Engine Module.

A production-quality module for backtesting trading strategies using indicator data.

Supports two execution modes:
- signal_mode="discrete": Binary rule evaluation with trade execution (default)
- signal_mode="score": Continuous scoring engine without trading

Usage (Python API - Discrete Mode):
    from backtest.engine import run_backtest
    result = run_backtest(
        data_file="SPY_train.csv",
        strategy_files=["strategy.yaml"],
    )

    # Get monthly/yearly reports
    monthly_df = result.get_monthly_summary()
    yearly_df = result.get_yearly_summary()
    transactions_df = result.get_transactions()

Usage (Python API - Score Mode):
    from backtest.engine import run_scoring
    result = run_scoring(
        data_file="SPY_train.csv",
        strategy_files=["strategy.yaml"],
    )

    # Access continuous scores
    signal_raw = result.signal_raw    # [-1, +1] range
    signal_0_1 = result.signal_0_1    # [0, 1] range
    scores_df = result.scores_df      # DataFrame with all scores

Usage (CLI):
    python -m backtest.runner --data SPY.csv --strategies strategy.yaml
    python -m backtest.runner --data SPY.csv --strategies strategy.yaml --signal-mode score
    python -m backtest.runner --data SPY.csv --strategies strategy.yaml --show-monthly --show-yearly
"""

from backtest.engine.constants import SignalMode
from backtest.engine.runner import (
    run_backtest,
    run_scoring,
    run_backtest_or_scoring,
    BacktestResult,
)
from backtest.engine.scoring import ScoringResult, StrategyScore, RuleScore
from backtest.engine.reports import (
    ReportResult,
    MonthlyRecord,
    YearlyRecord,
    TransactionRecord,
    generate_full_report,
    format_monthly_summary,
    format_yearly_summary,
)

__all__ = [
    # Main API functions
    "run_backtest",
    "run_scoring",
    "run_backtest_or_scoring",
    # Result classes
    "BacktestResult",
    "ScoringResult",
    "StrategyScore",
    "RuleScore",
    # Constants
    "SignalMode",
    # Report classes and functions
    "ReportResult",
    "MonthlyRecord",
    "YearlyRecord",
    "TransactionRecord",
    "generate_full_report",
    "format_monthly_summary",
    "format_yearly_summary",
]
