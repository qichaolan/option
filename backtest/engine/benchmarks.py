"""
Benchmark calculation module for backtesting.

This module provides benchmark comparisons including:
- Cash-Only: 5% annual risk-free rate
- Lump-Sum Buy-and-Hold: Invest all capital day 1
- Monthly DCA: Invest equal amounts every ~20 trading days
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd


# Constants
RISK_FREE_RATE = 0.05  # 5% annual
TRADING_DAYS_PER_MONTH = 20
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BenchmarkResult:
    """Results from a benchmark strategy."""

    name: str
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    daily_values: List[float]
    dates: List[datetime]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "Date": self.dates,
            "Portfolio_Value": self.daily_values,
        })


def calculate_cash_only(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
) -> BenchmarkResult:
    """
    Calculate cash-only benchmark with 5% annual risk-free rate.

    Args:
        df: DataFrame with 'Date' column.
        initial_capital: Starting capital.

    Returns:
        BenchmarkResult with daily compounded values.
    """
    dates = df["Date"].tolist()
    n_days = len(dates)

    # Daily risk-free rate (continuous compounding)
    daily_rate = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

    daily_values = []
    for i in range(n_days):
        # Simple daily compounding
        value = initial_capital * (1 + daily_rate) ** i
        daily_values.append(value)

    final_value = daily_values[-1] if daily_values else initial_capital
    total_return = final_value - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    return BenchmarkResult(
        name="Cash-Only (5% Risk-Free)",
        initial_capital=initial_capital,
        final_value=final_value,
        total_return=total_return,
        total_return_pct=total_return_pct,
        daily_values=daily_values,
        dates=dates,
    )


def calculate_lump_sum(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
) -> BenchmarkResult:
    """
    Calculate lump-sum buy-and-hold benchmark.

    Invests all capital on day 1 at Close price.

    Args:
        df: DataFrame with 'Date' and 'Close' columns.
        initial_capital: Starting capital.

    Returns:
        BenchmarkResult with daily portfolio values.
    """
    dates = df["Date"].tolist()
    closes = df["Close"].values

    # Buy all shares on day 1
    initial_price = closes[0]
    shares = initial_capital / initial_price

    # Track daily value
    daily_values = [shares * price for price in closes]

    final_value = daily_values[-1] if daily_values else initial_capital
    total_return = final_value - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    return BenchmarkResult(
        name="Lump-Sum Buy-and-Hold",
        initial_capital=initial_capital,
        final_value=final_value,
        total_return=total_return,
        total_return_pct=total_return_pct,
        daily_values=daily_values,
        dates=dates,
    )


def calculate_monthly_dca(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    period_days: int = TRADING_DAYS_PER_MONTH,
) -> BenchmarkResult:
    """
    Calculate monthly DCA (Dollar-Cost Averaging) benchmark.

    Invests equal amounts every N trading days (default 20).

    Args:
        df: DataFrame with 'Date' and 'Close' columns.
        initial_capital: Starting capital.
        period_days: Days between investments (default 20).

    Returns:
        BenchmarkResult with daily portfolio values.
    """
    dates = df["Date"].tolist()
    closes = df["Close"].values
    n_days = len(dates)

    # Calculate number of investment periods
    num_investments = (n_days // period_days) + 1
    investment_per_period = initial_capital / num_investments

    # Track state
    shares = 0.0
    cash_remaining = initial_capital
    next_investment_day = 0
    daily_values = []

    for i in range(n_days):
        price = closes[i]

        # Check if it's time to invest
        if i >= next_investment_day and cash_remaining >= investment_per_period:
            # Invest
            amount_to_invest = min(investment_per_period, cash_remaining)
            shares_bought = amount_to_invest / price
            shares += shares_bought
            cash_remaining -= amount_to_invest
            next_investment_day = i + period_days

        # If we still have cash on the last investment period, invest it
        elif i >= next_investment_day and cash_remaining > 0:
            shares_bought = cash_remaining / price
            shares += shares_bought
            cash_remaining = 0
            next_investment_day = i + period_days

        # Calculate daily portfolio value (shares * price + remaining cash)
        portfolio_value = shares * price + cash_remaining
        daily_values.append(portfolio_value)

    final_value = daily_values[-1] if daily_values else initial_capital
    total_return = final_value - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    return BenchmarkResult(
        name="Monthly DCA",
        initial_capital=initial_capital,
        final_value=final_value,
        total_return=total_return,
        total_return_pct=total_return_pct,
        daily_values=daily_values,
        dates=dates,
    )


def calculate_all_benchmarks(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
) -> List[BenchmarkResult]:
    """
    Calculate all benchmark strategies.

    Args:
        df: DataFrame with 'Date' and 'Close' columns.
        initial_capital: Starting capital.

    Returns:
        List of BenchmarkResults for all strategies.
    """
    return [
        calculate_cash_only(df, initial_capital),
        calculate_lump_sum(df, initial_capital),
        calculate_monthly_dca(df, initial_capital),
    ]


def compare_results(
    strategy_result: "PortfolioResult",
    benchmarks: List[BenchmarkResult],
) -> pd.DataFrame:
    """
    Create comparison table of strategy vs benchmarks.

    Args:
        strategy_result: PortfolioResult from strategy simulation.
        benchmarks: List of benchmark results.

    Returns:
        DataFrame with comparison metrics.
    """
    rows = []

    # Strategy row
    rows.append({
        "Strategy": "Trading Strategy",
        "Initial Capital": strategy_result.initial_capital,
        "Final Value": strategy_result.final_value,
        "Total Return": strategy_result.total_return,
        "Return %": strategy_result.total_return_pct,
    })

    # Benchmark rows
    for bm in benchmarks:
        rows.append({
            "Strategy": bm.name,
            "Initial Capital": bm.initial_capital,
            "Final Value": bm.final_value,
            "Total Return": bm.total_return,
            "Return %": bm.total_return_pct,
        })

    return pd.DataFrame(rows)
