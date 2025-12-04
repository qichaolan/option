"""
Reporting module for backtest results.

Provides monthly and yearly performance summaries, and detailed transaction logs
for all strategies (Trading Strategy, Lump-Sum, Monthly DCA).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backtest.engine.benchmarks import BenchmarkResult
from backtest.engine.portfolio import PortfolioResult, Trade


@dataclass
class MonthlyRecord:
    """Monthly performance record for a single strategy."""

    month: str  # Format: "YYYY-MM"
    strategy_name: str
    total_value: float
    stock_value: float
    cash_balance: float
    monthly_return_pct: float
    cumulative_return_pct: float
    num_trades: int = 0
    num_buys: int = 0
    num_sells: int = 0


@dataclass
class YearlyRecord:
    """Yearly performance record for a single strategy."""

    year: str  # Format: "YYYY"
    strategy_name: str
    total_value: float
    stock_value: float
    cash_balance: float
    total_return_pct: float
    annualized_return_pct: float
    num_trades: int = 0
    num_buys: int = 0
    num_sells: int = 0


@dataclass
class TransactionRecord:
    """Enhanced transaction record with monthly grouping."""

    strategy_name: str
    date: datetime
    month: str  # Format: "YYYY-MM"
    year: str  # Format: "YYYY"
    side: str  # "BUY" or "SELL"
    price: float
    quantity: float
    amount: float
    total_holding_shares: float
    cash_balance_after: float
    total_value_after: float


@dataclass
class ReportResult:
    """Complete report results."""

    monthly_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    transactions: Optional[pd.DataFrame] = None


def _get_month_key(date: datetime) -> str:
    """Get month key in YYYY-MM format."""
    return date.strftime("%Y-%m")


def _get_year_key(date: datetime) -> str:
    """Get year key in YYYY format."""
    return date.strftime("%Y")


def _safe_return_pct(end_value: float, start_value: float) -> float:
    """Calculate return percentage safely, avoiding division by zero."""
    if start_value <= 0:
        return 0.0
    return ((end_value / start_value) - 1) * 100


def _safe_annualized_return(
    total_return_ratio: float, n_days: int
) -> float:
    """
    Calculate annualized return safely.

    Args:
        total_return_ratio: (V_end / V_start) - 1
        n_days: Number of calendar days in period

    Returns:
        Annualized return percentage
    """
    if n_days <= 0:
        return 0.0
    if total_return_ratio <= -1:
        return -100.0
    try:
        annualized = ((1 + total_return_ratio) ** (365 / n_days) - 1) * 100
        return annualized
    except (ValueError, OverflowError):
        return 0.0


def generate_monthly_summary_from_daily(
    dates: List[datetime],
    portfolio_values: List[float],
    cash_values: List[float],
    position_values: List[float],
    strategy_name: str,
    initial_capital: float,
    trades: Optional[List[Trade]] = None,
) -> List[MonthlyRecord]:
    """
    Generate monthly summary from daily tracking data.

    Args:
        dates: List of dates (one per trading day)
        portfolio_values: Daily total portfolio values
        cash_values: Daily cash balances
        position_values: Daily stock position values
        strategy_name: Name of the strategy
        initial_capital: Starting capital
        trades: Optional list of trades for trade counts

    Returns:
        List of MonthlyRecord objects
    """
    if not dates:
        return []

    # Build DataFrame for grouping
    df = pd.DataFrame({
        "Date": dates,
        "Portfolio_Value": portfolio_values,
        "Cash": cash_values,
        "Position_Value": position_values,
    })
    df["Month"] = df["Date"].apply(_get_month_key)

    # Group by month
    monthly_records = []

    # Get unique months in order
    months = df["Month"].unique()

    # Track cumulative return from initial
    v_initial = initial_capital

    # Build trade counts by month if trades provided
    trade_counts_by_month: Dict[str, Dict[str, int]] = {}
    if trades:
        for trade in trades:
            month_key = _get_month_key(trade.date)
            if month_key not in trade_counts_by_month:
                trade_counts_by_month[month_key] = {"total": 0, "buys": 0, "sells": 0}
            trade_counts_by_month[month_key]["total"] += 1
            if trade.action == "BUY":
                trade_counts_by_month[month_key]["buys"] += 1
            else:
                trade_counts_by_month[month_key]["sells"] += 1

    prev_month_end_value = v_initial

    for month in months:
        month_df = df[df["Month"] == month]

        # Get month start and end values
        v_start = month_df["Portfolio_Value"].iloc[0]
        v_end = month_df["Portfolio_Value"].iloc[-1]

        # Use previous month's end value for monthly return calculation
        # For the first month, use initial capital
        monthly_return_pct = _safe_return_pct(v_end, prev_month_end_value)

        # Cumulative return from initial capital
        cumulative_return_pct = _safe_return_pct(v_end, v_initial)

        # Get end-of-month values
        stock_value = month_df["Position_Value"].iloc[-1]
        cash_balance = month_df["Cash"].iloc[-1]
        total_value = month_df["Portfolio_Value"].iloc[-1]

        # Get trade counts for this month
        tc = trade_counts_by_month.get(month, {"total": 0, "buys": 0, "sells": 0})

        record = MonthlyRecord(
            month=month,
            strategy_name=strategy_name,
            total_value=total_value,
            stock_value=stock_value,
            cash_balance=cash_balance,
            monthly_return_pct=monthly_return_pct,
            cumulative_return_pct=cumulative_return_pct,
            num_trades=tc["total"],
            num_buys=tc["buys"],
            num_sells=tc["sells"],
        )
        monthly_records.append(record)

        # Update for next iteration
        prev_month_end_value = v_end

    return monthly_records


def generate_yearly_summary_from_daily(
    dates: List[datetime],
    portfolio_values: List[float],
    cash_values: List[float],
    position_values: List[float],
    strategy_name: str,
    initial_capital: float,
    trades: Optional[List[Trade]] = None,
) -> List[YearlyRecord]:
    """
    Generate yearly summary from daily tracking data.

    Args:
        dates: List of dates (one per trading day)
        portfolio_values: Daily total portfolio values
        cash_values: Daily cash balances
        position_values: Daily stock position values
        strategy_name: Name of the strategy
        initial_capital: Starting capital
        trades: Optional list of trades for trade counts

    Returns:
        List of YearlyRecord objects
    """
    if not dates:
        return []

    # Build DataFrame for grouping
    df = pd.DataFrame({
        "Date": dates,
        "Portfolio_Value": portfolio_values,
        "Cash": cash_values,
        "Position_Value": position_values,
    })
    df["Year"] = df["Date"].apply(_get_year_key)

    # Group by year
    yearly_records = []
    years = df["Year"].unique()

    # Build trade counts by year if trades provided
    trade_counts_by_year: Dict[str, Dict[str, int]] = {}
    if trades:
        for trade in trades:
            year_key = _get_year_key(trade.date)
            if year_key not in trade_counts_by_year:
                trade_counts_by_year[year_key] = {"total": 0, "buys": 0, "sells": 0}
            trade_counts_by_year[year_key]["total"] += 1
            if trade.action == "BUY":
                trade_counts_by_year[year_key]["buys"] += 1
            else:
                trade_counts_by_year[year_key]["sells"] += 1

    for year in years:
        year_df = df[df["Year"] == year]

        # Get year start and end data
        start_date = year_df["Date"].iloc[0]
        end_date = year_df["Date"].iloc[-1]
        v_start = year_df["Portfolio_Value"].iloc[0]
        v_end = year_df["Portfolio_Value"].iloc[-1]

        # Calculate number of calendar days
        n_days = (end_date - start_date).days
        if n_days == 0:
            n_days = 1  # Avoid division by zero for single-day years

        # Calculate returns
        total_return_ratio = (v_end / v_start - 1) if v_start > 0 else 0
        total_return_pct = total_return_ratio * 100
        annualized_return_pct = _safe_annualized_return(total_return_ratio, n_days)

        # Get end-of-year values
        stock_value = year_df["Position_Value"].iloc[-1]
        cash_balance = year_df["Cash"].iloc[-1]
        total_value = year_df["Portfolio_Value"].iloc[-1]

        # Get trade counts for this year
        tc = trade_counts_by_year.get(year, {"total": 0, "buys": 0, "sells": 0})

        record = YearlyRecord(
            year=year,
            strategy_name=strategy_name,
            total_value=total_value,
            stock_value=stock_value,
            cash_balance=cash_balance,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            num_trades=tc["total"],
            num_buys=tc["buys"],
            num_sells=tc["sells"],
        )
        yearly_records.append(record)

    return yearly_records


def generate_transaction_log(
    trades: List[Trade],
    strategy_name: str,
) -> List[TransactionRecord]:
    """
    Generate enriched transaction log from trades.

    Args:
        trades: List of Trade objects
        strategy_name: Name of the strategy

    Returns:
        List of TransactionRecord objects
    """
    records = []

    for trade in trades:
        record = TransactionRecord(
            strategy_name=strategy_name,
            date=trade.date,
            month=_get_month_key(trade.date),
            year=_get_year_key(trade.date),
            side=trade.action,
            price=trade.price,
            quantity=trade.shares,
            amount=trade.value,
            total_holding_shares=trade.position_after,
            cash_balance_after=trade.cash_after,
            total_value_after=trade.portfolio_value,
        )
        records.append(record)

    return records


def generate_benchmark_monthly_summary(
    benchmark: BenchmarkResult,
    initial_capital: float,
) -> List[MonthlyRecord]:
    """
    Generate monthly summary for a benchmark strategy.

    Since benchmarks don't have separate cash/stock tracking,
    we treat the entire value as stock value (for buy-hold strategies)
    or cash (for cash-only).

    Args:
        benchmark: BenchmarkResult object
        initial_capital: Starting capital

    Returns:
        List of MonthlyRecord objects
    """
    if not benchmark.dates:
        return []

    # Determine if this is cash-only benchmark
    is_cash_only = "Cash" in benchmark.name

    # Build daily values
    if is_cash_only:
        # Cash-only: all value is cash
        cash_values = benchmark.daily_values
        position_values = [0.0] * len(benchmark.daily_values)
    else:
        # Buy-hold / DCA: all value is in stock
        cash_values = [0.0] * len(benchmark.daily_values)
        position_values = benchmark.daily_values

    return generate_monthly_summary_from_daily(
        dates=benchmark.dates,
        portfolio_values=benchmark.daily_values,
        cash_values=cash_values,
        position_values=position_values,
        strategy_name=benchmark.name,
        initial_capital=initial_capital,
        trades=None,  # Benchmarks don't have trades
    )


def generate_benchmark_yearly_summary(
    benchmark: BenchmarkResult,
    initial_capital: float,
) -> List[YearlyRecord]:
    """
    Generate yearly summary for a benchmark strategy.

    Args:
        benchmark: BenchmarkResult object
        initial_capital: Starting capital

    Returns:
        List of YearlyRecord objects
    """
    if not benchmark.dates:
        return []

    # Determine if this is cash-only benchmark
    is_cash_only = "Cash" in benchmark.name

    if is_cash_only:
        cash_values = benchmark.daily_values
        position_values = [0.0] * len(benchmark.daily_values)
    else:
        cash_values = [0.0] * len(benchmark.daily_values)
        position_values = benchmark.daily_values

    return generate_yearly_summary_from_daily(
        dates=benchmark.dates,
        portfolio_values=benchmark.daily_values,
        cash_values=cash_values,
        position_values=position_values,
        strategy_name=benchmark.name,
        initial_capital=initial_capital,
        trades=None,
    )


def generate_full_report(
    portfolio: PortfolioResult,
    benchmarks: List[BenchmarkResult],
    include_transactions: bool = False,
) -> ReportResult:
    """
    Generate complete report with monthly/yearly summaries for all strategies.

    Args:
        portfolio: PortfolioResult from trading strategy simulation
        benchmarks: List of BenchmarkResult objects
        include_transactions: Whether to include detailed transaction log

    Returns:
        ReportResult with monthly_summary, yearly_summary, and optional transactions
    """
    all_monthly_records: List[MonthlyRecord] = []
    all_yearly_records: List[YearlyRecord] = []
    all_transactions: List[TransactionRecord] = []

    # Process trading strategy
    strategy_monthly = generate_monthly_summary_from_daily(
        dates=portfolio.dates,
        portfolio_values=portfolio.portfolio_values,
        cash_values=portfolio.cash_values,
        position_values=portfolio.position_values,
        strategy_name="Trading Strategy",
        initial_capital=portfolio.initial_capital,
        trades=portfolio.trades,
    )
    all_monthly_records.extend(strategy_monthly)

    strategy_yearly = generate_yearly_summary_from_daily(
        dates=portfolio.dates,
        portfolio_values=portfolio.portfolio_values,
        cash_values=portfolio.cash_values,
        position_values=portfolio.position_values,
        strategy_name="Trading Strategy",
        initial_capital=portfolio.initial_capital,
        trades=portfolio.trades,
    )
    all_yearly_records.extend(strategy_yearly)

    if include_transactions:
        tx_records = generate_transaction_log(
            trades=portfolio.trades,
            strategy_name="Trading Strategy",
        )
        all_transactions.extend(tx_records)

    # Process benchmarks
    for benchmark in benchmarks:
        bm_monthly = generate_benchmark_monthly_summary(
            benchmark=benchmark,
            initial_capital=benchmark.initial_capital,
        )
        all_monthly_records.extend(bm_monthly)

        bm_yearly = generate_benchmark_yearly_summary(
            benchmark=benchmark,
            initial_capital=benchmark.initial_capital,
        )
        all_yearly_records.extend(bm_yearly)

    # Convert to DataFrames
    monthly_df = _monthly_records_to_dataframe(all_monthly_records)
    yearly_df = _yearly_records_to_dataframe(all_yearly_records)

    transactions_df = None
    if include_transactions and all_transactions:
        transactions_df = _transaction_records_to_dataframe(all_transactions)

    return ReportResult(
        monthly_summary=monthly_df,
        yearly_summary=yearly_df,
        transactions=transactions_df,
    )


def _monthly_records_to_dataframe(records: List[MonthlyRecord]) -> pd.DataFrame:
    """Convert MonthlyRecord list to DataFrame."""
    if not records:
        return pd.DataFrame(columns=[
            "month", "strategy_name", "total_value", "stock_value",
            "cash_balance", "monthly_return_pct", "cumulative_return_pct",
            "num_trades", "num_buys", "num_sells"
        ])

    return pd.DataFrame([
        {
            "month": r.month,
            "strategy_name": r.strategy_name,
            "total_value": r.total_value,
            "stock_value": r.stock_value,
            "cash_balance": r.cash_balance,
            "monthly_return_pct": r.monthly_return_pct,
            "cumulative_return_pct": r.cumulative_return_pct,
            "num_trades": r.num_trades,
            "num_buys": r.num_buys,
            "num_sells": r.num_sells,
        }
        for r in records
    ])


def _yearly_records_to_dataframe(records: List[YearlyRecord]) -> pd.DataFrame:
    """Convert YearlyRecord list to DataFrame."""
    if not records:
        return pd.DataFrame(columns=[
            "year", "strategy_name", "total_value", "stock_value",
            "cash_balance", "total_return_pct", "annualized_return_pct",
            "num_trades", "num_buys", "num_sells"
        ])

    return pd.DataFrame([
        {
            "year": r.year,
            "strategy_name": r.strategy_name,
            "total_value": r.total_value,
            "stock_value": r.stock_value,
            "cash_balance": r.cash_balance,
            "total_return_pct": r.total_return_pct,
            "annualized_return_pct": r.annualized_return_pct,
            "num_trades": r.num_trades,
            "num_buys": r.num_buys,
            "num_sells": r.num_sells,
        }
        for r in records
    ])


def _transaction_records_to_dataframe(records: List[TransactionRecord]) -> pd.DataFrame:
    """Convert TransactionRecord list to DataFrame."""
    if not records:
        return pd.DataFrame(columns=[
            "strategy_name", "date", "month", "year", "side",
            "price", "quantity", "amount", "total_holding_shares",
            "cash_balance_after", "total_value_after"
        ])

    return pd.DataFrame([
        {
            "strategy_name": r.strategy_name,
            "date": r.date,
            "month": r.month,
            "year": r.year,
            "side": r.side,
            "price": r.price,
            "quantity": r.quantity,
            "amount": r.amount,
            "total_holding_shares": r.total_holding_shares,
            "cash_balance_after": r.cash_balance_after,
            "total_value_after": r.total_value_after,
        }
        for r in records
    ])


def format_monthly_summary(df: pd.DataFrame) -> str:
    """
    Format monthly summary DataFrame as a readable string.

    Args:
        df: Monthly summary DataFrame

    Returns:
        Formatted string representation
    """
    if df.empty:
        return "No monthly data available."

    lines = [
        "=" * 100,
        "MONTHLY PERFORMANCE SUMMARY",
        "=" * 100,
        "",
    ]

    # Group by strategy for cleaner display
    for strategy in df["strategy_name"].unique():
        strategy_df = df[df["strategy_name"] == strategy]
        lines.append(f"\n{strategy}")
        lines.append("-" * 80)
        lines.append(
            f"{'Month':<10} {'Total Value':>15} {'Stock Value':>15} "
            f"{'Cash':>12} {'Monthly %':>10} {'Cumul %':>10} {'Trades':>8}"
        )
        lines.append("-" * 80)

        for _, row in strategy_df.iterrows():
            lines.append(
                f"{row['month']:<10} "
                f"${row['total_value']:>14,.2f} "
                f"${row['stock_value']:>14,.2f} "
                f"${row['cash_balance']:>11,.2f} "
                f"{row['monthly_return_pct']:>+9.2f}% "
                f"{row['cumulative_return_pct']:>+9.2f}% "
                f"{row['num_trades']:>8}"
            )

    # Add monthly comparison table
    lines.append("\n")
    lines.append("=" * 100)
    lines.append("MONTHLY COMPARISON TABLE")
    lines.append("=" * 100)

    strategies = df["strategy_name"].unique()
    months = df["month"].unique()

    # Build header row with strategy names
    header = f"{'Month':<10}"
    for strategy in strategies:
        # Truncate strategy name to fit
        short_name = strategy[:18] if len(strategy) > 18 else strategy
        header += f" | {short_name:>18} {'Monthly%':>9} {'Cumul%':>9}"
    lines.append(header)
    lines.append("-" * (10 + len(strategies) * 40))

    # Build data rows for each month
    for month in months:
        row = f"{month:<10}"
        for strategy in strategies:
            month_data = df[(df["month"] == month) & (df["strategy_name"] == strategy)]
            if not month_data.empty:
                total_val = month_data["total_value"].values[0]
                monthly_pct = month_data["monthly_return_pct"].values[0]
                cumul_pct = month_data["cumulative_return_pct"].values[0]
                row += f" | ${total_val:>17,.2f} {monthly_pct:>+8.2f}% {cumul_pct:>+8.2f}%"
            else:
                row += f" | {'N/A':>18} {'N/A':>9} {'N/A':>9}"
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


def format_yearly_summary(df: pd.DataFrame) -> str:
    """
    Format yearly summary DataFrame as a readable string.

    Args:
        df: Yearly summary DataFrame

    Returns:
        Formatted string representation
    """
    if df.empty:
        return "No yearly data available."

    lines = [
        "=" * 100,
        "YEARLY PERFORMANCE SUMMARY",
        "=" * 100,
        "",
    ]

    # Group by strategy
    for strategy in df["strategy_name"].unique():
        strategy_df = df[df["strategy_name"] == strategy]
        lines.append(f"\n{strategy}")
        lines.append("-" * 90)
        lines.append(
            f"{'Year':<6} {'Total Value':>15} {'Stock Value':>15} "
            f"{'Cash':>12} {'Return %':>10} {'Annualized %':>12} {'Trades':>8}"
        )
        lines.append("-" * 90)

        for _, row in strategy_df.iterrows():
            lines.append(
                f"{row['year']:<6} "
                f"${row['total_value']:>14,.2f} "
                f"${row['stock_value']:>14,.2f} "
                f"${row['cash_balance']:>11,.2f} "
                f"{row['total_return_pct']:>+9.2f}% "
                f"{row['annualized_return_pct']:>+11.2f}% "
                f"{row['num_trades']:>8}"
            )

    # Add yearly comparison table
    lines.append("\n")
    lines.append("=" * 100)
    lines.append("YEARLY COMPARISON TABLE")
    lines.append("=" * 100)

    strategies = df["strategy_name"].unique()
    years = df["year"].unique()

    # Build header row with strategy names
    header = f"{'Year':<6}"
    for strategy in strategies:
        # Truncate strategy name to fit
        short_name = strategy[:18] if len(strategy) > 18 else strategy
        header += f" | {short_name:>18} {'Return%':>9} {'Annual%':>9}"
    lines.append(header)
    lines.append("-" * (6 + len(strategies) * 40))

    # Build data rows for each year
    for year in years:
        row = f"{year:<6}"
        for strategy in strategies:
            year_data = df[(df["year"] == year) & (df["strategy_name"] == strategy)]
            if not year_data.empty:
                total_val = year_data["total_value"].values[0]
                return_pct = year_data["total_return_pct"].values[0]
                annual_pct = year_data["annualized_return_pct"].values[0]
                row += f" | ${total_val:>17,.2f} {return_pct:>+8.2f}% {annual_pct:>+8.2f}%"
            else:
                row += f" | {'N/A':>18} {'N/A':>9} {'N/A':>9}"
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)
