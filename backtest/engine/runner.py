"""
Main runner module for the backtest engine.

This module provides the primary API and CLI for running backtests.

Supports two execution modes:
- signal_mode="discrete": Binary rule evaluation with trade execution (default)
- signal_mode="score": Continuous scoring engine without trading
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from backtest.engine.benchmarks import (
    BenchmarkResult,
    calculate_all_benchmarks,
    compare_results,
)
from backtest.engine.constants import SignalMode
from backtest.engine.reports import (
    ReportResult,
    generate_full_report,
    format_monthly_summary,
    format_yearly_summary,
)
from backtest.engine.exceptions import (
    BacktestError,
    DataError,
    FileNotFoundError,
    InsufficientDataError,
    InvalidParameterError,
)
from backtest.engine.portfolio import PortfolioResult, simulate_portfolio
from backtest.engine.rule_engine import run_rule_engine
from backtest.engine.scoring import ScoringResult, run_scoring_engine
from backtest.engine.strategy_loader import (
    Strategy,
    load_strategies,
    normalize_weights,
)


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    # Core results
    portfolio: PortfolioResult
    benchmarks: List[BenchmarkResult]

    # Input info
    strategies: List[Strategy]
    data_file: str
    strategy_files: List[str]
    initial_capital: float
    date_range: Tuple[str, str]
    num_rows: int

    # Comparison DataFrame
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Reports (generated on demand)
    _report: Optional[ReportResult] = field(default=None, repr=False)

    def __post_init__(self):
        """Generate comparison after initialization."""
        if self.comparison.empty and self.benchmarks:
            self.comparison = compare_results(self.portfolio, self.benchmarks)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            f"Data: {self.data_file}",
            f"Date Range: {self.date_range[0]} to {self.date_range[1]}",
            f"Trading Days: {self.num_rows}",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            "",
            "Strategies Used:",
        ]

        for s in self.strategies:
            lines.append(f"  - {s.name} (weight: {s.weight:.2%})")

        lines.extend([
            "",
            "-" * 60,
            "PERFORMANCE COMPARISON",
            "-" * 60,
            "",
        ])

        # Format comparison table
        if not self.comparison.empty:
            for _, row in self.comparison.iterrows():
                lines.append(
                    f"{row['Strategy']:30} | "
                    f"${row['Final Value']:>12,.2f} | "
                    f"{row['Return %']:>+7.2f}%"
                )

        lines.extend([
            "",
            "-" * 60,
            "TRADING SUMMARY",
            "-" * 60,
            "",
            f"Total Trades: {self.portfolio.num_trades}",
            f"  - Buys: {self.portfolio.num_buys}",
            f"  - Sells: {self.portfolio.num_sells}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def get_report(self, include_transactions: bool = False) -> ReportResult:
        """
        Get monthly/yearly performance reports.

        Args:
            include_transactions: Whether to include detailed transaction log.

        Returns:
            ReportResult with monthly_summary, yearly_summary, and optional transactions.
        """
        # Generate report if not cached or if transactions requested but not included
        if self._report is None or (include_transactions and self._report.transactions is None):
            self._report = generate_full_report(
                portfolio=self.portfolio,
                benchmarks=self.benchmarks,
                include_transactions=include_transactions,
            )
        return self._report

    def get_monthly_summary(self) -> pd.DataFrame:
        """
        Get monthly performance summary for all strategies.

        Returns:
            DataFrame with columns: month, strategy_name, total_value, stock_value,
            cash_balance, monthly_return_pct, cumulative_return_pct, num_trades,
            num_buys, num_sells
        """
        return self.get_report().monthly_summary

    def get_yearly_summary(self) -> pd.DataFrame:
        """
        Get yearly performance summary for all strategies.

        Returns:
            DataFrame with columns: year, strategy_name, total_value, stock_value,
            cash_balance, total_return_pct, annualized_return_pct, num_trades,
            num_buys, num_sells
        """
        return self.get_report().yearly_summary

    def get_transactions(self) -> pd.DataFrame:
        """
        Get detailed transaction log.

        Returns:
            DataFrame with columns: strategy_name, date, month, year, side,
            price, quantity, amount, total_holding_shares, cash_balance_after,
            total_value_after
        """
        report = self.get_report(include_transactions=True)
        return report.transactions if report.transactions is not None else pd.DataFrame()

    def monthly_summary_text(self) -> str:
        """Get formatted monthly summary as text."""
        return format_monthly_summary(self.get_monthly_summary())

    def yearly_summary_text(self) -> str:
        """Get formatted yearly summary as text."""
        return format_yearly_summary(self.get_yearly_summary())


def validate_data(df: pd.DataFrame, file_path: str = "data") -> None:
    """
    Validate data for backtest requirements.

    Args:
        df: DataFrame to validate.
        file_path: Source file path for error messages.

    Raises:
        DataError: If data validation fails.
    """
    # Check for duplicate dates first (always check, regardless of order)
    if df["Date"].duplicated().any():
        dup_dates = df[df["Date"].duplicated()]["Date"].unique()
        raise DataError(
            f"Duplicate dates found in {file_path}: {list(dup_dates[:5])}"
            + (f" (and {len(dup_dates) - 5} more)" if len(dup_dates) > 5 else "")
        )

    # Check for monotonically increasing dates (strictly increasing, no duplicates)
    if not df["Date"].is_monotonic_increasing:
        # Dates are out of order - sort them
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Check for NaN in critical columns
    if df["Close"].isna().any():
        nan_count = df["Close"].isna().sum()
        raise DataError(f"Found {nan_count} NaN values in 'Close' column in {file_path}")

    # Check for non-positive Close prices
    if (df["Close"] <= 0).any():
        invalid_count = (df["Close"] <= 0).sum()
        raise DataError(f"Found {invalid_count} non-positive values in 'Close' column")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load indicator CSV data.

    Args:
        file_path: Path to CSV file.

    Returns:
        DataFrame with data, validated and sorted by date.

    Raises:
        FileNotFoundError: If file doesn't exist.
        DataError: If file is invalid.
        InsufficientDataError: If file is empty.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"Failed to load data from {file_path}: {e}")

    if df.empty:
        raise InsufficientDataError(0)

    # Ensure Date column exists and is datetime
    if "Date" not in df.columns:
        raise DataError(f"Missing 'Date' column in {file_path}")

    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure Close column exists
    if "Close" not in df.columns:
        raise DataError(f"Missing 'Close' column in {file_path}")

    # Validate data quality
    validate_data(df, file_path)

    return df


def run_backtest(
    data_file: Union[str, pd.DataFrame],
    strategy_files: Union[str, List[str]],
    initial_capital: float = 100000.0,
    output_file: Optional[str] = None,
    trades_file: Optional[str] = None,
) -> BacktestResult:
    """
    Run a complete backtest.

    This is the main API function for programmatic usage.

    Args:
        data_file: Path to CSV file with indicator data, or a DataFrame directly.
        strategy_files: Path(s) to YAML strategy file(s).
        initial_capital: Starting capital (default $100,000).
        output_file: Optional path to save daily results CSV.
        trades_file: Optional path to save trade log CSV.

    Returns:
        BacktestResult with all results and metrics.

    Raises:
        FileNotFoundError: If any file doesn't exist.
        InvalidStrategyError: If strategy YAML is invalid.
        MissingIndicatorError: If required indicators are missing.
        DataError: If data is invalid.

    Example:
        >>> result = run_backtest(
        ...     data_file="SPY_train.csv",
        ...     strategy_files=["strategy.yaml"],
        ... )
        >>> print(result.summary())
    """
    # Validate initial capital
    if initial_capital <= 0:
        raise InvalidParameterError(
            "initial_capital", initial_capital, "must be positive"
        )
    # Warn if capital is very small (may cause precision issues)
    if initial_capital < 100:
        import warnings
        warnings.warn(
            f"Initial capital ${initial_capital} is very small. "
            "Results may have precision issues with very small position sizes.",
            UserWarning,
        )

    # Load data - accept DataFrame or file path
    if isinstance(data_file, pd.DataFrame):
        df = data_file.copy()
        data_source = "<DataFrame>"
        # Ensure Date column is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        validate_data(df, data_source)
    else:
        df = load_data(data_file)
        data_source = data_file

    # Load and normalize strategies
    if isinstance(strategy_files, str):
        strategy_files = [strategy_files]

    strategies = load_strategies(strategy_files)
    strategies = normalize_weights(strategies)

    # Run rule engine to get signals
    signals, scores = run_rule_engine(df, strategies)

    # Simulate portfolio
    portfolio = simulate_portfolio(df, signals, scores, initial_capital)

    # Calculate benchmarks
    benchmarks = calculate_all_benchmarks(df, initial_capital)

    # Get date range
    date_range = (
        df["Date"].min().strftime("%Y-%m-%d"),
        df["Date"].max().strftime("%Y-%m-%d"),
    )

    # Create result
    result = BacktestResult(
        portfolio=portfolio,
        benchmarks=benchmarks,
        strategies=strategies,
        data_file=data_source,
        strategy_files=strategy_files,
        initial_capital=initial_capital,
        date_range=date_range,
        num_rows=len(df),
    )

    # Save outputs if requested
    if output_file:
        daily_df = portfolio.to_dataframe()
        daily_df.to_csv(output_file, index=False)

    if trades_file:
        trades_df = portfolio.trades_to_dataframe()
        if not trades_df.empty:
            trades_df.to_csv(trades_file, index=False)

    return result


def run_scoring(
    data_file: Union[str, pd.DataFrame],
    strategy_files: Union[str, List[str]],
    output_file: Optional[str] = None,
    include_rule_scores: bool = False,
    normalization: str = "none",
) -> ScoringResult:
    """
    Run the continuous scoring engine (score mode).

    This function evaluates strategies using continuous activation functions
    and returns scores without executing any trades.

    Output includes:
    - signal_raw: Score in [-1, +1] range (strong sell → strong buy)
    - signal_0_1: Score in [0, 1] range (UI-friendly format)
    - Per-strategy scores
    - Merged DataFrame with all indicators and scores

    Args:
        data_file: Path to CSV file with indicator data, or a DataFrame directly.
        strategy_files: Path(s) to YAML strategy file(s).
        output_file: Optional path to save scores CSV.
        include_rule_scores: Whether to include individual rule scores in output.
        normalization: Normalization method for signal_0_1:
            - "none": Simple linear transform (default)
            - "minmax": Rescale to use full [0,1] range
            - "zscore": Z-score with CDF mapping, spreads distribution:
                * Bearish: 0.0-0.3
                * Neutral: 0.4-0.6
                * Bullish: 0.7-1.0

    Returns:
        ScoringResult with all scores and merged DataFrame.

    Raises:
        FileNotFoundError: If any file doesn't exist.
        InvalidStrategyError: If strategy YAML is invalid.
        MissingIndicatorError: If required indicators are missing.
        DataError: If data is invalid.

    Example:
        >>> result = run_scoring(
        ...     data_file="SPY_train.csv",
        ...     strategy_files=["strategy.yaml"],
        ...     normalization="zscore",  # Spread to bearish/neutral/bullish ranges
        ... )
        >>> print(result.signal_raw.describe())
        >>> print(result.scores_df.head())
    """
    # Load data - accept DataFrame or file path
    if isinstance(data_file, pd.DataFrame):
        df = data_file.copy()
        data_source = "<DataFrame>"
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        validate_data(df, data_source)
    else:
        df = load_data(data_file)
        data_source = data_file

    # Load and normalize strategies
    if isinstance(strategy_files, str):
        strategy_files = [strategy_files]

    strategies = load_strategies(strategy_files)
    strategies = normalize_weights(strategies)

    # Run scoring engine
    result = run_scoring_engine(
        df=df,
        strategies=strategies,
        include_rule_scores=include_rule_scores,
        normalization=normalization,
    )

    # Save output if requested
    if output_file:
        result.scores_df.to_csv(output_file, index=False)

    return result


def run_backtest_or_scoring(
    data_file: Union[str, pd.DataFrame],
    strategy_files: Union[str, List[str]],
    signal_mode: str = SignalMode.DISCRETE,
    initial_capital: float = 100000.0,
    output_file: Optional[str] = None,
    trades_file: Optional[str] = None,
    include_rule_scores: bool = False,
) -> Union[BacktestResult, ScoringResult]:
    """
    Run backtest or scoring based on signal_mode.

    This is a convenience function that dispatches to either run_backtest
    or run_scoring based on the signal_mode parameter.

    Args:
        data_file: Path to CSV file with indicator data, or a DataFrame directly.
        strategy_files: Path(s) to YAML strategy file(s).
        signal_mode: "discrete" for trading backtest, "score" for scoring only.
        initial_capital: Starting capital (only used in discrete mode).
        output_file: Path to save results CSV.
        trades_file: Path to save trade log (only used in discrete mode).
        include_rule_scores: Include individual rule scores (only in score mode).

    Returns:
        BacktestResult if signal_mode="discrete", ScoringResult if signal_mode="score".

    Raises:
        InvalidParameterError: If signal_mode is invalid.
    """
    if signal_mode == SignalMode.DISCRETE:
        return run_backtest(
            data_file=data_file,
            strategy_files=strategy_files,
            initial_capital=initial_capital,
            output_file=output_file,
            trades_file=trades_file,
        )
    elif signal_mode == SignalMode.SCORE:
        return run_scoring(
            data_file=data_file,
            strategy_files=strategy_files,
            output_file=output_file,
            include_rule_scores=include_rule_scores,
        )
    else:
        raise InvalidParameterError(
            "signal_mode",
            signal_mode,
            f"must be '{SignalMode.DISCRETE}' or '{SignalMode.SCORE}'",
        )


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="backtest.runner",
        description="Run backtests using trading strategies on indicator data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtest.runner --data SPY_train.csv --strategies strategy.yaml
  python -m backtest.runner -d data.csv -s strat1.yaml strat2.yaml -c 50000
  python -m backtest.runner -d data.csv -s strategy.yaml -o results.csv -t trades.csv

Strategy YAML Format:
  strategies:
    - name: "MyStrategy"
      weight: 0.5
      combine: "all"  # or "any"
      rules:
        - indicator: "rsi_14"
          operator: "<"
          value: 30
          action: "buy"
          strength: 1.0
""",
    )

    parser.add_argument(
        "--data",
        "-d",
        required=True,
        type=str,
        help="Path to CSV file with indicator data",
    )

    parser.add_argument(
        "--strategies",
        "-s",
        required=True,
        nargs="+",
        type=str,
        help="Path(s) to YAML strategy file(s)",
    )

    parser.add_argument(
        "--capital",
        "-c",
        type=float,
        default=100000.0,
        help="Initial capital (default: $100,000)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save daily results CSV",
    )

    parser.add_argument(
        "--trades",
        "-t",
        type=str,
        default=None,
        help="Path to save trade log CSV",
    )

    parser.add_argument(
        "--monthly-report",
        "-m",
        type=str,
        default=None,
        help="Path to save monthly summary CSV",
    )

    parser.add_argument(
        "--yearly-report",
        "-y",
        type=str,
        default=None,
        help="Path to save yearly summary CSV",
    )

    parser.add_argument(
        "--transactions-report",
        type=str,
        default=None,
        help="Path to save detailed transactions CSV",
    )

    parser.add_argument(
        "--show-monthly",
        action="store_true",
        help="Print monthly summary to console",
    )

    parser.add_argument(
        "--show-yearly",
        action="store_true",
        help="Print yearly summary to console",
    )

    parser.add_argument(
        "--signal-mode",
        type=str,
        choices=[SignalMode.DISCRETE, SignalMode.SCORE],
        default=SignalMode.DISCRETE,
        help=f"Signal mode: '{SignalMode.DISCRETE}' for trading (default), "
             f"'{SignalMode.SCORE}' for continuous scoring only",
    )

    parser.add_argument(
        "--include-rule-scores",
        action="store_true",
        help="Include individual rule scores in score mode output",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "minmax", "zscore"],
        default="none",
        help="Normalization for signal_0_1: 'none' (default), 'minmax' (full range), "
             "'zscore' (spread: bearish=0-0.3, neutral=0.4-0.6, bullish=0.7-1.0)",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        args: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    try:
        if parsed_args.signal_mode == SignalMode.SCORE:
            # Score mode: continuous scoring without trading
            result = run_scoring(
                data_file=parsed_args.data,
                strategy_files=parsed_args.strategies,
                output_file=parsed_args.output,
                include_rule_scores=parsed_args.include_rule_scores,
                normalization=parsed_args.normalization,
            )

            # Print score summary
            print("=" * 60)
            print("SCORING ENGINE RESULTS")
            print("=" * 60)
            print(f"\nSignal Raw Score Range: [{result.signal_raw.min():.4f}, {result.signal_raw.max():.4f}]")
            print(f"Signal 0-1 Score Range: [{result.signal_0_1.min():.4f}, {result.signal_0_1.max():.4f}]")
            print(f"\nSignal Raw Statistics:")
            print(result.signal_raw.describe())
            print(f"\nStrategies evaluated: {len(result.strategy_scores)}")
            for ss in result.strategy_scores:
                print(f"  - {ss.strategy_name} (weight: {ss.weight:.2%})")

            if parsed_args.output:
                print(f"\nScores saved to: {parsed_args.output}")

        else:
            # Discrete mode: standard backtest with trading
            result = run_backtest(
                data_file=parsed_args.data,
                strategy_files=parsed_args.strategies,
                initial_capital=parsed_args.capital,
                output_file=parsed_args.output,
                trades_file=parsed_args.trades,
            )

            # Print summary
            print(result.summary())

            # Print monthly/yearly summaries if requested
            if parsed_args.show_monthly:
                print("\n" + result.monthly_summary_text())

            if parsed_args.show_yearly:
                print("\n" + result.yearly_summary_text())

            # Save output files
            if parsed_args.output:
                print(f"\nDaily results saved to: {parsed_args.output}")

            if parsed_args.trades:
                print(f"Trade log saved to: {parsed_args.trades}")

            if parsed_args.monthly_report:
                monthly_df = result.get_monthly_summary()
                monthly_df.to_csv(parsed_args.monthly_report, index=False)
                print(f"Monthly summary saved to: {parsed_args.monthly_report}")

            if parsed_args.yearly_report:
                yearly_df = result.get_yearly_summary()
                yearly_df.to_csv(parsed_args.yearly_report, index=False)
                print(f"Yearly summary saved to: {parsed_args.yearly_report}")

            if parsed_args.transactions_report:
                transactions_df = result.get_transactions()
                if not transactions_df.empty:
                    transactions_df.to_csv(parsed_args.transactions_report, index=False)
                    print(f"Transactions saved to: {parsed_args.transactions_report}")
                else:
                    print("No transactions to save (no trades executed)")

        return 0

    except BacktestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
