"""
Tests for the reports module.

Provides comprehensive test coverage for monthly/yearly summaries
and transaction logging functionality.
"""

from datetime import datetime
from typing import List

import pandas as pd
import pytest

from backtest.engine.benchmarks import BenchmarkResult
from backtest.engine.portfolio import PortfolioResult, Trade
from backtest.engine.reports import (
    MonthlyRecord,
    YearlyRecord,
    TransactionRecord,
    ReportResult,
    _get_month_key,
    _get_year_key,
    _safe_return_pct,
    _safe_annualized_return,
    generate_monthly_summary_from_daily,
    generate_yearly_summary_from_daily,
    generate_transaction_log,
    generate_benchmark_monthly_summary,
    generate_benchmark_yearly_summary,
    generate_full_report,
    _monthly_records_to_dataframe,
    _yearly_records_to_dataframe,
    _transaction_records_to_dataframe,
    format_monthly_summary,
    format_yearly_summary,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_dates() -> List[datetime]:
    """Generate sample dates spanning multiple months and years."""
    return [
        # January 2023
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
        datetime(2023, 1, 5),
        # February 2023
        datetime(2023, 2, 1),
        datetime(2023, 2, 2),
        datetime(2023, 2, 3),
        # January 2024
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
        datetime(2024, 1, 4),
    ]


@pytest.fixture
def sample_portfolio_values() -> List[float]:
    """Sample portfolio values corresponding to sample_dates."""
    return [
        100000.0, 101000.0, 102000.0,  # Jan 2023
        102000.0, 103000.0, 104000.0,  # Feb 2023
        110000.0, 111000.0, 112000.0,  # Jan 2024
    ]


@pytest.fixture
def sample_cash_values() -> List[float]:
    """Sample cash values."""
    return [
        0.0, 0.0, 0.0,      # Jan 2023 - fully invested
        0.0, 0.0, 0.0,      # Feb 2023
        50000.0, 50000.0, 50000.0,  # Jan 2024 - partial cash
    ]


@pytest.fixture
def sample_position_values() -> List[float]:
    """Sample stock position values."""
    return [
        100000.0, 101000.0, 102000.0,  # Jan 2023
        102000.0, 103000.0, 104000.0,  # Feb 2023
        60000.0, 61000.0, 62000.0,     # Jan 2024
    ]


@pytest.fixture
def sample_trades() -> List[Trade]:
    """Sample trades spanning multiple months."""
    return [
        Trade(
            date=datetime(2023, 1, 3),
            action="BUY",
            shares=100.0,
            price=1000.0,
            value=100000.0,
            position_before=0.0,
            position_after=100.0,
            cash_before=100000.0,
            cash_after=0.0,
            portfolio_value=100000.0,
            signal_score=0.5,
        ),
        Trade(
            date=datetime(2023, 2, 1),
            action="SELL",
            shares=100.0,
            price=1020.0,
            value=102000.0,
            position_before=100.0,
            position_after=0.0,
            cash_before=0.0,
            cash_after=102000.0,
            portfolio_value=102000.0,
            signal_score=-0.5,
        ),
        Trade(
            date=datetime(2024, 1, 2),
            action="BUY",
            shares=50.0,
            price=1100.0,
            value=55000.0,
            position_before=0.0,
            position_after=50.0,
            cash_before=110000.0,
            cash_after=55000.0,
            portfolio_value=110000.0,
            signal_score=0.4,
        ),
    ]


@pytest.fixture
def sample_portfolio_result(
    sample_dates,
    sample_portfolio_values,
    sample_cash_values,
    sample_position_values,
    sample_trades,
) -> PortfolioResult:
    """Create a sample PortfolioResult."""
    return PortfolioResult(
        dates=sample_dates,
        portfolio_values=sample_portfolio_values,
        cash_values=sample_cash_values,
        position_values=sample_position_values,
        shares_held=[100.0] * 6 + [50.0] * 3,
        signals=["BUY", "HOLD", "HOLD", "SELL", "HOLD", "HOLD", "BUY", "HOLD", "HOLD"],
        scores=[0.5, 0.1, 0.1, -0.5, -0.1, -0.1, 0.4, 0.1, 0.1],
        trades=sample_trades,
        initial_capital=100000.0,
        final_value=112000.0,
        total_return=12000.0,
        total_return_pct=12.0,
        num_trades=3,
        num_buys=2,
        num_sells=1,
    )


@pytest.fixture
def sample_benchmark() -> BenchmarkResult:
    """Create a sample benchmark result."""
    dates = [
        datetime(2023, 1, 3),
        datetime(2023, 1, 4),
        datetime(2023, 1, 5),
        datetime(2023, 2, 1),
        datetime(2023, 2, 2),
        datetime(2023, 2, 3),
    ]
    daily_values = [100000.0, 100500.0, 101000.0, 101500.0, 102000.0, 102500.0]

    return BenchmarkResult(
        name="Lump-Sum Buy-and-Hold",
        initial_capital=100000.0,
        final_value=102500.0,
        total_return=2500.0,
        total_return_pct=2.5,
        daily_values=daily_values,
        dates=dates,
    )


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_month_key(self):
        """Test month key formatting."""
        assert _get_month_key(datetime(2023, 1, 15)) == "2023-01"
        assert _get_month_key(datetime(2024, 12, 31)) == "2024-12"
        assert _get_month_key(datetime(2020, 6, 1)) == "2020-06"

    def test_get_year_key(self):
        """Test year key formatting."""
        assert _get_year_key(datetime(2023, 1, 15)) == "2023"
        assert _get_year_key(datetime(2024, 12, 31)) == "2024"

    def test_safe_return_pct_normal(self):
        """Test normal return calculation."""
        assert abs(_safe_return_pct(110.0, 100.0) - 10.0) < 0.0001
        assert abs(_safe_return_pct(90.0, 100.0) - (-10.0)) < 0.0001
        assert abs(_safe_return_pct(100.0, 100.0) - 0.0) < 0.0001

    def test_safe_return_pct_zero_start(self):
        """Test return calculation with zero start value."""
        assert _safe_return_pct(100.0, 0.0) == 0.0
        assert _safe_return_pct(100.0, -10.0) == 0.0

    def test_safe_annualized_return_normal(self):
        """Test normal annualized return calculation."""
        # 10% return over 365 days = 10% annualized
        result = _safe_annualized_return(0.10, 365)
        assert abs(result - 10.0) < 0.01

        # 10% return over half year = ~21% annualized
        result = _safe_annualized_return(0.10, 182)
        assert result > 20.0

    def test_safe_annualized_return_edge_cases(self):
        """Test annualized return edge cases."""
        # Zero days
        assert _safe_annualized_return(0.10, 0) == 0.0

        # Negative days
        assert _safe_annualized_return(0.10, -5) == 0.0

        # Total loss
        assert _safe_annualized_return(-1.0, 365) == -100.0

        # Very small period (1 day with 1% return)
        result = _safe_annualized_return(0.01, 1)
        assert result > 100  # Should be very high annualized


# ============================================================================
# Test Monthly Summary Generation
# ============================================================================

class TestMonthlySummary:
    """Tests for monthly summary generation."""

    def test_generate_monthly_summary_basic(
        self, sample_dates, sample_portfolio_values, sample_cash_values,
        sample_position_values, sample_trades
    ):
        """Test basic monthly summary generation."""
        records = generate_monthly_summary_from_daily(
            dates=sample_dates,
            portfolio_values=sample_portfolio_values,
            cash_values=sample_cash_values,
            position_values=sample_position_values,
            strategy_name="Test Strategy",
            initial_capital=100000.0,
            trades=sample_trades,
        )

        assert len(records) == 3  # Jan 2023, Feb 2023, Jan 2024

        # Check first month (Jan 2023)
        jan_2023 = records[0]
        assert jan_2023.month == "2023-01"
        assert jan_2023.strategy_name == "Test Strategy"
        assert jan_2023.total_value == 102000.0  # End of month value
        assert jan_2023.stock_value == 102000.0
        assert jan_2023.cash_balance == 0.0
        assert jan_2023.num_trades == 1
        assert jan_2023.num_buys == 1
        assert jan_2023.num_sells == 0

    def test_generate_monthly_summary_returns(
        self, sample_dates, sample_portfolio_values, sample_cash_values,
        sample_position_values
    ):
        """Test monthly return calculations."""
        records = generate_monthly_summary_from_daily(
            dates=sample_dates,
            portfolio_values=sample_portfolio_values,
            cash_values=sample_cash_values,
            position_values=sample_position_values,
            strategy_name="Test",
            initial_capital=100000.0,
            trades=None,
        )

        # Jan 2023: started at 100000, ended at 102000 = 2% cumulative
        assert abs(records[0].cumulative_return_pct - 2.0) < 0.01

        # Feb 2023: ended at 104000 = 4% cumulative from 100000
        assert abs(records[1].cumulative_return_pct - 4.0) < 0.01

    def test_generate_monthly_summary_empty(self):
        """Test with empty data."""
        records = generate_monthly_summary_from_daily(
            dates=[],
            portfolio_values=[],
            cash_values=[],
            position_values=[],
            strategy_name="Empty",
            initial_capital=100000.0,
        )
        assert records == []

    def test_generate_monthly_summary_single_day(self):
        """Test with single day of data."""
        records = generate_monthly_summary_from_daily(
            dates=[datetime(2023, 5, 15)],
            portfolio_values=[105000.0],
            cash_values=[5000.0],
            position_values=[100000.0],
            strategy_name="Single Day",
            initial_capital=100000.0,
        )

        assert len(records) == 1
        assert records[0].month == "2023-05"
        assert records[0].total_value == 105000.0

    def test_generate_monthly_summary_no_trades(
        self, sample_dates, sample_portfolio_values, sample_cash_values,
        sample_position_values
    ):
        """Test monthly summary without trades."""
        records = generate_monthly_summary_from_daily(
            dates=sample_dates,
            portfolio_values=sample_portfolio_values,
            cash_values=sample_cash_values,
            position_values=sample_position_values,
            strategy_name="No Trades",
            initial_capital=100000.0,
            trades=None,
        )

        for record in records:
            assert record.num_trades == 0
            assert record.num_buys == 0
            assert record.num_sells == 0


# ============================================================================
# Test Yearly Summary Generation
# ============================================================================

class TestYearlySummary:
    """Tests for yearly summary generation."""

    def test_generate_yearly_summary_basic(
        self, sample_dates, sample_portfolio_values, sample_cash_values,
        sample_position_values, sample_trades
    ):
        """Test basic yearly summary generation."""
        records = generate_yearly_summary_from_daily(
            dates=sample_dates,
            portfolio_values=sample_portfolio_values,
            cash_values=sample_cash_values,
            position_values=sample_position_values,
            strategy_name="Test Strategy",
            initial_capital=100000.0,
            trades=sample_trades,
        )

        assert len(records) == 2  # 2023 and 2024

        # Check 2023
        year_2023 = records[0]
        assert year_2023.year == "2023"
        assert year_2023.strategy_name == "Test Strategy"
        assert year_2023.total_value == 104000.0  # End of 2023
        assert year_2023.num_trades == 2
        assert year_2023.num_buys == 1
        assert year_2023.num_sells == 1

        # Check 2024
        year_2024 = records[1]
        assert year_2024.year == "2024"
        assert year_2024.total_value == 112000.0
        assert year_2024.num_trades == 1

    def test_generate_yearly_summary_annualized_return(
        self, sample_dates, sample_portfolio_values, sample_cash_values,
        sample_position_values
    ):
        """Test annualized return calculation."""
        records = generate_yearly_summary_from_daily(
            dates=sample_dates,
            portfolio_values=sample_portfolio_values,
            cash_values=sample_cash_values,
            position_values=sample_position_values,
            strategy_name="Test",
            initial_capital=100000.0,
        )

        # Both years should have annualized return calculated
        for record in records:
            assert record.annualized_return_pct is not None

    def test_generate_yearly_summary_empty(self):
        """Test with empty data."""
        records = generate_yearly_summary_from_daily(
            dates=[],
            portfolio_values=[],
            cash_values=[],
            position_values=[],
            strategy_name="Empty",
            initial_capital=100000.0,
        )
        assert records == []


# ============================================================================
# Test Transaction Log Generation
# ============================================================================

class TestTransactionLog:
    """Tests for transaction log generation."""

    def test_generate_transaction_log_basic(self, sample_trades):
        """Test basic transaction log generation."""
        records = generate_transaction_log(
            trades=sample_trades,
            strategy_name="Test Strategy",
        )

        assert len(records) == 3

        # Check first transaction
        first = records[0]
        assert first.strategy_name == "Test Strategy"
        assert first.date == datetime(2023, 1, 3)
        assert first.month == "2023-01"
        assert first.year == "2023"
        assert first.side == "BUY"
        assert first.price == 1000.0
        assert first.quantity == 100.0
        assert first.amount == 100000.0
        assert first.total_holding_shares == 100.0
        assert first.cash_balance_after == 0.0
        assert first.total_value_after == 100000.0

    def test_generate_transaction_log_sell(self, sample_trades):
        """Test sell transaction in log."""
        records = generate_transaction_log(
            trades=sample_trades,
            strategy_name="Test",
        )

        # Second trade is a sell
        sell = records[1]
        assert sell.side == "SELL"
        assert sell.total_holding_shares == 0.0  # All sold
        assert sell.cash_balance_after == 102000.0

    def test_generate_transaction_log_empty(self):
        """Test with no trades."""
        records = generate_transaction_log(
            trades=[],
            strategy_name="Empty",
        )
        assert records == []


# ============================================================================
# Test Benchmark Summaries
# ============================================================================

class TestBenchmarkSummaries:
    """Tests for benchmark summary generation."""

    def test_benchmark_monthly_summary(self, sample_benchmark):
        """Test benchmark monthly summary."""
        records = generate_benchmark_monthly_summary(
            benchmark=sample_benchmark,
            initial_capital=100000.0,
        )

        assert len(records) == 2  # Jan and Feb 2023

        # Lump-sum should have all value in stock
        jan = records[0]
        assert jan.strategy_name == "Lump-Sum Buy-and-Hold"
        assert jan.stock_value == jan.total_value
        assert jan.cash_balance == 0.0

    def test_benchmark_monthly_summary_cash_only(self):
        """Test cash-only benchmark monthly summary."""
        benchmark = BenchmarkResult(
            name="Cash-Only (5% Risk-Free)",
            initial_capital=100000.0,
            final_value=105000.0,
            total_return=5000.0,
            total_return_pct=5.0,
            daily_values=[100000.0, 100500.0, 101000.0],
            dates=[datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        )

        records = generate_benchmark_monthly_summary(
            benchmark=benchmark,
            initial_capital=100000.0,
        )

        # Cash-only should have all value in cash
        jan = records[0]
        assert jan.cash_balance == jan.total_value
        assert jan.stock_value == 0.0

    def test_benchmark_yearly_summary(self, sample_benchmark):
        """Test benchmark yearly summary."""
        records = generate_benchmark_yearly_summary(
            benchmark=sample_benchmark,
            initial_capital=100000.0,
        )

        assert len(records) == 1  # Only 2023
        assert records[0].year == "2023"
        assert records[0].strategy_name == "Lump-Sum Buy-and-Hold"


# ============================================================================
# Test Full Report Generation
# ============================================================================

class TestFullReport:
    """Tests for full report generation."""

    def test_generate_full_report_without_transactions(
        self, sample_portfolio_result, sample_benchmark
    ):
        """Test full report without transactions."""
        report = generate_full_report(
            portfolio=sample_portfolio_result,
            benchmarks=[sample_benchmark],
            include_transactions=False,
        )

        assert isinstance(report, ReportResult)
        assert not report.monthly_summary.empty
        assert not report.yearly_summary.empty
        assert report.transactions is None

        # Check that both strategy and benchmark are in monthly summary
        strategies_in_monthly = report.monthly_summary["strategy_name"].unique()
        assert "Trading Strategy" in strategies_in_monthly
        assert "Lump-Sum Buy-and-Hold" in strategies_in_monthly

    def test_generate_full_report_with_transactions(
        self, sample_portfolio_result, sample_benchmark
    ):
        """Test full report with transactions."""
        report = generate_full_report(
            portfolio=sample_portfolio_result,
            benchmarks=[sample_benchmark],
            include_transactions=True,
        )

        assert report.transactions is not None
        assert not report.transactions.empty
        assert len(report.transactions) == 3  # 3 trades

    def test_generate_full_report_multiple_benchmarks(
        self, sample_portfolio_result
    ):
        """Test full report with multiple benchmarks."""
        benchmarks = [
            BenchmarkResult(
                name="Benchmark 1",
                initial_capital=100000.0,
                final_value=105000.0,
                total_return=5000.0,
                total_return_pct=5.0,
                daily_values=[100000.0, 105000.0],
                dates=[datetime(2023, 1, 1), datetime(2023, 1, 2)],
            ),
            BenchmarkResult(
                name="Benchmark 2",
                initial_capital=100000.0,
                final_value=110000.0,
                total_return=10000.0,
                total_return_pct=10.0,
                daily_values=[100000.0, 110000.0],
                dates=[datetime(2023, 1, 1), datetime(2023, 1, 2)],
            ),
        ]

        report = generate_full_report(
            portfolio=sample_portfolio_result,
            benchmarks=benchmarks,
            include_transactions=False,
        )

        strategies = report.monthly_summary["strategy_name"].unique()
        assert len(strategies) == 3  # Trading Strategy + 2 benchmarks


# ============================================================================
# Test DataFrame Conversion
# ============================================================================

class TestDataFrameConversion:
    """Tests for DataFrame conversion functions."""

    def test_monthly_records_to_dataframe(self):
        """Test monthly records to DataFrame conversion."""
        records = [
            MonthlyRecord(
                month="2023-01",
                strategy_name="Test",
                total_value=100000.0,
                stock_value=90000.0,
                cash_balance=10000.0,
                monthly_return_pct=5.0,
                cumulative_return_pct=5.0,
                num_trades=2,
                num_buys=1,
                num_sells=1,
            )
        ]

        df = _monthly_records_to_dataframe(records)

        assert len(df) == 1
        assert df.iloc[0]["month"] == "2023-01"
        assert df.iloc[0]["total_value"] == 100000.0
        assert df.iloc[0]["monthly_return_pct"] == 5.0

    def test_monthly_records_to_dataframe_empty(self):
        """Test empty monthly records."""
        df = _monthly_records_to_dataframe([])
        assert df.empty
        assert "month" in df.columns
        assert "strategy_name" in df.columns

    def test_yearly_records_to_dataframe(self):
        """Test yearly records to DataFrame conversion."""
        records = [
            YearlyRecord(
                year="2023",
                strategy_name="Test",
                total_value=100000.0,
                stock_value=90000.0,
                cash_balance=10000.0,
                total_return_pct=5.0,
                annualized_return_pct=5.2,
                num_trades=10,
                num_buys=5,
                num_sells=5,
            )
        ]

        df = _yearly_records_to_dataframe(records)

        assert len(df) == 1
        assert df.iloc[0]["year"] == "2023"
        assert df.iloc[0]["annualized_return_pct"] == 5.2

    def test_transaction_records_to_dataframe(self):
        """Test transaction records to DataFrame conversion."""
        records = [
            TransactionRecord(
                strategy_name="Test",
                date=datetime(2023, 1, 1),
                month="2023-01",
                year="2023",
                side="BUY",
                price=100.0,
                quantity=10.0,
                amount=1000.0,
                total_holding_shares=10.0,
                cash_balance_after=9000.0,
                total_value_after=10000.0,
            )
        ]

        df = _transaction_records_to_dataframe(records)

        assert len(df) == 1
        assert df.iloc[0]["side"] == "BUY"
        assert df.iloc[0]["amount"] == 1000.0


# ============================================================================
# Test Formatting Functions
# ============================================================================

class TestFormatting:
    """Tests for formatting functions."""

    def test_format_monthly_summary(self, sample_portfolio_result, sample_benchmark):
        """Test monthly summary formatting."""
        report = generate_full_report(
            portfolio=sample_portfolio_result,
            benchmarks=[sample_benchmark],
        )

        formatted = format_monthly_summary(report.monthly_summary)

        assert "MONTHLY PERFORMANCE SUMMARY" in formatted
        assert "Trading Strategy" in formatted
        assert "Lump-Sum Buy-and-Hold" in formatted
        assert "2023-01" in formatted

    def test_format_monthly_summary_empty(self):
        """Test formatting empty monthly summary."""
        df = pd.DataFrame(columns=[
            "month", "strategy_name", "total_value", "stock_value",
            "cash_balance", "monthly_return_pct", "cumulative_return_pct",
            "num_trades", "num_buys", "num_sells"
        ])

        formatted = format_monthly_summary(df)
        assert "No monthly data available" in formatted

    def test_format_yearly_summary(self, sample_portfolio_result, sample_benchmark):
        """Test yearly summary formatting."""
        report = generate_full_report(
            portfolio=sample_portfolio_result,
            benchmarks=[sample_benchmark],
        )

        formatted = format_yearly_summary(report.yearly_summary)

        assert "YEARLY PERFORMANCE SUMMARY" in formatted
        assert "Trading Strategy" in formatted
        assert "2023" in formatted

    def test_format_yearly_summary_empty(self):
        """Test formatting empty yearly summary."""
        df = pd.DataFrame(columns=[
            "year", "strategy_name", "total_value", "stock_value",
            "cash_balance", "total_return_pct", "annualized_return_pct",
            "num_trades", "num_buys", "num_sells"
        ])

        formatted = format_yearly_summary(df)
        assert "No yearly data available" in formatted


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_month_data(self):
        """Test with data spanning only one month."""
        dates = [datetime(2023, 5, i) for i in range(1, 11)]
        values = [100000.0 + i * 100 for i in range(10)]

        records = generate_monthly_summary_from_daily(
            dates=dates,
            portfolio_values=values,
            cash_values=[0.0] * 10,
            position_values=values,
            strategy_name="Single Month",
            initial_capital=100000.0,
        )

        assert len(records) == 1
        assert records[0].month == "2023-05"

    def test_year_spanning_data(self):
        """Test with data spanning year boundary."""
        dates = [
            datetime(2022, 12, 28),
            datetime(2022, 12, 29),
            datetime(2022, 12, 30),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
        ]
        values = [100000.0, 101000.0, 102000.0, 103000.0, 104000.0, 105000.0]

        yearly = generate_yearly_summary_from_daily(
            dates=dates,
            portfolio_values=values,
            cash_values=[0.0] * 6,
            position_values=values,
            strategy_name="Year Span",
            initial_capital=100000.0,
        )

        assert len(yearly) == 2
        assert yearly[0].year == "2022"
        assert yearly[1].year == "2023"

    def test_negative_returns(self):
        """Test with negative returns."""
        dates = [datetime(2023, 1, i) for i in range(1, 4)]
        values = [100000.0, 90000.0, 80000.0]  # 20% loss

        records = generate_monthly_summary_from_daily(
            dates=dates,
            portfolio_values=values,
            cash_values=[0.0] * 3,
            position_values=values,
            strategy_name="Loss",
            initial_capital=100000.0,
        )

        assert records[0].cumulative_return_pct < 0
        assert abs(records[0].cumulative_return_pct - (-20.0)) < 0.01

    def test_large_values(self):
        """Test with large portfolio values."""
        dates = [datetime(2023, 1, i) for i in range(1, 4)]
        values = [1e9, 1.1e9, 1.2e9]  # Billion dollar portfolio

        records = generate_monthly_summary_from_daily(
            dates=dates,
            portfolio_values=values,
            cash_values=[0.0] * 3,
            position_values=values,
            strategy_name="Large",
            initial_capital=1e9,
        )

        assert records[0].total_value == 1.2e9
        assert abs(records[0].cumulative_return_pct - 20.0) < 0.0001

    def test_zero_initial_capital_handling(self):
        """Test handling of zero initial capital."""
        dates = [datetime(2023, 1, 1)]
        values = [100.0]

        # Should not crash with zero initial capital
        records = generate_monthly_summary_from_daily(
            dates=dates,
            portfolio_values=values,
            cash_values=[0.0],
            position_values=values,
            strategy_name="Zero Start",
            initial_capital=0.0,  # Edge case
        )

        # Return calculation should be safe (0% due to safe division)
        assert len(records) == 1
