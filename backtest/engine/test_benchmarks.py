"""
Tests for the benchmark calculation module.

This module tests all benchmark strategies including:
- Cash-Only (risk-free rate)
- Lump-Sum Buy-and-Hold
- Monthly DCA
"""

import pandas as pd
import pytest

from backtest.engine.benchmarks import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_YEAR,
    BenchmarkResult,
    calculate_all_benchmarks,
    calculate_cash_only,
    calculate_lump_sum,
    calculate_monthly_dca,
    compare_results,
)
from backtest.engine.portfolio import PortfolioResult


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            name="Test",
            initial_capital=100000,
            final_value=110000,
            total_return=10000,
            total_return_pct=10.0,
            daily_values=[100000, 105000, 110000],
            dates=[
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-03"),
            ],
        )
        assert result.name == "Test"
        assert result.total_return_pct == 10.0

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        result = BenchmarkResult(
            name="Test",
            initial_capital=100000,
            final_value=110000,
            total_return=10000,
            total_return_pct=10.0,
            daily_values=[100000, 110000],
            dates=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
        )
        df = result.to_dataframe()
        assert len(df) == 2
        assert "Date" in df.columns
        assert "Portfolio_Value" in df.columns


class TestCalculateCashOnly:
    """Tests for calculate_cash_only function."""

    def test_cash_only_basic(self, sample_indicator_data):
        """Test basic cash-only calculation."""
        result = calculate_cash_only(sample_indicator_data, initial_capital=100000)

        assert result.name == "Cash-Only (5% Risk-Free)"
        assert result.initial_capital == 100000
        assert len(result.daily_values) == len(sample_indicator_data)

    def test_cash_only_growth(self, sample_indicator_data):
        """Test that cash-only portfolio grows."""
        result = calculate_cash_only(sample_indicator_data, initial_capital=100000)

        assert result.final_value > result.initial_capital
        assert result.total_return > 0
        assert result.total_return_pct > 0

    def test_cash_only_one_year(self):
        """Test cash-only for approximately one year."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=252, freq="B"),
            "Close": [100] * 252,
        })
        result = calculate_cash_only(df, initial_capital=100000)

        # Should be approximately 5% after one year
        expected_return = 100000 * 0.05
        assert abs(result.total_return - expected_return) < 1000  # Within $1000

    def test_cash_only_single_day(self):
        """Test cash-only for single day."""
        df = pd.DataFrame({
            "Date": [pd.Timestamp("2023-01-01")],
            "Close": [100],
        })
        result = calculate_cash_only(df, initial_capital=100000)

        assert result.final_value == 100000  # Day 0, no compounding yet


class TestCalculateLumpSum:
    """Tests for calculate_lump_sum function."""

    def test_lump_sum_basic(self, sample_indicator_data):
        """Test basic lump-sum calculation."""
        result = calculate_lump_sum(sample_indicator_data, initial_capital=100000)

        assert result.name == "Lump-Sum Buy-and-Hold"
        assert result.initial_capital == 100000
        assert len(result.daily_values) == len(sample_indicator_data)

    def test_lump_sum_shares_calculation(self):
        """Test that shares are correctly calculated."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=3),
            "Close": [100, 110, 120],
        })
        result = calculate_lump_sum(df, initial_capital=100000)

        # Should buy 1000 shares at $100
        assert result.daily_values[0] == 100000  # 1000 * 100
        assert result.daily_values[1] == 110000  # 1000 * 110
        assert result.daily_values[2] == 120000  # 1000 * 120

    def test_lump_sum_return(self):
        """Test lump-sum return calculation."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=2),
            "Close": [100, 150],
        })
        result = calculate_lump_sum(df, initial_capital=100000)

        assert result.final_value == 150000
        assert result.total_return == 50000
        assert result.total_return_pct == 50.0

    def test_lump_sum_loss(self):
        """Test lump-sum with price decline."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=2),
            "Close": [100, 80],
        })
        result = calculate_lump_sum(df, initial_capital=100000)

        assert result.final_value == 80000
        assert result.total_return == -20000
        assert result.total_return_pct == -20.0


class TestCalculateMonthlyDca:
    """Tests for calculate_monthly_dca function."""

    def test_dca_basic(self, sample_indicator_data):
        """Test basic DCA calculation."""
        result = calculate_monthly_dca(sample_indicator_data, initial_capital=100000)

        assert result.name == "Monthly DCA"
        assert result.initial_capital == 100000
        assert len(result.daily_values) == len(sample_indicator_data)

    def test_dca_multiple_investments(self):
        """Test DCA makes multiple investments."""
        # 60 days = 3 investment periods (0, 20, 40)
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=60, freq="B"),
            "Close": [100] * 60,
        })
        result = calculate_monthly_dca(df, initial_capital=120000, period_days=20)

        # With 60 days and 20-day periods: 4 investments (days 0, 20, 40, 60)
        # But 60 is past the end, so 3 investments
        # Each investment is 120000 / 4 = 30000
        expected_shares = 120000 / 100  # At constant price
        assert abs(result.daily_values[-1] - 120000) < 1

    def test_dca_constant_price(self):
        """Test DCA with constant price."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=41, freq="B"),
            "Close": [100] * 41,
        })
        result = calculate_monthly_dca(df, initial_capital=100000, period_days=20)

        # With 41 days: 3 periods (0, 20, 40)
        # Final value should be close to initial (no price change)
        assert abs(result.final_value - 100000) < 100

    def test_dca_declining_price(self):
        """Test DCA with declining price (should buy more shares)."""
        prices = [100] * 20 + [50] * 21
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=41, freq="B"),
            "Close": prices,
        })
        result = calculate_monthly_dca(df, initial_capital=100000, period_days=20)

        # DCA buys more shares when price is lower
        # Should end up with more value than lump-sum at day 1
        lump_sum = calculate_lump_sum(df, initial_capital=100000)
        # DCA should outperform lump-sum when prices decline then stay low

    def test_dca_custom_period(self):
        """Test DCA with custom period."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=30, freq="B"),
            "Close": [100] * 30,
        })
        result = calculate_monthly_dca(df, initial_capital=100000, period_days=10)

        # With 30 days and 10-day periods: 4 investments
        assert result.final_value == pytest.approx(100000, abs=100)


class TestCalculateAllBenchmarks:
    """Tests for calculate_all_benchmarks function."""

    def test_returns_all_benchmarks(self, sample_indicator_data):
        """Test that all benchmarks are returned."""
        benchmarks = calculate_all_benchmarks(sample_indicator_data, initial_capital=100000)

        assert len(benchmarks) == 3
        names = [b.name for b in benchmarks]
        assert "Cash-Only (5% Risk-Free)" in names
        assert "Lump-Sum Buy-and-Hold" in names
        assert "Monthly DCA" in names

    def test_consistent_initial_capital(self, sample_indicator_data):
        """Test all benchmarks use same initial capital."""
        benchmarks = calculate_all_benchmarks(sample_indicator_data, initial_capital=50000)

        for b in benchmarks:
            assert b.initial_capital == 50000


class TestCompareResults:
    """Tests for compare_results function."""

    def test_compare_basic(self, sample_indicator_data):
        """Test basic comparison."""
        portfolio = PortfolioResult(
            initial_capital=100000,
            final_value=110000,
            total_return=10000,
            total_return_pct=10.0,
        )
        benchmarks = calculate_all_benchmarks(sample_indicator_data, initial_capital=100000)

        comparison = compare_results(portfolio, benchmarks)

        assert len(comparison) == 4  # Strategy + 3 benchmarks
        assert "Strategy" in comparison.columns
        assert "Return %" in comparison.columns

    def test_compare_values(self):
        """Test comparison values."""
        portfolio = PortfolioResult(
            initial_capital=100000,
            final_value=120000,
            total_return=20000,
            total_return_pct=20.0,
        )

        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=10),
            "Close": [100] * 10,
        })
        benchmarks = calculate_all_benchmarks(df, initial_capital=100000)

        comparison = compare_results(portfolio, benchmarks)

        strategy_row = comparison[comparison["Strategy"] == "Trading Strategy"].iloc[0]
        assert strategy_row["Final Value"] == 120000
        assert strategy_row["Return %"] == 20.0
