"""
Tests for the portfolio simulation module.

This module tests all portfolio simulation functionality including:
- Buy/sell execution
- Position tracking
- Trade logging
- Portfolio value calculation
"""

import pandas as pd
import pytest

from backtest.engine.portfolio import (
    PortfolioResult,
    PortfolioState,
    Trade,
    simulate_portfolio,
)


class TestPortfolioState:
    """Tests for PortfolioState class."""

    def test_initial_state(self):
        """Test initial portfolio state."""
        state = PortfolioState(cash=100000)
        assert state.cash == 100000
        assert state.shares == 0

    def test_total_value_no_shares(self):
        """Test total value with only cash."""
        state = PortfolioState(cash=100000, shares=0)
        assert state.get_total_value(100) == 100000

    def test_total_value_with_shares(self):
        """Test total value with shares."""
        state = PortfolioState(cash=50000, shares=500)
        assert state.get_total_value(100) == 100000  # 50000 + 500*100


class TestTrade:
    """Tests for Trade class."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            date=pd.Timestamp("2023-01-01"),
            action="BUY",
            shares=100,
            price=100.0,
            value=10000.0,
            position_before=0,
            position_after=100,
            cash_before=100000,
            cash_after=90000,
            portfolio_value=100000,
            signal_score=0.5,
        )
        assert trade.action == "BUY"
        assert trade.shares == 100
        assert trade.value == 10000.0

    def test_trade_to_dict(self):
        """Test converting trade to dictionary."""
        trade = Trade(
            date=pd.Timestamp("2023-01-01"),
            action="BUY",
            shares=100,
            price=100.0,
            value=10000.0,
            position_before=0,
            position_after=100,
            cash_before=100000,
            cash_after=90000,
            portfolio_value=100000,
            signal_score=0.5,
        )
        d = trade.to_dict()
        assert d["action"] == "BUY"
        assert d["shares"] == 100
        assert "date" in d


class TestPortfolioResult:
    """Tests for PortfolioResult class."""

    def test_empty_result(self):
        """Test empty portfolio result."""
        result = PortfolioResult()
        assert result.num_trades == 0
        assert result.total_return == 0

    def test_to_dataframe(self):
        """Test converting result to DataFrame."""
        result = PortfolioResult(
            dates=[pd.Timestamp("2023-01-01")],
            portfolio_values=[100000],
            cash_values=[100000],
            position_values=[0],
            shares_held=[0],
            signals=["HOLD"],
            scores=[0],
        )
        df = result.to_dataframe()
        assert len(df) == 1
        assert "Date" in df.columns
        assert "Portfolio_Value" in df.columns

    def test_trades_to_dataframe_empty(self):
        """Test trades DataFrame when empty."""
        result = PortfolioResult()
        df = result.trades_to_dataframe()
        assert df.empty

    def test_trades_to_dataframe(self):
        """Test trades DataFrame with trades."""
        trade = Trade(
            date=pd.Timestamp("2023-01-01"),
            action="BUY",
            shares=100,
            price=100.0,
            value=10000.0,
            position_before=0,
            position_after=100,
            cash_before=100000,
            cash_after=90000,
            portfolio_value=100000,
            signal_score=0.5,
        )
        result = PortfolioResult(trades=[trade])
        df = result.trades_to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["action"] == "BUY"


class TestSimulatePortfolio:
    """Tests for simulate_portfolio function."""

    def test_hold_only(self):
        """Test portfolio with only HOLD signals."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5),
            "Close": [100, 101, 102, 103, 104],
        })
        signals = pd.Series(["HOLD"] * 5)
        scores = pd.Series([0.0] * 5)

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        assert result.num_trades == 0
        assert result.final_value == 100000  # No change
        assert result.total_return == 0

    def test_buy_and_hold(self):
        """Test buy once and hold."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5),
            "Close": [100, 100, 100, 100, 110],  # Price increases at end
        })
        signals = pd.Series(["BUY", "HOLD", "HOLD", "HOLD", "HOLD"])
        scores = pd.Series([0.5, 0, 0, 0, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        assert result.num_trades == 1
        assert result.num_buys == 1
        assert result.shares_held[-1] == 1000  # 100000 / 100
        assert result.final_value == 110000  # 1000 * 110

    def test_buy_and_sell(self):
        """Test buying and selling."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5),
            "Close": [100, 110, 120, 130, 100],
        })
        signals = pd.Series(["BUY", "HOLD", "SELL", "HOLD", "HOLD"])
        scores = pd.Series([0.5, 0, -0.5, 0, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        assert result.num_trades == 2
        assert result.num_buys == 1
        assert result.num_sells == 1
        # Buy at 100, sell at 120: profit = 20%
        assert result.final_value == 120000

    def test_multiple_cycles(self):
        """Test multiple buy/sell cycles."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=6),
            "Close": [100, 110, 105, 115, 110, 120],
        })
        signals = pd.Series(["BUY", "SELL", "BUY", "SELL", "HOLD", "HOLD"])
        scores = pd.Series([0.5, -0.5, 0.5, -0.5, 0, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        assert result.num_trades == 4
        assert result.num_buys == 2
        assert result.num_sells == 2

    def test_buy_signal_no_cash(self):
        """Test BUY signal when no cash available."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=3),
            "Close": [100, 100, 100],
        })
        # Two buy signals in a row
        signals = pd.Series(["BUY", "BUY", "HOLD"])
        scores = pd.Series([0.5, 0.5, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        # Second BUY should not execute (no cash)
        assert result.num_trades == 1

    def test_sell_signal_no_shares(self):
        """Test SELL signal when no shares held."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=3),
            "Close": [100, 100, 100],
        })
        signals = pd.Series(["SELL", "HOLD", "HOLD"])
        scores = pd.Series([-0.5, 0, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        # SELL should not execute (no shares)
        assert result.num_trades == 0

    def test_fractional_shares(self):
        """Test fractional share calculation."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=2),
            "Close": [33.33, 33.33],
        })
        signals = pd.Series(["BUY", "HOLD"])
        scores = pd.Series([0.5, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        # Should buy fractional shares
        expected_shares = 100000 / 33.33
        assert abs(result.shares_held[-1] - expected_shares) < 0.01

    def test_trade_log_values(self):
        """Test trade log contains correct values."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=3),
            "Close": [100, 100, 110],
        })
        signals = pd.Series(["BUY", "HOLD", "SELL"])
        scores = pd.Series([0.5, 0, -0.5])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        assert len(result.trades) == 2

        buy_trade = result.trades[0]
        assert buy_trade.action == "BUY"
        assert buy_trade.price == 100
        assert buy_trade.shares == 1000
        assert buy_trade.position_before == 0
        assert buy_trade.position_after == 1000

        sell_trade = result.trades[1]
        assert sell_trade.action == "SELL"
        assert sell_trade.price == 110
        assert sell_trade.shares == 1000
        assert sell_trade.position_after == 0

    def test_daily_tracking(self):
        """Test daily values are tracked correctly."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=3),
            "Close": [100, 110, 120],
        })
        signals = pd.Series(["BUY", "HOLD", "HOLD"])
        scores = pd.Series([0.5, 0, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        assert len(result.dates) == 3
        assert len(result.portfolio_values) == 3
        assert result.portfolio_values[0] == 100000  # After buying
        assert result.portfolio_values[1] == 110000  # 1000 shares * 110
        assert result.portfolio_values[2] == 120000  # 1000 shares * 120

    def test_return_calculation(self):
        """Test return calculations."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=2),
            "Close": [100, 120],
        })
        signals = pd.Series(["BUY", "HOLD"])
        scores = pd.Series([0.5, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=100000)

        assert result.initial_capital == 100000
        assert result.final_value == 120000
        assert result.total_return == 20000
        assert result.total_return_pct == 20.0

    def test_custom_initial_capital(self):
        """Test with custom initial capital."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=2),
            "Close": [100, 100],
        })
        signals = pd.Series(["BUY", "HOLD"])
        scores = pd.Series([0.5, 0])

        result = simulate_portfolio(df, signals, scores, initial_capital=50000)

        assert result.initial_capital == 50000
        assert result.shares_held[-1] == 500  # 50000 / 100
