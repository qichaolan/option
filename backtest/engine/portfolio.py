"""
Portfolio simulation module for backtesting.

This module simulates long-only portfolio trading based on signals,
tracking positions, cash, and generating trade logs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Represents a single trade execution."""

    date: datetime
    action: str  # "BUY" or "SELL"
    shares: float
    price: float
    value: float
    position_before: float
    position_after: float
    cash_before: float
    cash_after: float
    portfolio_value: float
    signal_score: float

    def to_dict(self) -> Dict:
        """Convert trade to dictionary."""
        return {
            "date": self.date,
            "action": self.action,
            "shares": self.shares,
            "price": self.price,
            "value": self.value,
            "position_before": self.position_before,
            "position_after": self.position_after,
            "cash_before": self.cash_before,
            "cash_after": self.cash_after,
            "portfolio_value": self.portfolio_value,
            "signal_score": self.signal_score,
        }


@dataclass
class PortfolioState:
    """Tracks current portfolio state."""

    cash: float
    shares: float = 0.0

    def get_total_value(self, price: float) -> float:
        """Calculate total portfolio value at current price."""
        return self.cash + self.shares * price


@dataclass
class PortfolioResult:
    """Results from portfolio simulation."""

    # Daily tracking
    dates: List[datetime] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    cash_values: List[float] = field(default_factory=list)
    position_values: List[float] = field(default_factory=list)
    shares_held: List[float] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    # Trade log
    trades: List[Trade] = field(default_factory=list)

    # Summary metrics
    initial_capital: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    num_trades: int = 0
    num_buys: int = 0
    num_sells: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert daily tracking to DataFrame."""
        return pd.DataFrame({
            "Date": self.dates,
            "Portfolio_Value": self.portfolio_values,
            "Cash": self.cash_values,
            "Position_Value": self.position_values,
            "Shares": self.shares_held,
            "Signal": self.signals,
            "Score": self.scores,
        })

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trade log to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])


def simulate_portfolio(
    df: pd.DataFrame,
    signals: pd.Series,
    scores: pd.Series,
    initial_capital: float = 100000.0,
) -> PortfolioResult:
    """
    Simulate portfolio based on trading signals.

    Long-only strategy:
    - BUY: Invest all available cash at Close price
    - SELL: Sell entire position at Close price
    - HOLD: No action

    Args:
        df: DataFrame with at least 'Date' and 'Close' columns.
        signals: Series of trading signals ("BUY", "SELL", "HOLD").
        scores: Series of signal scores.
        initial_capital: Starting capital.

    Returns:
        PortfolioResult with daily values and trade log.

    Raises:
        ValueError: If signals/scores length doesn't match data length,
                    or if DataFrame is empty.
    """
    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError("Cannot simulate portfolio on empty DataFrame")

    # Validate input lengths match
    if len(signals) != len(df):
        raise ValueError(
            f"Signals length ({len(signals)}) does not match data length ({len(df)})"
        )
    if len(scores) != len(df):
        raise ValueError(
            f"Scores length ({len(scores)}) does not match data length ({len(df)})"
        )

    result = PortfolioResult(initial_capital=initial_capital)
    state = PortfolioState(cash=initial_capital)

    dates = df["Date"].tolist()
    closes = df["Close"].values
    signals_list = signals.tolist()
    scores_list = scores.tolist()

    for i in range(len(df)):
        date = dates[i]
        price = closes[i]
        signal = signals_list[i]
        score = scores_list[i]

        # Record pre-trade state
        position_before = state.shares
        cash_before = state.cash

        # Execute trade if applicable
        trade = None

        if signal == "BUY" and state.cash > 0:
            # Buy as many shares as possible with available cash
            shares_to_buy = state.cash / price
            trade_value = shares_to_buy * price

            state.shares += shares_to_buy
            # Round cash to avoid floating point accumulation errors
            state.cash = round(state.cash - trade_value, 2)

            trade = Trade(
                date=date,
                action="BUY",
                shares=shares_to_buy,
                price=price,
                value=trade_value,
                position_before=position_before,
                position_after=state.shares,
                cash_before=cash_before,
                cash_after=state.cash,
                portfolio_value=state.get_total_value(price),
                signal_score=score,
            )

        elif signal == "SELL" and state.shares > 0:
            # Sell entire position
            shares_to_sell = state.shares
            trade_value = shares_to_sell * price

            state.shares = 0
            # Round cash to avoid floating point accumulation errors
            state.cash = round(state.cash + trade_value, 2)

            trade = Trade(
                date=date,
                action="SELL",
                shares=shares_to_sell,
                price=price,
                value=trade_value,
                position_before=position_before,
                position_after=state.shares,
                cash_before=cash_before,
                cash_after=state.cash,
                portfolio_value=state.get_total_value(price),
                signal_score=score,
            )

        # Record trade if one occurred
        if trade:
            result.trades.append(trade)

        # Record daily state
        portfolio_value = state.get_total_value(price)
        position_value = state.shares * price

        result.dates.append(date)
        result.portfolio_values.append(portfolio_value)
        result.cash_values.append(state.cash)
        result.position_values.append(position_value)
        result.shares_held.append(state.shares)
        result.signals.append(signal)
        result.scores.append(score)

    # Calculate summary metrics
    result.final_value = result.portfolio_values[-1] if result.portfolio_values else initial_capital
    result.total_return = result.final_value - initial_capital
    result.total_return_pct = (result.total_return / initial_capital) * 100
    result.num_trades = len(result.trades)
    result.num_buys = sum(1 for t in result.trades if t.action == "BUY")
    result.num_sells = sum(1 for t in result.trades if t.action == "SELL")

    return result
