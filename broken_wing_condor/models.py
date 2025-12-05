"""
Data models for the broken-wing condor screener.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class OptionLeg:
    """
    Represents a single option leg in the condor structure.

    Attributes:
        contract_symbol: Full option contract symbol
        strike: Strike price
        option_type: 'call' or 'put'
        expiration: Expiration date
        bid: Bid price
        ask: Ask price
        mid: Mid price ((bid + ask) / 2)
        iv: Implied volatility (0-1 scale)
        delta: Option delta
        volume: Trading volume
        open_interest: Open interest
    """
    contract_symbol: str
    strike: float
    option_type: str  # 'call' or 'put'
    expiration: date
    bid: float
    ask: float
    mid: float
    iv: Optional[float] = None
    delta: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    @property
    def is_liquid(self) -> bool:
        """Check if option has reasonable liquidity."""
        if self.open_interest is None:
            return True  # Assume liquid if no data
        return self.open_interest >= 100

    @property
    def spread_width(self) -> float:
        """Bid-ask spread width."""
        return self.ask - self.bid


@dataclass
class PayoffScenario:
    """
    Payoff at a specific price scenario.

    Attributes:
        scenario_name: Description of the scenario
        price_condition: Price condition description
        profit_loss: P/L at this scenario
        is_max_profit: Whether this is a max profit scenario
        is_max_loss: Whether this is a max loss scenario
    """
    scenario_name: str
    price_condition: str
    profit_loss: float
    is_max_profit: bool = False
    is_max_loss: bool = False


@dataclass
class CondorScore:
    """
    Scoring components for a broken-wing condor trade.

    All component scores are normalized to 0-1 range.
    """
    risk_score: float
    credit_score: float
    skew_score: float
    call_score: float
    rrr_score: float
    ev_score: float
    pop_score: float
    final_score: float

    # Raw metrics for transparency
    max_risk: float
    reward_to_risk: float
    put_credit_pct: float
    call_spread_cost: float
    iv_skew: float
    pop: float
    expected_value: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "risk_score": round(self.risk_score, 4),
            "credit_score": round(self.credit_score, 4),
            "skew_score": round(self.skew_score, 4),
            "call_score": round(self.call_score, 4),
            "rrr_score": round(self.rrr_score, 4),
            "ev_score": round(self.ev_score, 4),
            "pop_score": round(self.pop_score, 4),
            "final_score": round(self.final_score, 4),
            "max_risk": round(self.max_risk, 2),
            "reward_to_risk": round(self.reward_to_risk, 2),
            "put_credit_pct": round(self.put_credit_pct, 4),
            "call_spread_cost": round(self.call_spread_cost, 2),
            "iv_skew": round(self.iv_skew, 4),
            "pop": round(self.pop, 4),
            "expected_value": round(self.expected_value, 2),
        }


@dataclass
class BrokenWingCondor:
    """
    Complete broken-wing condor trade structure.

    Structure:
    - Long Put (lowest strike) - protection
    - Short Put (higher strike) - credit
    - Short Call (higher strike) - cap
    - Long Call (highest strike) - upside participation
    """
    # The four legs
    long_put: OptionLeg
    short_put: OptionLeg
    short_call: OptionLeg
    long_call: OptionLeg

    # Computed metrics
    put_spread_credit: float
    call_spread_debit: float
    net_credit: float
    put_spread_width: float
    call_spread_width: float

    # Risk/reward
    max_loss: float
    max_profit_credit_only: float  # If stays between strikes
    max_profit_with_calls: float   # If goes above long call

    # Scoring
    score: Optional[CondorScore] = None

    # Payoff scenarios
    payoffs: Optional[list[PayoffScenario]] = None

    @property
    def symbol(self) -> str:
        """Extract underlying symbol from contract."""
        # Contract symbols typically start with underlying
        return self.long_put.contract_symbol.split()[0] if " " in self.long_put.contract_symbol else "UNKNOWN"

    @property
    def expiration(self) -> date:
        """Expiration date of the condor."""
        return self.long_put.expiration

    @property
    def dte(self) -> int:
        """Days to expiration."""
        return (self.expiration - date.today()).days

    @property
    def is_free_call_spread(self) -> bool:
        """Check if call spread is effectively free (cost <= $0.05)."""
        return self.call_spread_debit <= 0.05

    @property
    def credit_capture_pct(self) -> float:
        """Put credit as percentage of put spread width."""
        if self.put_spread_width == 0:
            return 0.0
        return self.put_spread_credit / self.put_spread_width

    def get_payoff_table(self) -> list[PayoffScenario]:
        """
        Generate payoff table for all scenarios.

        Returns list of PayoffScenario objects for:
        1. Below long put (max loss)
        2. Between put strikes
        3. Between put and call strikes (max profit zone 1)
        4. Between call strikes
        5. Above long call (max profit zone 2)
        """
        if self.payoffs is not None:
            return self.payoffs

        scenarios = []

        # Scenario 1: Below long put - max loss
        scenarios.append(PayoffScenario(
            scenario_name="Below Long Put",
            price_condition=f"Price < ${self.long_put.strike:.2f}",
            profit_loss=-self.max_loss,
            is_max_loss=True,
        ))

        # Scenario 2: Between put strikes - partial loss
        between_puts_pl = self.net_credit * 100  # Just keep credit, no intrinsic
        scenarios.append(PayoffScenario(
            scenario_name="Between Put Strikes",
            price_condition=f"${self.long_put.strike:.2f} < Price < ${self.short_put.strike:.2f}",
            profit_loss=between_puts_pl,
        ))

        # Scenario 3: Between short put and short call - max profit zone 1
        scenarios.append(PayoffScenario(
            scenario_name="Between Put & Call Spreads",
            price_condition=f"${self.short_put.strike:.2f} < Price < ${self.short_call.strike:.2f}",
            profit_loss=self.max_profit_credit_only,
            is_max_profit=True,
        ))

        # Scenario 4: Between call strikes - gaining on calls
        scenarios.append(PayoffScenario(
            scenario_name="Between Call Strikes",
            price_condition=f"${self.short_call.strike:.2f} < Price < ${self.long_call.strike:.2f}",
            profit_loss=self.max_profit_credit_only,  # Still profitable
        ))

        # Scenario 5: Above long call - max profit zone 2
        scenarios.append(PayoffScenario(
            scenario_name="Above Long Call",
            price_condition=f"Price > ${self.long_call.strike:.2f}",
            profit_loss=self.max_profit_with_calls,
            is_max_profit=True,
        ))

        self.payoffs = scenarios
        return scenarios

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "expiration": self.expiration.isoformat(),
            "dte": self.dte,
            "legs": {
                "long_put": {
                    "contract": self.long_put.contract_symbol,
                    "strike": self.long_put.strike,
                    "price": self.long_put.mid,
                    "iv": self.long_put.iv,
                },
                "short_put": {
                    "contract": self.short_put.contract_symbol,
                    "strike": self.short_put.strike,
                    "price": self.short_put.mid,
                    "iv": self.short_put.iv,
                },
                "short_call": {
                    "contract": self.short_call.contract_symbol,
                    "strike": self.short_call.strike,
                    "price": self.short_call.mid,
                    "iv": self.short_call.iv,
                },
                "long_call": {
                    "contract": self.long_call.contract_symbol,
                    "strike": self.long_call.strike,
                    "price": self.long_call.mid,
                    "iv": self.long_call.iv,
                },
            },
            "premium": {
                "put_spread_credit": round(self.put_spread_credit, 2),
                "call_spread_debit": round(self.call_spread_debit, 2),
                "net_credit": round(self.net_credit, 2),
            },
            "risk_reward": {
                "max_loss": round(self.max_loss, 2),
                "max_profit_credit_only": round(self.max_profit_credit_only, 2),
                "max_profit_with_calls": round(self.max_profit_with_calls, 2),
                "put_spread_width": self.put_spread_width,
                "call_spread_width": self.call_spread_width,
                "credit_capture_pct": round(self.credit_capture_pct * 100, 2),
            },
            "payoffs": [
                {
                    "scenario": p.scenario_name,
                    "condition": p.price_condition,
                    "pnl": round(p.profit_loss, 2),
                }
                for p in self.get_payoff_table()
            ],
            "score": self.score.to_dict() if self.score else None,
        }
