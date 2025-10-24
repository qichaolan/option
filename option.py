#!/usr/bin/env python3
"""
option_pricer.py — A unified, class-based option pricing toolkit.

Implements:
  • Black–Scholes model (European options, continuous dividend yield)
  • Binomial Tree (Cox–Ross–Rubinstein, supports American & European)
  • Intrinsic value computation at expiration

Exports:
  - black_scholes_price(...)
  - binomial_option_price(...)

Usage Examples:
---------------
>>> from option_pricer import OptionPricer, black_scholes_price, binomial_option_price

# --- Black–Scholes (European) ---
>>> black_scholes_price(S=100, K=95, T=1, r=0.05, sigma=0.2, option_type="call")
12.90
>>> black_scholes_price(S=100, K=95, T=1, r=0.05, sigma=0.2, option_type="put")
4.35

# --- Binomial Tree (American) ---
>>> binomial_option_price(S=100, K=95, T=1, r=0.05, sigma=0.2, option_type="call", N=500, style="american")
≈ 12.90
>>> binomial_option_price(S=100, K=95, T=1, r=0.05, sigma=0.2, option_type="put", N=500, style="american")
≈ 4.50

# --- Intrinsic Value ---
# Method 1: via class instance (recommended)
>>> opt = OptionPricer(S=100, K=95, T=1, r=0.05, sigma=0.2, option_type="call")
>>> opt.intrinsic_value(spot=110)
15.0
>>> opt.intrinsic_value(spot=90)
0.0

# Method 2: (optional) via static helper if added later
>>> OptionPricer.intrinsic_value_static("call", spot=110, strike=95)
15.0

Notes:
------
- Time to expiry (T) is in years.
- Rates (r, q) and volatility (sigma) are annualized decimals.
- For T=0 or sigma=0, both pricing models return the forward-discounted intrinsic value.
- The intrinsic_value() method requires an OptionPricer instance,
  since it depends on the object’s strike and option_type.
"""

from __future__ import annotations
from math import log, sqrt, exp, erf, isfinite
from typing import Literal, Optional

OptionType = Literal["call", "put"]
StyleType = Literal["american", "european"]


class OptionPricer:
    """
    Encapsulates option parameters and pricing methods.
    Provides Black–Scholes, Binomial (CRR), and intrinsic value computations.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType = "call",
        q: float = 0.0,
        N: Optional[int] = None,
        style: StyleType = "american",
    ) -> None:
        # --------------------------
        # Input validation
        # --------------------------
        for name, val in (("S", S), ("K", K), ("T", T), ("r", r), ("sigma", sigma), ("q", q)):
            if not isfinite(val):
                raise ValueError(f"{name} must be a finite number.")
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive.")
        if T < 0:
            raise ValueError("T cannot be negative.")
        if sigma < 0:
            raise ValueError("sigma cannot be negative.")
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if style not in ("american", "european"):
            raise ValueError("style must be 'american' or 'european'.")
        if N is not None and N < 1:
            raise ValueError("N must be >= 1 when provided.")

        # --------------------------
        # Core attributes
        # --------------------------
        self.S = float(S)       # spot price
        self.K = float(K)       # strike price
        self.T = float(T)       # time to maturity (years)
        self.r = float(r)       # risk-free rate
        self.sigma = float(sigma)  # volatility
        self.q = float(q)       # dividend yield
        self.option_type = option_type
        self.N = int(N) if N is not None else 500
        self.style = style

    # ---------- Utilities ----------
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    # ---------- Models / Payoffs ----------

    def intrinsic_value(self, spot: Optional[float] = None) -> float:
        """
        Calculate intrinsic value at expiration (T).

        call = max(S_T - K, 0)
        put  = max(K - S_T, 0)

        Parameters
        ----------
        spot : float, optional
            Underlying price at expiration. Defaults to self.S.

        Returns
        -------
        float
            Intrinsic value at expiration.
        """
        S_T = self.S if spot is None else float(spot)
        if not isfinite(S_T) or S_T < 0:
            raise ValueError("Spot price at expiration must be a finite, non-negative number.")

        if self.option_type == "call":
            return max(S_T - self.K, 0.0)
        return max(self.K - S_T, 0.0)

    def black_scholes(self) -> float:
        """
        Compute the European option price using the Black–Scholes–Merton formula.
        Assumes continuous dividend yield (q).

        Returns
        -------
        float
            Present value of the European option.
        """
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q

        # Handle degenerate cases: zero time or zero volatility
        if T == 0.0 or sigma == 0.0:
            forward_pv_diff = S * exp(-q * T) - K * exp(-r * T)
            return max(forward_pv_diff, 0.0) if self.option_type == "call" else max(-forward_pv_diff, 0.0)

        d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if self.option_type == "call":
            return S * exp(-q * T) * self._norm_cdf(d1) - K * exp(-r * T) * self._norm_cdf(d2)
        return K * exp(-r * T) * self._norm_cdf(-d2) - S * exp(-q * T) * self._norm_cdf(-d1)

    def binomial_crr(self) -> float:
        """
        Price an option using the Cox–Ross–Rubinstein (CRR) binomial tree model.
        Supports both American and European styles.

        Returns
        -------
        float
            Option price under the CRR framework.
        """
        S, K, T, r, sigma, q, N = self.S, self.K, self.T, self.r, self.sigma, self.q, self.N

        # Degenerate cases -> forward-discounted intrinsic
        if T == 0.0 or sigma == 0.0:
            forward_pv_diff = S * exp(-q * T) - K * exp(-r * T)
            return max(forward_pv_diff, 0.0) if self.opti
