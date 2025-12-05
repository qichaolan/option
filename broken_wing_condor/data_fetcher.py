"""
Data fetching module for options chain data.

Supports OpenBB as primary source with yfinance fallback.
"""

import logging
import math
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from broken_wing_condor.models import OptionLeg

logger = logging.getLogger(__name__)


def _safe_int(value, default: int = 0) -> int:
    """Safely convert value to int, handling NaN and None."""
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, handling NaN and None."""
    if value is None:
        return default
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


class OptionsDataFetcher(ABC):
    """Abstract base class for options data fetching."""

    @abstractmethod
    def get_underlying_price(self, symbol: str) -> float:
        """Get current underlying price."""
        pass

    @abstractmethod
    def get_expirations(
        self, symbol: str, min_dte: int, max_dte: int
    ) -> list[date]:
        """Get available expiration dates within DTE range."""
        pass

    @abstractmethod
    def get_options_chain(
        self, symbol: str, expiration: date
    ) -> tuple[list[OptionLeg], list[OptionLeg]]:
        """
        Get options chain for a specific expiration.

        Returns:
            Tuple of (calls, puts) as lists of OptionLeg objects
        """
        pass


class OpenBBFetcher(OptionsDataFetcher):
    """
    Options data fetcher using OpenBB SDK.
    """

    def __init__(self) -> None:
        self._obb = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of OpenBB."""
        if self._initialized:
            return

        try:
            from openbb import obb
            self._obb = obb
            self._initialized = True
            logger.info("OpenBB SDK initialized successfully")
        except ImportError:
            raise ImportError(
                "OpenBB SDK not installed. Install with: pip install openbb"
            )

    def get_underlying_price(self, symbol: str) -> float:
        """Get current underlying price using OpenBB."""
        self._ensure_initialized()

        # Try OpenBB with multiple providers
        providers = ["fmp", "intrinio", "yfinance"]
        price_columns = ["last_price", "close", "price", "regularMarketPrice", "previousClose"]

        for provider in providers:
            try:
                result = self._obb.equity.price.quote(symbol=symbol.upper(), provider=provider)
                df = result.to_df()

                for col in price_columns:
                    if col in df.columns:
                        price = df[col].iloc[0]
                        if price is not None and not pd.isna(price) and price > 0:
                            logger.info(f"Got price for {symbol}: ${price:.2f} via {provider}")
                            return float(price)
            except Exception as e:
                logger.debug(f"OpenBB price via {provider} failed: {e}")
                continue

        # Fallback to yfinance
        logger.warning(f"OpenBB price fetch failed for {symbol}, trying yfinance fallback")
        return self._yfinance_price_fallback(symbol)

    def _yfinance_price_fallback(self, symbol: str) -> float:
        """Fallback to yfinance for price."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return float(info.get("regularMarketPrice", info.get("currentPrice", 0)))
        except Exception as e:
            raise ValueError(f"Could not get price for {symbol}: {e}")

    def get_expirations(
        self, symbol: str, min_dte: int, max_dte: int
    ) -> list[date]:
        """Get available expiration dates within DTE range."""
        self._ensure_initialized()

        today = date.today()
        min_date = today + timedelta(days=min_dte)
        max_date = today + timedelta(days=max_dte)

        # Try OpenBB with multiple providers
        providers = ["cboe", "intrinio", "yfinance"]
        for provider in providers:
            try:
                result = self._obb.derivatives.options.chains(
                    symbol=symbol.upper(),
                    provider=provider,
                )
                df = result.to_df()

                if not df.empty and "expiration" in df.columns:
                    expirations = df["expiration"].unique()
                    valid_dates = []
                    for exp in expirations:
                        if isinstance(exp, str):
                            exp_date = datetime.strptime(exp[:10], "%Y-%m-%d").date()
                        elif isinstance(exp, datetime):
                            exp_date = exp.date()
                        elif isinstance(exp, date):
                            exp_date = exp
                        else:
                            continue

                        if min_date <= exp_date <= max_date:
                            valid_dates.append(exp_date)

                    if valid_dates:
                        logger.info(f"Got {len(valid_dates)} expirations via {provider}")
                        return sorted(valid_dates)
            except Exception as e:
                logger.debug(f"OpenBB expirations via {provider} failed: {e}")
                continue

        # Fallback to yfinance
        logger.warning(f"OpenBB expirations failed for {symbol}, using yfinance fallback")
        return self._yfinance_expirations_fallback(symbol, min_dte, max_dte)

    def _yfinance_expirations_fallback(
        self, symbol: str, min_dte: int, max_dte: int
    ) -> list[date]:
        """Fallback to yfinance for expirations."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            today = date.today()
            min_date = today + timedelta(days=min_dte)
            max_date = today + timedelta(days=max_dte)

            valid_dates = []
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if min_date <= exp_date <= max_date:
                    valid_dates.append(exp_date)

            return sorted(valid_dates)
        except Exception as e:
            logger.error(f"yfinance expirations fallback failed: {e}")
            return []

    def get_options_chain(
        self, symbol: str, expiration: date
    ) -> tuple[list[OptionLeg], list[OptionLeg]]:
        """Get options chain for a specific expiration."""
        self._ensure_initialized()

        # Try OpenBB with multiple providers
        providers = ["cboe", "intrinio", "yfinance"]
        for provider in providers:
            try:
                result = self._obb.derivatives.options.chains(
                    symbol=symbol.upper(),
                    provider=provider,
                )
                df = result.to_df()

                if not df.empty:
                    calls, puts = self._parse_chain_dataframe(df, expiration, symbol)
                    if calls or puts:
                        logger.info(f"Got {len(calls)} calls and {len(puts)} puts via {provider}")
                        return calls, puts
            except Exception as e:
                logger.debug(f"OpenBB chain via {provider} failed: {e}")
                continue

        # Fallback to yfinance
        logger.warning(f"OpenBB chain failed for {symbol}, using yfinance fallback")
        return self._yfinance_chain_fallback(symbol, expiration)

    def _parse_chain_dataframe(
        self, df: pd.DataFrame, expiration: date, symbol: str
    ) -> tuple[list[OptionLeg], list[OptionLeg]]:
        """Parse OpenBB dataframe into OptionLeg objects."""
        calls = []
        puts = []

        # Filter to expiration
        if "expiration" in df.columns:
            exp_str = expiration.isoformat()
            df = df[df["expiration"].astype(str).str[:10] == exp_str]

        for _, row in df.iterrows():
            opt_type = str(row.get("option_type", "")).lower()
            if opt_type not in ("call", "put"):
                continue

            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            # Try multiple column names for last price (varies by provider)
            last_price = _safe_float(
                row.get("last_trade_price") or row.get("lastPrice") or row.get("last") or row.get("close")
            )

            # Calculate mid price - prefer bid/ask, fallback to lastPrice
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            elif last_price > 0:
                mid = last_price
                # Also set bid/ask to lastPrice for consistency
                if bid == 0:
                    bid = last_price
                if ask == 0:
                    ask = last_price
            else:
                mid = 0

            leg = OptionLeg(
                contract_symbol=str(row.get("contract_symbol", f"{symbol}_{row.get('strike', 0)}_{opt_type}")),
                strike=_safe_float(row.get("strike")),
                option_type=opt_type,
                expiration=expiration,
                bid=bid,
                ask=ask,
                mid=mid,
                iv=_safe_float(row.get("implied_volatility")),
                delta=_safe_float(row.get("delta")),
                volume=_safe_int(row.get("volume")),
                open_interest=_safe_int(row.get("open_interest")),
            )

            if opt_type == "call":
                calls.append(leg)
            else:
                puts.append(leg)

        return calls, puts

    def _yfinance_chain_fallback(
        self, symbol: str, expiration: date
    ) -> tuple[list[OptionLeg], list[OptionLeg]]:
        """Fallback to yfinance for options chain."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            exp_str = expiration.strftime("%Y-%m-%d")
            chain = ticker.option_chain(exp_str)

            calls = self._parse_yfinance_chain(chain.calls, "call", expiration, symbol)
            puts = self._parse_yfinance_chain(chain.puts, "put", expiration, symbol)

            return calls, puts
        except Exception as e:
            logger.error(f"yfinance chain fallback failed: {e}")
            return [], []

    def _parse_yfinance_chain(
        self, df: pd.DataFrame, opt_type: str, expiration: date, symbol: str
    ) -> list[OptionLeg]:
        """Parse yfinance dataframe into OptionLeg objects."""
        legs = []

        for _, row in df.iterrows():
            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            last_price = _safe_float(row.get("lastPrice"))

            # Calculate mid price - prefer bid/ask, fallback to lastPrice
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            elif last_price > 0:
                mid = last_price
                # Also set bid/ask to lastPrice for consistency
                if bid == 0:
                    bid = last_price
                if ask == 0:
                    ask = last_price
            else:
                mid = 0

            leg = OptionLeg(
                contract_symbol=str(row.get("contractSymbol", f"{symbol}_{row.get('strike', 0)}_{opt_type}")),
                strike=_safe_float(row.get("strike")),
                option_type=opt_type,
                expiration=expiration,
                bid=bid,
                ask=ask,
                mid=mid,
                iv=_safe_float(row.get("impliedVolatility")),
                delta=None,  # yfinance doesn't provide delta
                volume=_safe_int(row.get("volume")),
                open_interest=_safe_int(row.get("openInterest")),
            )
            legs.append(leg)

        return legs


class YFinanceFetcher(OptionsDataFetcher):
    """
    Options data fetcher using yfinance directly.
    Simpler implementation when OpenBB is not available.
    """

    def __init__(self) -> None:
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError(
                "yfinance not installed. Install with: pip install yfinance"
            )

    def get_underlying_price(self, symbol: str) -> float:
        """Get current underlying price."""
        ticker = self._yf.Ticker(symbol)
        info = ticker.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price is None:
            # Try fast_info
            fast = ticker.fast_info
            price = getattr(fast, "last_price", None)
        if price is None:
            raise ValueError(f"Could not get price for {symbol}")
        return float(price)

    def get_expirations(
        self, symbol: str, min_dte: int, max_dte: int
    ) -> list[date]:
        """Get available expiration dates within DTE range."""
        ticker = self._yf.Ticker(symbol)
        expirations = ticker.options

        today = date.today()
        min_date = today + timedelta(days=min_dte)
        max_date = today + timedelta(days=max_dte)

        valid_dates = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if min_date <= exp_date <= max_date:
                valid_dates.append(exp_date)

        return sorted(valid_dates)

    def get_options_chain(
        self, symbol: str, expiration: date
    ) -> tuple[list[OptionLeg], list[OptionLeg]]:
        """Get options chain for a specific expiration."""
        ticker = self._yf.Ticker(symbol)
        exp_str = expiration.strftime("%Y-%m-%d")

        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            logger.error(f"Failed to get chain for {symbol} {exp_str}: {e}")
            return [], []

        calls = self._parse_chain(chain.calls, "call", expiration, symbol)
        puts = self._parse_chain(chain.puts, "put", expiration, symbol)

        return calls, puts

    def _parse_chain(
        self, df: pd.DataFrame, opt_type: str, expiration: date, symbol: str
    ) -> list[OptionLeg]:
        """Parse yfinance dataframe into OptionLeg objects."""
        legs = []

        for _, row in df.iterrows():
            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            last_price = _safe_float(row.get("lastPrice"))

            # Calculate mid price - prefer bid/ask, fallback to lastPrice
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            elif last_price > 0:
                mid = last_price
                # Also set bid/ask to lastPrice for consistency
                if bid == 0:
                    bid = last_price
                if ask == 0:
                    ask = last_price
            else:
                mid = 0

            leg = OptionLeg(
                contract_symbol=str(row.get("contractSymbol", f"{symbol}_{row.get('strike', 0)}_{opt_type}")),
                strike=_safe_float(row.get("strike")),
                option_type=opt_type,
                expiration=expiration,
                bid=bid,
                ask=ask,
                mid=mid,
                iv=_safe_float(row.get("impliedVolatility")),
                delta=None,
                volume=_safe_int(row.get("volume")),
                open_interest=_safe_int(row.get("openInterest")),
            )
            legs.append(leg)

        return legs


def get_fetcher(prefer_openbb: bool = True) -> OptionsDataFetcher:
    """
    Factory function to get the best available data fetcher.

    Args:
        prefer_openbb: Whether to prefer OpenBB over yfinance

    Returns:
        An OptionsDataFetcher instance
    """
    if prefer_openbb:
        try:
            return OpenBBFetcher()
        except ImportError:
            logger.info("OpenBB not available, falling back to yfinance")
            return YFinanceFetcher()
    else:
        return YFinanceFetcher()
