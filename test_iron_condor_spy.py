#!/usr/bin/env python3
"""
Test Iron Condor module with real SPY data from credit_spread_screener.
"""

import yfinance as yf
from credit_spread_screener import ScreenerConfig, run_screener
from iron_condor import (
    CreditSpread,
    rank_iron_condors,
    payoff_roi_curve,
)


def main():
    print("=" * 80)
    print("IRON CONDOR TEST - SPY Real Data")
    print("=" * 80)

    # Fetch SPY credit spreads using the screener
    config = ScreenerConfig(
        tickers=["SPY"],
        min_dte=14,
        max_dte=45,
        min_delta=0.08,
        max_delta=0.30,
        max_width=10,
        min_roc=0.15,
    )

    print(f"\nFetching SPY options data...")
    df = run_screener(config)

    if df is None or df.empty:
        print("No spreads returned from screener. Check your data source.")
        return

    # Get underlying price
    try:
        ticker = yf.Ticker("SPY")
        underlying_price = ticker.info.get("regularMarketPrice", ticker.fast_info.get("lastPrice", 600.0))
    except Exception:
        underlying_price = 600.0  # Fallback

    print(f"Underlying: SPY @ ${underlying_price:.2f}")
    print(f"Found {len(df)} total spreads")

    # Separate PCS and CCS
    pcs_df = df[df["type"] == "PCS"]
    ccs_df = df[df["type"] == "CCS"]

    print(f"  - {len(pcs_df)} Put Credit Spreads (PCS)")
    print(f"  - {len(ccs_df)} Call Credit Spreads (CCS)")

    if pcs_df.empty or ccs_df.empty:
        print("\nNeed both PCS and CCS to build Iron Condors.")
        return

    # Convert DataFrame rows to CreditSpread objects
    def row_to_credit_spread(row) -> CreditSpread:
        return CreditSpread(
            underlying=row.get("symbol", "SPY"),
            expiration=row["expiration"],
            spread_type=row["type"],  # Column is "type" not "spread_type"
            short_strike=row["short_strike"],
            long_strike=row["long_strike"],
            credit=row["credit"],
            short_delta=abs(row.get("short_delta", 0.15)),
            bid_ask_spread=row.get("bid_ask_spread", 0.10),
            volume=int(row.get("volume", 100)),
            open_interest=int(row.get("open_interest", 500)),
        )

    put_spreads = [row_to_credit_spread(row) for _, row in pcs_df.iterrows()]
    call_spreads = [row_to_credit_spread(row) for _, row in ccs_df.iterrows()]

    # Get DTE from first spread
    days_to_expiration = int(pcs_df.iloc[0].get("dte", 30)) if not pcs_df.empty else 30

    # Build and rank Iron Condors
    print(f"\nBuilding Iron Condors from {len(put_spreads)} PCS x {len(call_spreads)} CCS...")

    top_condors = rank_iron_condors(
        put_spreads=put_spreads,
        call_spreads=call_spreads,
        underlying_price=underlying_price,
        days_to_expiration=days_to_expiration,
        top_n=10,
    )

    if not top_condors:
        print("No valid Iron Condors found.")
        return

    print(f"\nTop {len(top_condors)} Iron Condors:\n")

    # Print header
    header = (
        f"{'#':>2} | {'ShortP':>6} {'LongP':>6} | {'ShortC':>6} {'LongC':>6} | "
        f"{'Credit':>6} | {'MaxLoss':>7} | {'ROC':>5} | {'POP':>4} | {'SCORE':>5}"
    )
    print(header)
    print("-" * len(header))

    for i, c in enumerate(top_condors, 1):
        row = (
            f"{i:>2} | "
            f"{c.short_put_strike:>6.0f} {c.long_put_strike:>6.0f} | "
            f"{c.short_call_strike:>6.0f} {c.long_call_strike:>6.0f} | "
            f"${c.total_credit:>5.2f} | "
            f"${c.max_loss_dollars:>6.0f} | "
            f"{c.roc_raw:>5.1%} | "
            f"{c.pop:>4.0%} | "
            f"{c.total_score:>5.2f}"
        )
        print(row)

    # Print payoff curve for best condor
    best = top_condors[0]
    print(f"\n{'=' * 60}")
    print(f"BEST IRON CONDOR - Payoff Curve")
    print(f"{'=' * 60}")
    print(f"\nPut Spread: Short ${best.short_put_strike:.0f} / Long ${best.long_put_strike:.0f}")
    print(f"Call Spread: Short ${best.short_call_strike:.0f} / Long ${best.long_call_strike:.0f}")
    print(f"Total Credit: ${best.total_credit:.2f}/share (${best.max_profit_dollars:.0f}/contract)")
    print(f"Max Loss: ${best.max_loss_dollars:.0f}/contract")
    print(f"Breakevens: ${best.breakeven_low:.2f} - ${best.breakeven_high:.2f}")
    print()

    curve = payoff_roi_curve(best, move_low_pct=-0.05, move_high_pct=0.05, step_pct=0.01)

    print(f"{'Move':>6} | {'Price':>8} | {'Payoff':>10} | {'ROI':>8}")
    print("-" * 42)

    for point in curve:
        move_str = f"{point['move_pct']:+.0%}"
        payoff_str = f"${point['payoff']:+,.0f}"
        roi_str = f"{point['roi']:+.1%}"
        print(f"{move_str:>6} | ${point['price']:>7.2f} | {payoff_str:>10} | {roi_str:>8}")

    print()


if __name__ == "__main__":
    main()
