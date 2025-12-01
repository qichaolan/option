"""
Corrections Finder Module

Detects historical market corrections from price data using OpenBB.

A "correction" is defined as:
- A drop from the most recent all-time high (ATH) to a subsequent lowest point (trough),
- followed by a recovery back to that same ATH level.
- Once we leave an ATH, everything until we get back to that same ATH is considered
  one correction episode, no matter how choppy it is inside.

Author: Quantitative Engineering Team
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Any

import pandas as pd

# Suppress OpenBB warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


def _fetch_price_data(symbol: str, period: str = "max", start_date: str | None = None) -> pd.DataFrame:
    """
    Fetch historical price data from OpenBB.

    Parameters
    ----------
    symbol : str
        The ticker symbol (e.g., "QQQ", "SPY", "AAPL").
    period : str
        The lookback period. Common values: "1y", "5y", "10y", "max".
        Note: "max" uses start_date="1990-01-01" for reliable full history.
    start_date : str, optional
        Explicit start date in "YYYY-MM-DD" format. Overrides period if provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, open, high, low, close, volume.
        Sorted by date ascending.

    Raises
    ------
    ValueError
        If OpenBB returns empty data or required columns are missing.
    """
    from openbb import obb

    # For "max" period, use explicit start_date for reliable full history
    if period == "max" and start_date is None:
        start_date = "1990-01-01"

    if start_date:
        result = obb.equity.price.historical(symbol, start_date=start_date)
    else:
        result = obb.equity.price.historical(symbol, period=period)
    df = result.to_df()

    if df.empty:
        raise ValueError(f"No price data returned for symbol '{symbol}' with period '{period}'.")

    # Reset index if date is in index
    if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Ensure we have the required columns
    required_cols = {"date", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Sort by date ascending
    df = df.sort_values("date").reset_index(drop=True)

    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    return df


def _compute_running_ath(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Compute the running all-time high (ATH) for each row.

    For each row (date t):
    - If today's price > current ATH, update ATH to today's price and date.
    - Otherwise, keep the existing ATH price and date.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'date' and price_col columns. Must be sorted by date ascending.
    price_col : str
        Column name for the price to use (default: "close").

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two new columns:
        - 'running_ath_price': The ATH price as of each date.
        - 'running_ath_date': The date when the ATH was first reached.
    """
    df = df.copy()

    n = len(df)
    running_ath_price = [None] * n
    running_ath_date = [None] * n

    current_ath_price = float("-inf")
    current_ath_date = None

    for i in range(n):
        p = df[price_col].iloc[i]
        current_date = df["date"].iloc[i]

        # Check if today's price is a new ATH
        if pd.notna(p) and p > current_ath_price:
            current_ath_price = p
            current_ath_date = current_date

        running_ath_price[i] = current_ath_price if current_ath_price != float("-inf") else None
        running_ath_date[i] = current_ath_date

    df["running_ath_price"] = running_ath_price
    df["running_ath_date"] = pd.to_datetime(running_ath_date)

    return df


def _compute_drawdowns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Compute the drawdown series relative to the running ATH.

    For each row (date t):
    - drawdown_pct = price / running_ath_price - 1.0
      (0 when at ATH, negative when below ATH)
    - drawdown_abs = running_ath_price - price
      (positive dollar amount below peak)

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'running_ath_price' column (from _compute_running_ath).
    price_col : str
        Column name for the price to use (default: "close").

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two new columns:
        - 'drawdown_pct': Percent drawdown (0 at ATH, negative below).
        - 'drawdown_abs': Absolute drawdown in dollars.
    """
    df = df.copy()

    n = len(df)
    drawdown_pct = [None] * n
    drawdown_abs = [None] * n

    for i in range(n):
        p = df[price_col].iloc[i]
        ath = df["running_ath_price"].iloc[i]

        if pd.notna(p) and pd.notna(ath) and ath > 0:
            dd_pct = p / ath - 1.0
            # Clamp tiny positive values to 0 to avoid floating point noise
            if dd_pct > 0:
                dd_pct = 0.0
            drawdown_pct[i] = dd_pct
            drawdown_abs[i] = ath - p
        else:
            drawdown_pct[i] = None
            drawdown_abs[i] = None

    df["drawdown_pct"] = drawdown_pct
    df["drawdown_abs"] = drawdown_abs

    return df


def _segment_corrections(
    df: pd.DataFrame,
    min_correction_pct: float,
    recovery_tolerance: float,
    symbol: str,
    include_open_corrections: bool,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Segment the time series into correction episodes.

    Key rule: Once we leave an ATH, everything until we get back to that same ATH
    is one correction episode, no matter how choppy it is inside.

    States:
    - AT_ATH: Price is at ATH or making a new ATH; no active episode.
    - IN_DRAWDOWN_FROM_ATH: Price is below the anchor ATH; tracking the deepest trough.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with running ATH and drawdown columns.
    min_correction_pct : float
        Minimum depth (absolute value) to qualify as a correction (e.g., 0.10 for 10%).
    recovery_tolerance : float
        Tolerance for recovery. Price >= peak * (1 - recovery_tolerance) counts as recovered.
    symbol : str
        The ticker symbol for labeling.
    include_open_corrections : bool
        Whether to include corrections that haven't recovered yet.
    price_col : str
        Column name for the price to use (default: "close").

    Returns
    -------
    pd.DataFrame
        One row per detected correction episode with columns:
        symbol, peak_date, peak_price, trough_date, trough_price,
        recovery_date, recovery_price, correction_pct, correction_abs,
        drawdown_days, total_correction_days, status
    """
    corrections: list[dict[str, Any]] = []

    # State machine
    AT_ATH = "AT_ATH"
    IN_DRAWDOWN_FROM_ATH = "IN_DRAWDOWN_FROM_ATH"

    state = AT_ATH

    # Anchor (peak) tracking
    anchor_peak_price: float | None = None
    anchor_peak_date: date | None = None

    # Episode tracking
    episode_trough_price: float | None = None
    episode_trough_date: date | None = None
    episode_max_drawdown_pct: float = 0.0

    n = len(df)

    for i in range(n):
        p = df[price_col].iloc[i]
        current_date = df["date"].iloc[i]
        ath = df["running_ath_price"].iloc[i]
        ath_date = df["running_ath_date"].iloc[i]
        dd = df["drawdown_pct"].iloc[i]

        # Skip rows with missing data
        if pd.isna(p) or pd.isna(ath) or pd.isna(dd):
            continue

        if state == AT_ATH:
            if dd >= 0:
                # We are at or making ATH, update anchor
                anchor_peak_price = ath
                anchor_peak_date = ath_date
            else:
                # We just left the ATH -> start a new drawdown episode
                state = IN_DRAWDOWN_FROM_ATH
                # anchor_peak_price and anchor_peak_date should already be set
                # from previous iterations when we were at ATH
                if anchor_peak_price is None:
                    # Edge case: first data point is already in drawdown
                    anchor_peak_price = ath
                    anchor_peak_date = ath_date

                episode_trough_price = p
                episode_trough_date = current_date
                episode_max_drawdown_pct = dd

        elif state == IN_DRAWDOWN_FROM_ATH:
            # Check if a NEW all-time high has been made (higher than our anchor)
            # This means we've not only recovered but exceeded the old peak
            if ath > anchor_peak_price:
                # Close out the current correction episode (recovered + new ATH)
                if episode_max_drawdown_pct <= -min_correction_pct:
                    peak_dt = pd.to_datetime(anchor_peak_date)
                    trough_dt = pd.to_datetime(episode_trough_date)
                    # Recovery happened at the point where we matched the old ATH
                    # Use current date as recovery (we've exceeded it now)
                    recovery_dt = pd.to_datetime(current_date)

                    corrections.append({
                        "symbol": symbol,
                        "peak_date": peak_dt,
                        "peak_price": anchor_peak_price,
                        "trough_date": trough_dt,
                        "trough_price": episode_trough_price,
                        "recovery_date": recovery_dt,
                        "recovery_price": p,
                        "correction_pct": episode_max_drawdown_pct,
                        "correction_abs": anchor_peak_price - episode_trough_price,
                        "drawdown_days": (trough_dt - peak_dt).days,
                        "total_correction_days": (recovery_dt - peak_dt).days,
                        "status": "closed",
                    })

                # Reset and update anchor to new ATH
                state = AT_ATH
                anchor_peak_price = ath
                anchor_peak_date = ath_date
                episode_trough_price = None
                episode_trough_date = None
                episode_max_drawdown_pct = 0.0
                continue

            # Compute drawdown relative to our ANCHOR peak (not running ATH)
            # This is important: dd from dataframe is vs running ATH, but we need vs anchor
            anchor_dd = p / anchor_peak_price - 1.0

            # Update deepest trough if this is more negative
            if anchor_dd < episode_max_drawdown_pct:
                episode_max_drawdown_pct = anchor_dd
                episode_trough_price = p
                episode_trough_date = current_date

            # Check if we have recovered back to the anchor ATH level
            recovery_threshold = anchor_peak_price * (1 - recovery_tolerance)
            if p >= recovery_threshold:
                # The drawdown episode from this ATH has ended
                # Decide if this is a real correction
                if episode_max_drawdown_pct <= -min_correction_pct:
                    # This is a correction episode
                    peak_dt = pd.to_datetime(anchor_peak_date)
                    trough_dt = pd.to_datetime(episode_trough_date)
                    recovery_dt = pd.to_datetime(current_date)

                    corrections.append({
                        "symbol": symbol,
                        "peak_date": peak_dt,
                        "peak_price": anchor_peak_price,
                        "trough_date": trough_dt,
                        "trough_price": episode_trough_price,
                        "recovery_date": recovery_dt,
                        "recovery_price": p,
                        "correction_pct": episode_max_drawdown_pct,
                        "correction_abs": anchor_peak_price - episode_trough_price,
                        "drawdown_days": (trough_dt - peak_dt).days,
                        "total_correction_days": (recovery_dt - peak_dt).days,
                        "status": "closed",
                    })

                # Reset and return to AT_ATH state
                state = AT_ATH
                # Update anchor to current ATH (which may have changed)
                anchor_peak_price = ath
                anchor_peak_date = ath_date
                episode_trough_price = None
                episode_trough_date = None
                episode_max_drawdown_pct = 0.0

    # End of dataset: check for open corrections
    if state == IN_DRAWDOWN_FROM_ATH and include_open_corrections:
        if episode_max_drawdown_pct <= -min_correction_pct:
            peak_dt = pd.to_datetime(anchor_peak_date)
            trough_dt = pd.to_datetime(episode_trough_date)

            corrections.append({
                "symbol": symbol,
                "peak_date": peak_dt,
                "peak_price": anchor_peak_price,
                "trough_date": trough_dt,
                "trough_price": episode_trough_price,
                "recovery_date": None,
                "recovery_price": None,
                "correction_pct": episode_max_drawdown_pct,
                "correction_abs": anchor_peak_price - episode_trough_price,
                "drawdown_days": (trough_dt - peak_dt).days,
                "total_correction_days": None,
                "status": "open",
            })

    # Convert to DataFrame
    if not corrections:
        return pd.DataFrame(columns=[
            "symbol", "peak_date", "peak_price", "trough_date", "trough_price",
            "recovery_date", "recovery_price", "correction_pct", "correction_abs",
            "drawdown_days", "total_correction_days", "status"
        ])

    result_df = pd.DataFrame(corrections)
    return result_df


def find_corrections(
    symbol: str,
    period: str = "max",
    min_correction_pct: float = 0.10,
    price_col: str = "close",
    recovery_tolerance: float = 0.001,
    include_open_corrections: bool = True,
    start_date: str | None = None,
) -> pd.DataFrame:
    """
    Find all correction episodes for a given symbol.

    A correction is defined as a drop from the most recent all-time high (ATH) to a
    subsequent lowest point (trough), followed by a recovery back to that same ATH level.
    Once we leave an ATH, everything until we get back to that same ATH is considered
    one correction episode, no matter how choppy it is inside.

    Parameters
    ----------
    symbol : str
        The ticker symbol (e.g., "QQQ", "SPY", "AAPL").
    period : str, default "max"
        The lookback period for historical data. Common values: "1y", "5y", "10y", "max".
    min_correction_pct : float, default 0.10
        Minimum depth (in absolute value) to call an episode a "correction".
        E.g., 0.10 means a 10% or greater drop qualifies.
    price_col : str, default "close"
        Which price column to use for calculations.
    recovery_tolerance : float, default 0.001
        Tolerance for recovery. Price >= peak * (1 - recovery_tolerance) counts as recovered.
        E.g., 0.001 means within 0.1% of the peak.
    include_open_corrections : bool, default True
        Whether to include corrections that haven't recovered yet (still in drawdown).

    Returns
    -------
    pd.DataFrame
        One row per detected correction episode with columns:
        - symbol: The ticker symbol.
        - peak_date: Date of the ATH before the correction.
        - peak_price: Price at the ATH.
        - trough_date: Date of the lowest point during the correction.
        - trough_price: Price at the lowest point.
        - recovery_date: Date when price recovered to ATH (None if open).
        - recovery_price: Price at recovery (None if open).
        - correction_pct: Peak-to-trough percent drop (negative number).
        - correction_abs: Peak-to-trough dollar drop (positive number).
        - drawdown_days: Days from peak to trough.
        - total_correction_days: Days from peak to recovery (None if open).
        - status: "closed" or "open".

    Raises
    ------
    ValueError
        If the symbol returns no data or required columns are missing.

    Examples
    --------
    >>> corrections = find_corrections("QQQ", period="max", min_correction_pct=0.10)
    >>> print(corrections[["peak_date", "trough_date", "correction_pct", "status"]])
    """
    # Step 0: Fetch and validate data
    df = _fetch_price_data(symbol, period, start_date=start_date)

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found. Available: {list(df.columns)}")

    # Drop rows with missing prices
    df = df.dropna(subset=[price_col]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid price data after dropping missing values for '{symbol}'.")

    # Step 1: Compute running ATH
    df = _compute_running_ath(df, price_col)

    # Step 2: Compute drawdowns
    df = _compute_drawdowns(df, price_col)

    # Step 3: Segment into correction episodes
    corrections_df = _segment_corrections(
        df=df,
        min_correction_pct=min_correction_pct,
        recovery_tolerance=recovery_tolerance,
        symbol=symbol,
        include_open_corrections=include_open_corrections,
        price_col=price_col,
    )

    # Attach metadata about the analysis
    corrections_df.attrs["data_points"] = len(df)
    corrections_df.attrs["start_date"] = df["date"].min()
    corrections_df.attrs["end_date"] = df["date"].max()
    corrections_df.attrs["symbol"] = symbol

    return corrections_df


def format_corrections_report(corrections_df: pd.DataFrame) -> str:
    """
    Format corrections DataFrame into a readable report string.

    Parameters
    ----------
    corrections_df : pd.DataFrame
        Output from find_corrections().

    Returns
    -------
    str
        Formatted report string.
    """
    if corrections_df.empty:
        return "No corrections found matching the criteria."

    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"CORRECTIONS REPORT")
    lines.append(f"{'='*80}")
    lines.append(f"Total corrections found: {len(corrections_df)}")
    lines.append("")

    for idx, row in corrections_df.iterrows():
        peak_date = row["peak_date"].strftime("%Y-%m-%d") if pd.notna(row["peak_date"]) else "N/A"
        trough_date = row["trough_date"].strftime("%Y-%m-%d") if pd.notna(row["trough_date"]) else "N/A"
        recovery_date = row["recovery_date"].strftime("%Y-%m-%d") if pd.notna(row["recovery_date"]) else "N/A"

        lines.append(f"Correction #{idx + 1} [{row['status'].upper()}]")
        lines.append(f"  Peak:     {peak_date} @ ${row['peak_price']:.2f}")
        lines.append(f"  Trough:   {trough_date} @ ${row['trough_price']:.2f}")
        lines.append(f"  Recovery: {recovery_date} @ ${row['recovery_price']:.2f}" if pd.notna(row["recovery_price"]) else f"  Recovery: {recovery_date}")
        lines.append(f"  Drop:     {row['correction_pct']*100:.2f}% (${row['correction_abs']:.2f})")
        lines.append(f"  Duration: {row['drawdown_days']} days to trough, {row['total_correction_days']} days total" if pd.notna(row["total_correction_days"]) else f"  Duration: {row['drawdown_days']} days to trough, still open")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Find historical market corrections for a given symbol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "symbol",
        type=str,
        help="Ticker symbol (e.g., QQQ, SPY, AAPL)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="Lookback period (e.g., 1y, 5y, 10y, max). 'max' fetches from 1990-01-01",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        dest="start_date",
        help="Explicit start date (YYYY-MM-DD). Overrides --period if provided",
    )
    parser.add_argument(
        "--min-correction",
        type=float,
        default=0.05,
        dest="min_correction_pct",
        help="Minimum correction depth as decimal (e.g., 0.10 for 10%%)",
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="close",
        dest="price_col",
        help="Price column to use for calculations",
    )
    parser.add_argument(
        "--recovery-tolerance",
        type=float,
        default=0.001,
        dest="recovery_tolerance",
        help="Tolerance for recovery (e.g., 0.001 means within 0.1%% of peak)",
    )
    parser.add_argument(
        "--exclude-open",
        action="store_true",
        dest="exclude_open",
        help="Exclude corrections that haven't recovered yet",
    )

    args = parser.parse_args()

    symbol = args.symbol.upper()
    period = args.period
    start_date = args.start_date
    min_correction_pct = args.min_correction_pct
    include_open_corrections = not args.exclude_open

    if start_date:
        print(f"Finding corrections for {symbol} (start_date={start_date}, min_correction={min_correction_pct*100:.0f}%)...")
    else:
        print(f"Finding corrections for {symbol} (period={period}, min_correction={min_correction_pct*100:.0f}%)...")
    print()

    try:
        corrections = find_corrections(
            symbol=symbol,
            period=period,
            min_correction_pct=min_correction_pct,
            price_col=args.price_col,
            recovery_tolerance=args.recovery_tolerance,
            include_open_corrections=include_open_corrections,
            start_date=start_date,
        )

        # Print data analysis summary
        data_points = corrections.attrs.get("data_points", "N/A")
        data_start = corrections.attrs.get("start_date")
        data_end = corrections.attrs.get("end_date")
        start_str = data_start.strftime("%Y-%m-%d") if pd.notna(data_start) else "N/A"
        end_str = data_end.strftime("%Y-%m-%d") if pd.notna(data_end) else "N/A"
        print(f"Analyzed {data_points} data points from {start_str} to {end_str}")
        print()

        # Print full report
        print(format_corrections_report(corrections))

        # Print summary table
        if not corrections.empty:
            print("="*80)
            print("SUMMARY TABLE")
            print("="*80)

            # Format for display
            display_df = corrections.copy()
            display_df["peak_date"] = display_df["peak_date"].dt.strftime("%Y-%m-%d")
            display_df["trough_date"] = display_df["trough_date"].dt.strftime("%Y-%m-%d")
            display_df["recovery_date"] = display_df["recovery_date"].apply(
                lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "N/A"
            )
            display_df["peak_price"] = display_df["peak_price"].apply(lambda x: f"${x:.2f}")
            display_df["trough_price"] = display_df["trough_price"].apply(lambda x: f"${x:.2f}")
            display_df["recovery_price"] = display_df["recovery_price"].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
            )

            # Classify correction type based on depth
            def classify_correction(pct: float) -> str:
                """Classify correction by depth (pct is negative)."""
                if pct > -0.10:
                    return "volatility_dip"
                elif pct > -0.15:
                    return "correction"
                else:
                    return "bear_market"

            display_df["type"] = corrections["correction_pct"].apply(classify_correction)
            display_df["correction_pct"] = display_df["correction_pct"].apply(lambda x: f"{x*100:.1f}%")

            print(display_df[[
                "peak_date", "peak_price", "trough_date", "trough_price",
                "recovery_date", "recovery_price", "correction_pct",
                "drawdown_days", "total_correction_days", "type", "status"
            ]].to_string(index=False))
            print()

            # Top 5 deepest corrections
            print("="*80)
            print("TOP 5 DEEPEST CORRECTIONS")
            print("="*80)
            top_5 = corrections.nsmallest(5, "correction_pct")
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                peak_date = row["peak_date"].strftime("%Y-%m-%d")
                trough_date = row["trough_date"].strftime("%Y-%m-%d")
                print(f"{idx}. {row['correction_pct']*100:.1f}% drop: {peak_date} -> {trough_date} ({row['drawdown_days']} days)")

            # Generate histograms
            print()
            print("="*80)
            print("GENERATING HISTOGRAMS...")
            print("="*80)

            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"{symbol} Correction Analysis", fontsize=14, fontweight="bold")

                # Histogram 1: Correction Percentage (convert to positive for readability)
                correction_pcts = corrections["correction_pct"].abs() * 100
                axes[0].hist(correction_pcts, bins=15, edgecolor="black", alpha=0.7, color="steelblue")
                axes[0].set_xlabel("Correction Depth (%)")
                axes[0].set_ylabel("Frequency")
                axes[0].set_title("Distribution of Correction Depth")
                axes[0].axvline(correction_pcts.mean(), color="red", linestyle="--", label=f"Mean: {correction_pcts.mean():.1f}%")
                axes[0].axvline(correction_pcts.median(), color="orange", linestyle="--", label=f"Median: {correction_pcts.median():.1f}%")
                axes[0].legend()

                # Histogram 2: Drawdown Days (peak to trough)
                drawdown_days = corrections["drawdown_days"]
                axes[1].hist(drawdown_days, bins=15, edgecolor="black", alpha=0.7, color="coral")
                axes[1].set_xlabel("Days")
                axes[1].set_ylabel("Frequency")
                axes[1].set_title("Distribution of Drawdown Duration\n(Peak to Trough)")
                axes[1].axvline(drawdown_days.mean(), color="red", linestyle="--", label=f"Mean: {drawdown_days.mean():.0f} days")
                axes[1].axvline(drawdown_days.median(), color="orange", linestyle="--", label=f"Median: {drawdown_days.median():.0f} days")
                axes[1].legend()

                # Histogram 3: Total Correction Days (peak to recovery)
                # Filter out open corrections (NaN values)
                total_days = corrections["total_correction_days"].dropna()
                if len(total_days) > 0:
                    axes[2].hist(total_days, bins=15, edgecolor="black", alpha=0.7, color="seagreen")
                    axes[2].set_xlabel("Days")
                    axes[2].set_ylabel("Frequency")
                    axes[2].set_title("Distribution of Total Recovery Time\n(Peak to Recovery)")
                    axes[2].axvline(total_days.mean(), color="red", linestyle="--", label=f"Mean: {total_days.mean():.0f} days")
                    axes[2].axvline(total_days.median(), color="orange", linestyle="--", label=f"Median: {total_days.median():.0f} days")
                    axes[2].legend()
                else:
                    axes[2].text(0.5, 0.5, "No closed corrections\n(all still open)", ha="center", va="center", transform=axes[2].transAxes)
                    axes[2].set_title("Distribution of Total Recovery Time\n(Peak to Recovery)")

                plt.tight_layout()

                # Save the figure
                output_file = f"{symbol.lower()}_corrections_histogram.png"
                plt.savefig(output_file, dpi=150, bbox_inches="tight")
                print(f"Histograms saved to: {output_file}")
                plt.close()

            except ImportError:
                print("matplotlib not installed. Skipping histogram generation.")
                print("Install with: pip install matplotlib")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        print("Make sure OpenBB is installed: pip install openbb", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
