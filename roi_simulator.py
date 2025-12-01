#!/usr/bin/env python3
"""
ROI Simulator - Option ROI Curve Generator

A CLI tool to simulate option ROI over a range of target underlying prices.
Fetches option data via OpenBB and computes ROI curves for specified contracts.

Features:
- Fetch specific option contract data via OpenBB Platform
- Compute ROI at different target underlying prices
- Generate structured table showing ROI curves
- Support for calls and puts
- Plot ROI curves with matplotlib (single or multiple contracts)

Usage:
    python roi_simulator.py QQQ 500 2025-01-17 --range-min 0.1 --range-max 0.8
    python roi_simulator.py NVDA 150 2025-06-20 --range-min -0.2 --range-max 0.5 --option-type put
    python roi_simulator.py SPY 600 2026-01-16 --step-pct 0.02
    python roi_simulator.py QQQ 500 2025-01-17 --plot --plot-output roi_chart.png

Programmatic plotting (multiple contracts):
    from roi_simulator import simulate_roi, plot_roi_curves

    roi1, info1 = simulate_roi("QQQ", 500, "2025-01-17")
    roi2, info2 = simulate_roi("QQQ", 520, "2025-01-17")
    plot_roi_curves(
        [roi1, roi2],
        ["QQQ $500 Call", "QQQ $520 Call"],
        output_path="comparison.png"
    )
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import matplotlib (lazy import in plotting function to avoid import errors if not needed)
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Module-level logger
logger = logging.getLogger(__name__)

# Global flag for OpenBB availability
_USE_OPENBB = False
_obb = None

try:
    from openbb import obb

    _obb = obb
    _USE_OPENBB = True
except ImportError:
    logger.warning("OpenBB not installed, will use yfinance directly")
    _USE_OPENBB = False


# =============================================================================
# CONSTANTS
# =============================================================================

CONTRACT_SIZE = 100  # Standard options contract multiplier


# =============================================================================
# DATA FETCHING
# =============================================================================


def _get_yfinance_option(
    symbol: str,
    strike: float,
    expiration: str,
    option_type: str = "call",
) -> Tuple[Optional[pd.Series], float]:
    """
    Fetch a specific option contract from yfinance.

    Args:
        symbol: Underlying ticker symbol.
        strike: Strike price of the option.
        expiration: Expiration date (YYYY-MM-DD format).
        option_type: Type of option ('call' or 'put').

    Returns:
        Tuple of (option Series or None, underlying price).

    Raises:
        RuntimeError: If yfinance fails to fetch data.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance required. Install with: pip install yfinance")

    logger.info(f"Fetching {option_type} option via yfinance: {symbol} ${strike} exp {expiration}")

    try:
        ticker = yf.Ticker(symbol.upper())

        # Get underlying price
        info = ticker.info
        underlying_price = info.get("regularMarketPrice") or info.get(
            "previousClose", 0
        )

        # Get option chain for the expiration
        opt_chain = ticker.option_chain(expiration)
        if option_type == "call":
            chain = opt_chain.calls
        else:
            chain = opt_chain.puts

        # Find the specific strike
        option = chain[chain["strike"] == strike]

        if option.empty:
            # Try to find closest strike
            available_strikes = chain["strike"].tolist()
            logger.warning(
                f"Strike ${strike} not found. Available strikes: {available_strikes[:10]}..."
            )
            return None, underlying_price

        return option.iloc[0], underlying_price

    except Exception as e:
        raise RuntimeError(f"yfinance failed for {symbol}: {e}") from e


def fetch_option_contract(
    symbol: str,
    strike: float,
    expiration: str,
    option_type: str = "call",
    provider: str = "cboe",
) -> Tuple[Dict[str, Any], float]:
    """
    Fetch a specific option contract.

    Args:
        symbol: Underlying ticker symbol.
        strike: Strike price of the option.
        expiration: Expiration date (YYYY-MM-DD format).
        option_type: Type of option ('call' or 'put').
        provider: Data provider to use.

    Returns:
        Tuple of (option data dict, underlying price).

    Raises:
        ValueError: If the option contract is not found.
    """
    logger.info(f"Fetching {option_type} option: {symbol} ${strike} exp {expiration}")

    underlying_price = 0.0
    option_data = None

    # Try OpenBB first
    if _USE_OPENBB and _obb is not None:
        for prov in [provider, "intrinio", "yfinance"]:
            try:
                logger.debug(f"Trying OpenBB provider: {prov}")
                result = _obb.derivatives.options.chains(
                    symbol=symbol.upper(),
                    provider=prov,
                )
                df = result.to_df()

                if not df.empty:
                    # Extract underlying price if available
                    if "underlying_price" in df.columns:
                        underlying_price = float(df["underlying_price"].iloc[0])

                    # Filter to the specific contract
                    df["expiration_str"] = df["expiration"].astype(str).str[:10]

                    # Filter by option type
                    if "option_type" in df.columns:
                        df = df[df["option_type"] == option_type]

                    # Filter by expiration
                    df = df[df["expiration_str"] == expiration]

                    # Filter by strike
                    df = df[df["strike"] == strike]

                    if not df.empty:
                        option_data = df.iloc[0].to_dict()
                        logger.info(f"Found contract via OpenBB {prov}")
                        break

            except Exception as e:
                logger.debug(f"OpenBB {prov} failed: {e}")
                continue

    # Fallback to direct yfinance
    if option_data is None:
        logger.info("Falling back to direct yfinance")
        option_series, underlying_price = _get_yfinance_option(
            symbol, strike, expiration, option_type
        )

        if option_series is not None:
            option_data = option_series.to_dict()

    if option_data is None:
        raise ValueError(
            f"Option contract not found: {symbol} {option_type} ${strike} exp {expiration}"
        )

    # Get underlying price if not already set
    if underlying_price <= 0:
        underlying_price = _get_underlying_price(symbol)

    return option_data, underlying_price


def _get_underlying_price(symbol: str) -> float:
    """
    Get the current underlying price for a symbol.

    Args:
        symbol: Underlying ticker symbol.

    Returns:
        Current underlying price.

    Raises:
        ValueError: If unable to determine the price.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        for key in ["regularMarketPrice", "previousClose", "ask", "bid"]:
            price = info.get(key)
            if price is not None and price > 0:
                return float(price)

    except Exception as e:
        logger.debug(f"Failed to get underlying price: {e}")

    raise ValueError(f"Could not determine underlying price for {symbol}")


# =============================================================================
# PREMIUM EXTRACTION
# =============================================================================


def extract_premium(option_data: Dict[str, Any]) -> float:
    """
    Extract the premium (price) from option data.

    Args:
        option_data: Dictionary containing option data.

    Returns:
        Premium per share.

    Raises:
        ValueError: If no valid premium can be determined.
    """
    # Priority order for premium selection
    premium_keys = [
        "mark",
        "mid",
        "lastPrice",
        "last_trade_price",
        "last",
        "theoretical_price",
        "close",
    ]

    for key in premium_keys:
        if key in option_data:
            val = option_data[key]
            if val is not None and not pd.isna(val) and val > 0:
                return float(val)

    # Try mid-price from bid/ask
    bid = option_data.get("bid", 0) or 0
    ask = option_data.get("ask", 0) or 0

    if bid > 0 and ask > 0:
        return float((bid + ask) / 2)
    elif bid > 0:
        return float(bid)
    elif ask > 0:
        return float(ask)

    raise ValueError("Could not determine option premium from available data")


# =============================================================================
# ROI SIMULATION
# =============================================================================


def compute_roi_curve(
    strike: float,
    premium: float,
    underlying_price: float,
    option_type: str,
    range_min: float,
    range_max: float,
    step_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Compute ROI curve across a range of target underlying prices.

    The breakeven point is always included in the output. Points are generated
    at regular percentage intervals from range_min to range_max.

    Args:
        strike: Strike price of the option.
        premium: Option premium per share.
        underlying_price: Current underlying price.
        option_type: Type of option ('call' or 'put').
        range_min: Minimum target move as fraction (e.g., 0.1 for +10%).
        range_max: Maximum target move as fraction (e.g., 0.8 for +80%).
        step_pct: Percentage step between price points (e.g., 0.01 for 1%).

    Returns:
        DataFrame with columns: target_pct, target_price, intrinsic, payoff, cost, roi_pct
    """
    # Compute breakeven price and percentage
    breakeven_price = compute_breakeven(strike, premium, option_type)
    breakeven_pct = (breakeven_price / underlying_price) - 1

    # Generate target percentages using step_pct intervals
    # np.arange excludes endpoint, so add a small epsilon to include range_max
    target_pcts = np.arange(range_min, range_max + step_pct / 2, step_pct)

    # Always add breakeven point to the curve
    # Use a small threshold to detect if breakeven is already close to an existing point
    be_threshold = step_pct / 10

    # Check if breakeven is already close to an existing point
    if not any(abs(target_pcts - breakeven_pct) < be_threshold):
        target_pcts = np.sort(np.append(target_pcts, breakeven_pct))

    # Compute target prices
    target_prices = underlying_price * (1 + target_pcts)

    # Compute intrinsic value at each target
    if option_type == "call":
        intrinsic_values = np.maximum(target_prices - strike, 0)
    else:  # put
        intrinsic_values = np.maximum(strike - target_prices, 0)

    # Compute payoff per contract
    payoffs = intrinsic_values * CONTRACT_SIZE

    # Compute cost per contract
    cost = premium * CONTRACT_SIZE

    # Compute ROI as percentage
    roi_pcts = ((payoffs - cost) / cost) * 100

    # Mark breakeven row (use same threshold as insertion)
    is_breakeven = np.abs(target_pcts - breakeven_pct) < be_threshold

    # Build DataFrame
    df = pd.DataFrame(
        {
            "target_pct": target_pcts * 100,  # Convert to percentage
            "target_price": target_prices,
            "intrinsic": intrinsic_values,
            "payoff": payoffs,
            "cost": cost,
            "roi_pct": roi_pcts,
            "is_breakeven": is_breakeven,
        }
    )

    return df


def compute_breakeven(
    strike: float,
    premium: float,
    option_type: str,
) -> float:
    """
    Compute the breakeven price for an option.

    Args:
        strike: Strike price of the option.
        premium: Option premium per share.
        option_type: Type of option ('call' or 'put').

    Returns:
        Breakeven underlying price.
    """
    if option_type == "call":
        return strike + premium
    else:  # put
        return strike - premium


# =============================================================================
# PLOTTING
# =============================================================================

# High-contrast color palette optimized for distinguishing overlapping lines
_DISTINCT_COLORS = [
    "#1f77b4",  # Blue
    "#d62728",  # Red
    "#2ca02c",  # Green
    "#9467bd",  # Purple
    "#ff7f0e",  # Orange
    "#17becf",  # Cyan
    "#e377c2",  # Pink
    "#8c564b",  # Brown
    "#bcbd22",  # Yellow-green
    "#7f7f7f",  # Gray
]

# Marker styles for different contracts (cycle through these)
_MARKER_STYLES = ["o", "s", "^", "D", "v", "P", "X", "p", "h", "*"]

# Line styles for additional differentiation
_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

# Line widths that vary for additional distinction
_LINE_WIDTHS = [3.0, 2.5, 2.8, 2.3, 2.6, 2.4]


def plot_roi_curves(
    roi_data: Union[pd.DataFrame, List[pd.DataFrame]],
    contract_labels: Union[str, List[str]],
    output_path: Optional[str] = None,
    display: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
    symbol: Optional[str] = None,
    option_type: Optional[str] = None,
    expiration: Optional[str] = None,
    underlying_price: Optional[float] = None,
) -> None:
    """
    Plot professional ROI line charts comparing one or more option contracts.

    Generates a trader-friendly visualization where:
    - X-axis (bottom) = target underlying move (% change from current price)
    - X-axis (top) = corresponding target underlying price (if underlying_price provided)
    - Y-axis = ROI (%)

    Features:
    - Distinct line styles, marker shapes, and high-contrast colors for each contract
    - Staggered marker placement to reduce visual overlap
    - Bold break-even reference line with annotation
    - Break-even points marked with diamond markers and labels
    - Minor grid lines for precise reading
    - Legend positioned outside plot area
    - Dynamic title with symbol, option type, expiration, and strike count

    Args:
        roi_data: Single DataFrame or list of DataFrames from compute_roi_curve().
                  Each DataFrame must have 'target_pct' and 'roi_pct' columns.
        contract_labels: Single label string or list of labels for each contract.
                         Used in legend when multiple contracts are plotted.
        output_path: Optional file path to save the plot (PNG format).
                     If None, plot is not saved to file.
        display: If True, display the plot interactively (default: True).
        title: Optional custom title for the plot. If None, auto-generated.
        figsize: Figure size as (width, height) in inches.
        symbol: Underlying ticker symbol (for dynamic title generation).
        option_type: Option type 'call' or 'put' (for dynamic title generation).
        expiration: Expiration date string (for dynamic title generation).
        underlying_price: Current underlying price (for dual x-axis showing prices).

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If roi_data and contract_labels lengths don't match.

    Example:
        # Single contract
        roi_df, info = simulate_roi("QQQ", 500, "2025-01-17")
        plot_roi_curves(roi_df, "QQQ $500 Call", output_path="roi_chart.png")

        # Multiple contracts with full metadata
        roi1, info1 = simulate_roi("QQQ", 500, "2025-01-17")
        roi2, info2 = simulate_roi("QQQ", 520, "2025-01-17")
        plot_roi_curves(
            [roi1, roi2],
            ["$500 Strike", "$520 Strike"],
            symbol="QQQ",
            option_type="call",
            expiration="2025-01-17",
            underlying_price=info1["underlying_price"],
            output_path="comparison.png"
        )
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Normalize inputs to lists
    if isinstance(roi_data, pd.DataFrame):
        roi_data = [roi_data]
    if isinstance(contract_labels, str):
        contract_labels = [contract_labels]

    # Validate inputs
    if len(roi_data) != len(contract_labels):
        raise ValueError(
            f"Number of DataFrames ({len(roi_data)}) must match "
            f"number of labels ({len(contract_labels)})"
        )

    if len(roi_data) == 0:
        raise ValueError("At least one ROI DataFrame is required")

    # Validate DataFrame columns
    required_cols = {"target_pct", "roi_pct"}
    for i, df in enumerate(roi_data):
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame {i} missing required columns: {missing}")

    # Create figure and primary axis
    fig, ax = plt.subplots(figsize=figsize)

    # Track breakeven points for annotation
    breakeven_points = []

    # Calculate number of contracts for styling decisions
    num_contracts = len(roi_data)

    # Plot each contract with distinct styling
    for idx, (df, label) in enumerate(zip(roi_data, contract_labels)):
        # Use high-contrast color palette
        color = _DISTINCT_COLORS[idx % len(_DISTINCT_COLORS)]

        # Vary line style for additional differentiation (especially when >3 lines)
        linestyle = _LINE_STYLES[idx % len(_LINE_STYLES)] if num_contracts > 3 else "-"

        # Vary line width slightly
        linewidth = _LINE_WIDTHS[idx % len(_LINE_WIDTHS)]

        # Different marker for each contract
        marker = _MARKER_STYLES[idx % len(_MARKER_STYLES)]

        # Extract data
        x = df["target_pct"].values  # Already in percentage form
        y = df["roi_pct"].values

        # Stagger markers to reduce overlap: show marker every N points, offset by index
        # This ensures markers don't all appear at the same x positions
        marker_every = max(3, len(x) // 10)  # Show ~10 markers per line
        marker_offset = idx % marker_every   # Offset start position for each line

        # Create marker indices with staggered offset
        marker_indices = list(range(marker_offset, len(x), marker_every))

        # Plot the main line (no markers initially)
        ax.plot(
            x, y,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=3 + idx * 0.1,  # Slight z-order variation
            alpha=0.9,
        )

        # Plot markers separately at staggered positions
        if marker_indices:
            ax.scatter(
                x[marker_indices],
                y[marker_indices],
                color=color,
                marker=marker,
                s=80,  # Marker size
                edgecolors="white",
                linewidths=1.5,
                zorder=4 + idx * 0.1,
            )

        # Find and mark breakeven point (first point where ROI >= 0)
        be_x, be_y = None, None

        # First check for explicit breakeven marker
        if "is_breakeven" in df.columns:
            be_rows = df[df["is_breakeven"]]
            if not be_rows.empty:
                be_x = be_rows["target_pct"].values[0]
                be_y = be_rows["roi_pct"].values[0]

        # Fallback: find first point crossing zero or already at/above zero
        if be_x is None:
            if len(y) > 0 and y[0] >= 0:
                # Already profitable at start
                be_x, be_y = x[0], y[0]
            else:
                for i in range(len(y) - 1):
                    if y[i] < 0 <= y[i + 1]:
                        # Linear interpolation to find exact crossing
                        if y[i + 1] != y[i]:
                            ratio = -y[i] / (y[i + 1] - y[i])
                            be_x = x[i] + ratio * (x[i + 1] - x[i])
                            be_y = 0.0
                        else:
                            be_x = x[i + 1]
                            be_y = y[i + 1]
                        break

        # Plot breakeven diamond marker (larger and more prominent)
        if be_x is not None:
            ax.scatter(
                [be_x], [be_y if be_y is not None else 0],
                color=color,
                s=220,  # Larger size for breakeven
                zorder=6,
                edgecolors="black",
                linewidths=2.5,
                marker="D",
            )
            breakeven_points.append((be_x, be_y if be_y is not None else 0, color, label, idx))

    # -------------------------------------------------------------------------
    # Bold horizontal line at ROI = 0% (break-even reference)
    # -------------------------------------------------------------------------
    ax.axhline(y=0, color="#2C3E50", linestyle="-", linewidth=2.5, zorder=2)

    # Add "Break-even" annotation on the zero line
    x_range = ax.get_xlim()
    ax.annotate(
        "Break-even",
        xy=(x_range[0] + (x_range[1] - x_range[0]) * 0.02, 0),
        xytext=(0, 12),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color="#2C3E50",
        ha="left",
        va="bottom",
    )

    # -------------------------------------------------------------------------
    # Annotate breakeven points with % move (improved positioning)
    # -------------------------------------------------------------------------
    # Sort breakeven points by x position to better stagger annotations
    breakeven_points_sorted = sorted(breakeven_points, key=lambda p: p[0])

    for i, (be_x, be_y, color, _label, _orig_idx) in enumerate(breakeven_points_sorted):
        # Use offset points for consistent annotation positioning
        # Stagger both vertically and horizontally to avoid overlap
        y_offset_pts = 18 + (i % 4) * 16  # 18, 34, 50, 66 points offset
        x_offset_pts = ((i % 3) - 1) * 15  # -15, 0, +15 points offset

        ax.annotate(
            f"BE: {be_x:+.1f}%",
            xy=(be_x, be_y),
            xytext=(x_offset_pts, y_offset_pts),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.95,
                linewidth=1.5,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.5,
                connectionstyle="arc3,rad=0.1" if x_offset_pts != 0 else "arc3,rad=0",
            ),
            zorder=7,
        )

    # -------------------------------------------------------------------------
    # Add vertical line at x = 0 (current price reference)
    # -------------------------------------------------------------------------
    ax.axvline(x=0, color="#7F8C8D", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

    # -------------------------------------------------------------------------
    # Configure major and minor grid lines
    # -------------------------------------------------------------------------
    # Major grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.4, color="#BDC3C7")
    # Minor grid
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3, color="#D5DBDB")

    # Enable minor ticks
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # -------------------------------------------------------------------------
    # Axis labels with enhanced styling
    # -------------------------------------------------------------------------
    ax.set_xlabel("Target Underlying Move (%)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("ROI (%)", fontsize=13, fontweight="bold", labelpad=10)

    # -------------------------------------------------------------------------
    # Format axis tick labels as percentages with +/- signs
    # -------------------------------------------------------------------------
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:+.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:+.0f}%"))

    # Increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=9)

    # -------------------------------------------------------------------------
    # Dual X-axis: show target underlying prices on top (if underlying_price provided)
    # -------------------------------------------------------------------------
    if underlying_price is not None and underlying_price > 0:
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xlabel("Target Underlying Price ($)", fontsize=11, labelpad=8)

        # Convert % move to actual price
        def pct_to_price(pct: float) -> float:
            return underlying_price * (1 + pct / 100)

        def price_formatter(val: float, _) -> str:
            price = pct_to_price(val)
            return f"${price:.0f}"

        ax_top.xaxis.set_major_formatter(plt.FuncFormatter(price_formatter))
        ax_top.tick_params(axis="x", labelsize=10)

    # -------------------------------------------------------------------------
    # Dynamic title generation
    # -------------------------------------------------------------------------
    if title:
        chart_title = title
    else:
        # Build dynamic title from metadata
        title_parts = []

        if symbol:
            title_parts.append(symbol.upper())

        if option_type:
            title_parts.append(option_type.upper())

        title_parts.append("ROI Simulation")

        if expiration:
            title_parts.append(f"— Exp {expiration}")

        if len(roi_data) > 1:
            title_parts.append(f"({len(roi_data)} Strikes)")
        elif len(contract_labels) == 1 and contract_labels[0]:
            # For single contract, include the label in title
            chart_title = f"ROI Curve: {contract_labels[0]}"
            if expiration:
                chart_title += f" — Exp {expiration}"
            title_parts = []  # Clear to use custom title

        chart_title = " ".join(title_parts) if title_parts else chart_title

    ax.set_title(chart_title, fontsize=15, fontweight="bold", pad=20)

    # -------------------------------------------------------------------------
    # Legend: positioned outside the plot on the right
    # -------------------------------------------------------------------------
    if len(contract_labels) > 1 or (len(contract_labels) == 1 and contract_labels[0]):
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            edgecolor="#BDC3C7",
            title="Contracts",
            title_fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

    # -------------------------------------------------------------------------
    # Adjust layout to accommodate legend outside plot
    # -------------------------------------------------------------------------
    plt.tight_layout()
    if len(contract_labels) > 1:
        # Make room for external legend
        fig.subplots_adjust(right=0.78)

    # -------------------------------------------------------------------------
    # Save to file if requested (enhanced export settings)
    # -------------------------------------------------------------------------
    if output_path:
        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )
        logger.info(f"Plot saved to: {output_path}")

    # -------------------------------------------------------------------------
    # Display or close figure
    # -------------------------------------------------------------------------
    if display:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


def format_contract_info(
    symbol: str,
    strike: float,
    expiration: str,
    option_type: str,
    premium: float,
    underlying_price: float,
    breakeven: float,
) -> str:
    """
    Format contract information as a string.

    Args:
        symbol: Underlying ticker symbol.
        strike: Strike price.
        expiration: Expiration date.
        option_type: Option type.
        premium: Option premium.
        underlying_price: Current underlying price.
        breakeven: Breakeven price.

    Returns:
        Formatted string with contract details.
    """
    cost = premium * CONTRACT_SIZE
    be_pct = ((breakeven / underlying_price) - 1) * 100

    lines = [
        "=" * 70,
        f"Option Contract: {symbol.upper()} {option_type.upper()}",
        "=" * 70,
        f"  Strike:           ${strike:.2f}",
        f"  Expiration:       {expiration}",
        f"  Premium:          ${premium:.2f} per share",
        f"  Cost:             ${cost:.2f} per contract",
        f"  Underlying:       ${underlying_price:.2f}",
        f"  Breakeven:        ${breakeven:.2f} ({be_pct:+.2f}% from current)",
        "=" * 70,
    ]

    return "\n".join(lines)


def format_roi_table(df: pd.DataFrame, decimals: int = 2) -> str:
    """
    Format ROI curve DataFrame as a nicely formatted table.

    Breakeven row is marked with '<-- BE' indicator.

    Args:
        df: DataFrame with ROI curve data.
        decimals: Number of decimal places for formatting.

    Returns:
        Formatted table string.
    """
    # Create a copy for formatting
    display_df = df.copy()

    # Format columns with breakeven marker
    def format_target_pct(row):
        pct_str = f"{row['target_pct']:+.1f}%"
        if row.get("is_breakeven", False):
            return f"{pct_str} <-- BE"
        return pct_str

    display_df["target_pct"] = display_df.apply(format_target_pct, axis=1)
    display_df["target_price"] = display_df["target_price"].apply(
        lambda x: f"${x:.2f}"
    )
    display_df["intrinsic"] = display_df["intrinsic"].apply(lambda x: f"${x:.2f}")
    display_df["payoff"] = display_df["payoff"].apply(lambda x: f"${x:.2f}")
    display_df["cost"] = display_df["cost"].apply(lambda x: f"${x:.2f}")
    display_df["roi_pct"] = display_df["roi_pct"].apply(lambda x: f"{x:+.1f}%")

    # Drop is_breakeven column for display
    if "is_breakeven" in display_df.columns:
        display_df = display_df.drop(columns=["is_breakeven"])

    # Rename columns for display
    display_df = display_df.rename(
        columns={
            "target_pct": "Target Move",
            "target_price": "Target Price",
            "intrinsic": "Intrinsic",
            "payoff": "Payoff",
            "cost": "Cost",
            "roi_pct": "ROI",
        }
    )

    return display_df.to_string(index=False)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def simulate_roi(
    symbol: str,
    strike: float,
    expiration: str,
    option_type: str = "call",
    range_min: float = 0.0,
    range_max: float = 0.6,
    step_pct: float = 0.01,
    provider: str = "cboe",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main workflow to simulate ROI for an option contract.

    Args:
        symbol: Underlying ticker symbol.
        strike: Strike price of the option.
        expiration: Expiration date (YYYY-MM-DD).
        option_type: Type of option ('call' or 'put').
        range_min: Minimum target move as fraction.
        range_max: Maximum target move as fraction.
        step_pct: Percentage step between price points (e.g., 0.01 for 1%).
        provider: Data provider to use.

    Returns:
        Tuple of (ROI curve DataFrame, contract info dict).
    """
    # Fetch option contract
    option_data, underlying_price = fetch_option_contract(
        symbol=symbol,
        strike=strike,
        expiration=expiration,
        option_type=option_type,
        provider=provider,
    )

    # Extract premium
    premium = extract_premium(option_data)
    logger.info(f"Premium: ${premium:.2f}")

    # Compute breakeven
    breakeven = compute_breakeven(strike, premium, option_type)

    # Compute ROI curve
    roi_df = compute_roi_curve(
        strike=strike,
        premium=premium,
        underlying_price=underlying_price,
        option_type=option_type,
        range_min=range_min,
        range_max=range_max,
        step_pct=step_pct,
    )

    # Build contract info
    contract_info = {
        "symbol": symbol.upper(),
        "strike": strike,
        "expiration": expiration,
        "option_type": option_type,
        "premium": premium,
        "cost": premium * CONTRACT_SIZE,
        "underlying_price": underlying_price,
        "breakeven": breakeven,
        "breakeven_pct": ((breakeven / underlying_price) - 1) * 100,
    }

    return roi_df, contract_info


# =============================================================================
# CLI INTERFACE
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="ROI Simulator - Compute option ROI curves across target prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s QQQ 500 2025-01-17
  %(prog)s QQQ 500 520 540 2025-01-17 --plot    # Compare multiple strikes
  %(prog)s NVDA 150 2025-06-20 --range-min 0.1 --range-max 1.0
  %(prog)s SPY 600 2026-01-16 --option-type put --step-pct 0.02
  %(prog)s AAPL 200 2025-03-21 --range-min -0.2 --range-max 0.5 --step-pct 0.05

Output:
  Displays a table showing ROI at different target underlying prices.
  The breakeven point is always included and marked with '<-- BE'.
        """,
    )

    # Required positional arguments
    parser.add_argument(
        "symbol",
        type=str,
        help="Underlying ticker symbol (e.g., QQQ, NVDA, SPY)",
    )

    parser.add_argument(
        "strikes",
        type=float,
        nargs="+",
        metavar="STRIKE",
        help="Strike price(s) of the option(s). Multiple strikes can be specified for comparison.",
    )

    parser.add_argument(
        "expiration",
        type=str,
        help="Expiration date in YYYY-MM-DD format",
    )

    # Optional arguments
    parser.add_argument(
        "--option-type",
        type=str,
        choices=["call", "put"],
        default="call",
        help="Option type: call or put (default: call)",
    )

    parser.add_argument(
        "--range-min",
        type=float,
        default=0.0,
        help="Minimum target move as fraction (e.g., 0.1 for +10%%, -0.2 for -20%%)",
    )

    parser.add_argument(
        "--range-max",
        type=float,
        default=0.6,
        help="Maximum target move as fraction (e.g., 0.8 for +80%%)",
    )

    parser.add_argument(
        "--step-pct",
        type=float,
        default=0.01,
        help="Percentage step between price points as fraction (e.g., 0.01 for 1%%, 0.05 for 5%%)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cboe",
        help="OpenBB data provider (default: cboe)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    parser.add_argument(
        "--csv",
        type=str,
        metavar="FILE",
        help="Export ROI curve to CSV file",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display ROI curve plot",
    )

    parser.add_argument(
        "--plot-output",
        type=str,
        metavar="FILE",
        help="Save ROI curve plot to PNG file",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging.

    Args:
        verbose: If True, use DEBUG level.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def main() -> int:
    """
    Main entry point for the ROI Simulator CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Validate expiration date format
        try:
            datetime.strptime(args.expiration, "%Y-%m-%d")
        except ValueError:
            print(
                f"Error: Invalid expiration date format: {args.expiration}. "
                "Use YYYY-MM-DD format.",
                file=sys.stderr,
            )
            return 1

        # Validate range
        if args.range_min >= args.range_max:
            print(
                f"Error: range-min ({args.range_min}) must be less than "
                f"range-max ({args.range_max})",
                file=sys.stderr,
            )
            return 1

        # Process all strikes
        strikes = args.strikes
        roi_dfs = []
        contract_infos = []
        contract_labels = []

        for strike in strikes:
            logger.info(
                f"Simulating ROI for {args.symbol.upper()} {args.option_type} "
                f"${strike} exp {args.expiration}"
            )

            roi_df, contract_info = simulate_roi(
                symbol=args.symbol,
                strike=strike,
                expiration=args.expiration,
                option_type=args.option_type,
                range_min=args.range_min,
                range_max=args.range_max,
                step_pct=args.step_pct,
                provider=args.provider,
            )

            roi_dfs.append(roi_df)
            contract_infos.append(contract_info)
            contract_labels.append(
                f"{contract_info['symbol']} ${contract_info['strike']:.0f} "
                f"{contract_info['option_type'].upper()}"
            )

        # Print contract info for each strike
        for contract_info in contract_infos:
            print(
                format_contract_info(
                    symbol=contract_info["symbol"],
                    strike=contract_info["strike"],
                    expiration=contract_info["expiration"],
                    option_type=contract_info["option_type"],
                    premium=contract_info["premium"],
                    underlying_price=contract_info["underlying_price"],
                    breakeven=contract_info["breakeven"],
                )
            )
            print()

        # Print ROI table for each strike (only if single strike or no plot)
        if len(strikes) == 1 or not (args.plot or args.plot_output):
            for roi_df, contract_info in zip(roi_dfs, contract_infos):
                if len(strikes) > 1:
                    print(f"\n--- Strike ${contract_info['strike']:.0f} ---")
                print("\nROI Curve:")
                print("-" * 70)
                print(format_roi_table(roi_df))
                print("-" * 70)

        # Export to CSV if requested (only for single strike)
        if args.csv:
            if len(strikes) == 1:
                roi_dfs[0].to_csv(args.csv, index=False)
                print(f"\nROI curve exported to: {args.csv}")
            else:
                print("\nWarning: CSV export only supported for single strike", file=sys.stderr)

        # Plot if requested
        if args.plot or args.plot_output:
            # Get underlying price from first contract for dual x-axis
            underlying_price = contract_infos[0]["underlying_price"] if contract_infos else None

            # Generate concise labels for multi-strike comparison
            if len(strikes) > 1:
                plot_labels = [f"${info['strike']:.0f}" for info in contract_infos]
            else:
                plot_labels = contract_labels

            plot_roi_curves(
                roi_data=roi_dfs,
                contract_labels=plot_labels,
                output_path=args.plot_output,
                display=args.plot,
                symbol=args.symbol,
                option_type=args.option_type,
                expiration=args.expiration,
                underlying_price=underlying_price,
            )
            if args.plot_output:
                print(f"\nPlot saved to: {args.plot_output}")

        return 0

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    sys.exit(main())
