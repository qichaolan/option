"""
CLI interface for broken-wing condor screener.

Usage:
    python -m broken_wing_condor SPY
    python -m broken_wing_condor SPY --min-dte 30 --max-dte 45 --top 5
    python -m broken_wing_condor SPY --direction bullish --csv output.csv
"""

import argparse
import logging
import sys
from pathlib import Path

from broken_wing_condor.config import CondorConfig, ScoringWeights
from broken_wing_condor.screener import CondorScreener, screen_condors


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="broken_wing_condor",
        description="Screen for broken-wing condor options trades with near-free call spreads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m broken_wing_condor SPY
  python -m broken_wing_condor SPY --min-dte 7 --max-dte 14
  python -m broken_wing_condor SPY --direction bullish --top 5
  python -m broken_wing_condor SPY --csv results.csv
  python -m broken_wing_condor SPY --max-call-cost 0.10 --min-put-credit 0.25

Strategy Overview:
  A broken-wing condor is a neutral-to-bullish strategy that combines:
  - A short put spread (sell higher strike, buy lower strike) for credit
  - A long call spread (buy lower strike, sell higher strike) for upside

  The goal is to collect enough put spread credit to make the call spread
  essentially free (cost <= $0.05), giving free upside potential.
        """,
    )

    # Positional arguments
    parser.add_argument(
        "symbol",
        type=str,
        help="Underlying symbol (e.g., SPY, QQQ, AAPL)",
    )

    # DTE range
    parser.add_argument(
        "--min-dte",
        type=int,
        default=3,
        help="Minimum days to expiration (default: 3)",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=14,
        help="Maximum days to expiration (default: 14)",
    )

    # Strategy parameters
    parser.add_argument(
        "--max-call-cost",
        type=float,
        default=0.05,
        help="Maximum call spread cost in dollars (default: 0.05 for near-free)",
    )
    parser.add_argument(
        "--min-put-credit",
        type=float,
        default=0.30,
        help="Minimum put credit as %% of spread width (default: 0.30 = 30%%)",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=10.0,
        help="Maximum loss per contract in dollars (default: 10.0 = $1000)",
    )

    # Spread widths
    parser.add_argument(
        "--put-width-min",
        type=int,
        default=5,
        help="Minimum put spread width in points (default: 5)",
    )
    parser.add_argument(
        "--put-width-max",
        type=int,
        default=15,
        help="Maximum put spread width in points (default: 15)",
    )
    parser.add_argument(
        "--call-width",
        type=int,
        default=10,
        help="Call spread width in points (default: 10)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.03,
        help="Safety margin: long put must be at least this %% below spot (default: 0.03 = 3%%)",
    )

    # Direction
    parser.add_argument(
        "--direction",
        type=str,
        choices=["neutral", "bullish", "bearish"],
        default="neutral",
        help="Directional bias for filtering (default: neutral)",
    )

    # Results
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of top results to show (default: 10)",
    )

    # Output format
    parser.add_argument(
        "--csv",
        type=str,
        metavar="FILE",
        help="Output results to CSV file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )

    # Scoring weights (advanced)
    parser.add_argument(
        "--weight-risk",
        type=float,
        default=0.25,
        help="Weight for risk score (default: 0.25)",
    )
    parser.add_argument(
        "--weight-credit",
        type=float,
        default=0.20,
        help="Weight for credit score (default: 0.20)",
    )
    parser.add_argument(
        "--weight-skew",
        type=float,
        default=0.20,
        help="Weight for skew score (default: 0.20)",
    )
    parser.add_argument(
        "--weight-call",
        type=float,
        default=0.10,
        help="Weight for call score (default: 0.10)",
    )
    parser.add_argument(
        "--weight-rrr",
        type=float,
        default=0.10,
        help="Weight for risk/reward score (default: 0.10)",
    )
    parser.add_argument(
        "--weight-ev",
        type=float,
        default=0.10,
        help="Weight for expected value score (default: 0.10)",
    )
    parser.add_argument(
        "--weight-pop",
        type=float,
        default=0.05,
        help="Weight for probability of profit score (default: 0.05)",
    )

    # Data source
    parser.add_argument(
        "--yfinance",
        action="store_true",
        help="Use yfinance directly instead of OpenBB",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for CLI."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose, args.debug)

    # Build config
    config = CondorConfig(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        max_call_cost=args.max_call_cost,
        min_put_credit_pct=args.min_put_credit,
        max_loss_per_contract=args.max_loss,
        put_spread_width_min=args.put_width_min,
        put_spread_width_max=args.put_width_max,
        call_spread_width=args.call_width,
        safety_margin_pct=args.safety_margin,
        top_n=args.top,
    )

    # Build weights
    weights = ScoringWeights(
        risk_weight=args.weight_risk,
        credit_weight=args.weight_credit,
        skew_weight=args.weight_skew,
        call_weight=args.weight_call,
        rrr_weight=args.weight_rrr,
        ev_weight=args.weight_ev,
        pop_weight=args.weight_pop,
    )

    # Run screener
    try:
        print(f"Screening {args.symbol} for broken-wing condors...")
        print(f"DTE range: {args.min_dte}-{args.max_dte} days")
        print(f"Direction: {args.direction}")
        print()

        result = screen_condors(
            symbol=args.symbol.upper(),
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            max_call_cost=args.max_call_cost,
            min_put_credit_pct=args.min_put_credit,
            top_n=args.top,
            direction=args.direction,
            weights=weights,
            prefer_openbb=not args.yfinance,
        )

        # Output results
        if args.json:
            import json
            output = {
                "symbol": result.symbol,
                "underlying_price": result.underlying_price,
                "total_candidates": result.total_candidates,
                "expirations_scanned": [e.isoformat() for e in result.expirations_scanned],
                "condors": [
                    rc.to_json_dict(result.symbol)
                    for rc in result.ranked_condors
                ],
            }
            print(json.dumps(output, indent=2))

        elif args.csv:
            csv_content = result.to_csv()
            output_path = Path(args.csv)
            output_path.write_text(csv_content)
            print(f"Results saved to {args.csv}")
            print()
            print(result.to_report())

        else:
            # Default: print report
            print(result.to_report())

        return 0

    except Exception as e:
        logging.exception("Error during screening")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
