"""
CLI entry point for daily scorer.

Usage:
    python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml
    python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml --refresh
    python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml --clear-cache
"""

import argparse
import sys
from typing import List, Optional

from backtest.daily_scorer.exceptions import DailyScorerError
from backtest.daily_scorer.scorer import DailyScorer


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Daily stock scorer with caching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get latest cached score (or None if no cache)
  python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml

  # Refresh: download new data and score unscored days
  python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml --refresh

  # Clear cache and start fresh
  python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml --clear-cache --refresh

  # Use custom cache directory
  python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml --cache-dir ./my_cache

  # Use different normalization
  python -m backtest.daily_scorer --symbol SPY --strategy strategy.yaml --normalization minmax
        """,
    )

    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Stock ticker symbol (e.g., SPY, AAPL)",
    )

    parser.add_argument(
        "--strategy",
        required=True,
        nargs="+",
        help="Path(s) to strategy YAML file(s)",
    )

    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory for cache files (default: ./cache)",
    )

    parser.add_argument(
        "--normalization",
        choices=["none", "minmax", "zscore"],
        default="zscore",
        help="Normalization method for scores (default: zscore)",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=365,
        help="Number of days of historical data (default: 365)",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download new data and score unscored days",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the score cache before running",
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all cached scores",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output the score value",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        # Initialize scorer
        scorer = DailyScorer(
            symbol=args.symbol,
            strategy_files=args.strategy,
            cache_dir=args.cache_dir,
            normalization=args.normalization,
            lookback_days=args.lookback,
        )

        # Clear cache if requested
        if args.clear_cache:
            scorer.clear_cache()
            if not args.quiet:
                print(f"Cleared cache for {args.symbol}")

        # Show all scores if requested
        if args.show_all:
            df = scorer.get_all_scores()
            if df.empty:
                print("No cached scores.")
            else:
                print(df.to_string(index=False))
            return 0

        # Refresh if requested
        if args.refresh:
            if not args.quiet:
                print(f"Refreshing scores for {args.symbol}...")
            result = scorer.refresh()
        else:
            result = scorer.get_latest_score()

        # Output result
        if result is None:
            if args.quiet:
                print("N/A")
            else:
                print(f"No scores available for {args.symbol}. Use --refresh to download and score.")
            return 0

        if args.quiet:
            print(f"{result.signal_0_1:.4f}")
        else:
            print("=" * 50)
            print(f"DAILY SCORE: {args.symbol}")
            print("=" * 50)
            print(f"Date:           {result.date.strftime('%Y-%m-%d')}")
            print(f"Signal Raw:     {result.signal_raw:+.4f}")
            print(f"Signal (0-1):   {result.signal_0_1:.4f}")
            print(f"Status:         {'cached' if result.is_cached else 'new'}")
            print("=" * 50)

            # Interpretation
            if result.signal_0_1 < 0.3:
                interpretation = "BEARISH"
            elif result.signal_0_1 < 0.4:
                interpretation = "SLIGHTLY BEARISH"
            elif result.signal_0_1 < 0.6:
                interpretation = "NEUTRAL"
            elif result.signal_0_1 < 0.7:
                interpretation = "SLIGHTLY BULLISH"
            else:
                interpretation = "BULLISH"

            print(f"Interpretation: {interpretation}")
            print(f"Cache size:     {len(scorer.cache)} scores")

        return 0

    except DailyScorerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
