#!/usr/bin/env python3
"""
Example usage of SECDownloader module.

This script demonstrates both CLI and programmatic usage.
"""

from pathlib import Path
from sec_downloader import SECDownloader, load_config


def example_programmatic_usage():
    """Example: Using SECDownloader programmatically."""
    print("=" * 70)
    print("Example: Programmatic Usage")
    print("=" * 70)

    # Load configuration
    config_path = Path("config/sec_api_config.yaml")
    config = load_config(config_path)

    # Initialize downloader
    downloader = SECDownloader(
        api_key=config['SEC_API_KEY'],
        timeout=config.get('timeout_sec', 30),
        retries=config.get('retries', 3),
        concurrency=config.get('concurrency', 4),
        user_agent=config.get('user_agent', 'SECDownloader/1.0')
    )

    # Search for TSLA filings (no date range needed - gets latest)
    print("\n1. Searching for latest TSLA filings...")
    filings = downloader.search_filings(
        ticker="TSLA",
        limit=5  # Only get 5 latest filings
    )

    print(f"   Found {len(filings)} filings\n")

    # Print filing details
    print("2. Filing details:")
    for i, filing in enumerate(filings, 1):
        print(f"   {i}. {filing['formType']:6} filed {filing['filedAt']} - {filing['companyName']}")

    # Download PDFs
    print("\n3. Downloading PDFs...")
    out_dir = Path("./downloads/examples")
    summary = downloader.download_filings(
        filings=filings,
        out_dir=out_dir
    )

    # Print summary
    print("\n4. Download Summary:")
    print(f"   Found:     {summary.found}")
    print(f"   Attempted: {summary.attempted}")
    print(f"   Succeeded: {summary.succeeded}")
    print(f"   Failed:    {summary.failed}")

    if summary.succeeded > 0:
        print(f"\n   ✓ PDFs saved to: {out_dir}")

    print("=" * 70)


def example_search_multiple_tickers():
    """Example: Search filings for multiple tickers."""
    print("\n" + "=" * 70)
    print("Example: Multiple Tickers")
    print("=" * 70)

    config_path = Path("config/sec_api_config.yaml")
    config = load_config(config_path)

    downloader = SECDownloader(
        api_key=config['SEC_API_KEY'],
        concurrency=2  # Lower concurrency for multiple tickers
    )

    tickers = ["MSFT", "AAPL", "GOOGL"]
    all_filings = []

    print("\nSearching filings for multiple tickers...")
    for ticker in tickers:
        print(f"\n  {ticker}:")
        try:
            filings = downloader.search_filings(
                ticker=ticker,
                start="2024-01-01",
                end="2024-12-31",
                limit=2  # Only 2 per ticker
            )
            print(f"    Found {len(filings)} filings")
            all_filings.extend(filings)

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nTotal filings across all tickers: {len(all_filings)}")
    print("=" * 70)


def example_filter_by_form_type():
    """Example: Filter and process specific form types."""
    print("\n" + "=" * 70)
    print("Example: Filter by Form Type")
    print("=" * 70)

    config_path = Path("config/sec_api_config.yaml")
    config = load_config(config_path)

    downloader = SECDownloader(api_key=config['SEC_API_KEY'])

    # Search filings
    print("\n1. Searching for TSLA filings...")
    all_filings = downloader.search_filings(
        ticker="TSLA",
        start="2023-01-01",
        end="2025-10-26",
        limit=20
    )

    # Filter only 10-K (annual reports)
    filings_10k = [f for f in all_filings if f['formType'] == '10-K']
    print(f"   Total found: {len(all_filings)}")
    print(f"   10-K only:   {len(filings_10k)}")

    # Filter only 10-Q (quarterly reports)
    filings_10q = [f for f in all_filings if f['formType'] == '10-Q']
    print(f"   10-Q only:   {len(filings_10q)}")

    # Download only 10-K filings
    if filings_10k:
        print(f"\n2. Downloading {len(filings_10k)} 10-K filings...")
        out_dir = Path("./downloads/annual_only")
        summary = downloader.download_filings(
            filings=filings_10k,
            out_dir=out_dir
        )
        print(f"   Succeeded: {summary.succeeded}/{summary.attempted}")

    print("=" * 70)


def example_error_handling():
    """Example: Demonstrate error handling."""
    print("\n" + "=" * 70)
    print("Example: Error Handling")
    print("=" * 70)

    config_path = Path("config/sec_api_config.yaml")

    try:
        # This will fail - invalid config path
        print("\n1. Testing invalid config path...")
        bad_config = load_config(Path("nonexistent.json"))

    except FileNotFoundError as e:
        print(f"   ✓ Caught error: {e}")

    # Load valid config
    config = load_config(config_path)
    downloader = SECDownloader(api_key=config['SEC_API_KEY'])

    try:
        # This will fail - invalid ticker
        print("\n2. Testing invalid ticker...")
        filings = downloader.search_filings(
            ticker="INVALID_TICKER_123",
            start="2024-01-01",
            end="2024-12-31"
        )

    except ValueError as e:
        print(f"   ✓ Caught error: {e}")

    try:
        # This will fail - invalid date range
        print("\n3. Testing invalid date range...")
        filings = downloader.search_filings(
            ticker="TSLA",
            start="2025-01-01",
            end="2024-01-01"  # End before start
        )

    except ValueError as e:
        print(f"   ✓ Caught error: {e}")

    try:
        # This will fail - invalid date format
        print("\n4. Testing invalid date format...")
        filings = downloader.search_filings(
            ticker="TSLA",
            start="01/01/2024",  # Wrong format
            end="12/31/2024"
        )

    except ValueError as e:
        print(f"   ✓ Caught error: {e}")

    print("\n   All error handling tests passed!")
    print("=" * 70)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SEC Downloader - Example Usage")
    print("=" * 70)
    print("\nNOTE: Make sure to configure your API key in config/sec_api_config.yaml")
    print("      before running these examples.")
    print("=" * 70)

    # Check if config exists
    config_path = Path("config/sec_api_config.yaml")
    if not config_path.exists():
        print(f"\n✗ Config file not found: {config_path}")
        print("  Create it with your SEC API key before running examples.")
        return

    try:
        config = load_config(config_path)
        if config['SEC_API_KEY'] == "YOUR_API_KEY_HERE":
            print("\n✗ Please update SEC_API_KEY in config/sec_api_config.yaml")
            print("  Get your API key at: https://sec-api.io/")
            return
    except Exception as e:
        print(f"\n✗ Error loading config: {e}")
        return

    # Run examples
    choice = input("\nWhich example would you like to run?\n"
                   "  1. Basic programmatic usage\n"
                   "  2. Multiple tickers\n"
                   "  3. Filter by form type\n"
                   "  4. Error handling\n"
                   "  5. All examples\n"
                   "\nChoice (1-5): ")

    examples = {
        '1': example_programmatic_usage,
        '2': example_search_multiple_tickers,
        '3': example_filter_by_form_type,
        '4': example_error_handling,
    }

    if choice == '5':
        for func in examples.values():
            func()
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
