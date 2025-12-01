#!/usr/bin/env /usr/bin/python3
"""
Finviz Options Data Downloader and ROI Calculator

This module provides functionality to download options data from Finviz Elite
and calculate call option ROI based on target prices.
"""

import requests
import yaml
import csv
import io
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockDownloader:
    """
    A class to download stock options data from Finviz Elite and calculate ROI.

    Attributes:
        base_url (str): Base URL for Finviz Elite
        option_url (str): URL endpoint for options data
        auth_token (str): Authentication token for API access
        session (requests.Session): Persistent session for HTTP requests
    """

    REQUEST_TIMEOUT = 30  # seconds

    def __init__(self, auth_file: str):
        """
        Initialize the StockDownloader.

        Args:
            auth_file: Path to YAML file containing authentication token

        Raises:
            FileNotFoundError: If auth_file doesn't exist
            ValueError: If auth_file format is invalid
        """
        self.base_url = "https://elite.finviz.com/"
        self.option_url = self.base_url + "export/options"
        self.auth_token = self.load_auth_token(auth_file)
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def load_auth_token(self, auth_file: str) -> str:
        """
        Load authentication token from YAML file.

        Args:
            auth_file: Path to YAML file containing auth_token

        Returns:
            Authentication token string

        Raises:
            FileNotFoundError: If auth_file doesn't exist
            ValueError: If auth_file doesn't contain 'auth_token' key
        """
        auth_path = Path(auth_file)

        if not auth_path.exists():
            logger.error(f"Authentication file not found: {auth_file}")
            raise FileNotFoundError(f"Authentication file not found: {auth_file}")

        try:
            with open(auth_path, 'r') as file:
                auth_data = yaml.safe_load(file)

            if not auth_data or 'auth_token' not in auth_data:
                raise ValueError("Invalid authentication file format. Expected 'auth_token' key.")

            token = auth_data['auth_token']
            if not token or not isinstance(token, str):
                raise ValueError("auth_token must be a non-empty string")

            logger.info("Authentication token loaded successfully")
            return token

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file: {e}")
            raise ValueError(f"Invalid YAML format in {auth_file}: {e}")

    def download_option_data(
        self,
        stock_name: str,
        expiry_date: Optional[str] = None,
        strike_price: Optional[float] = None
    ) -> csv.DictReader:
        """
        Download options data for a given stock.

        Args:
            stock_name: Stock ticker symbol (e.g., 'GOOG')
            expiry_date: Option expiry date in YYYY-MM-DD format
            strike_price: Strike price to filter by

        Returns:
            CSV DictReader with the options data

        Raises:
            ValueError: If neither expiry_date nor strike_price is provided
            requests.RequestException: If download fails
        """
        if not stock_name:
            raise ValueError("stock_name cannot be empty")

        if not expiry_date and not strike_price:
            raise ValueError("Either expiry_date or strike_price must be provided.")

        # Validate expiry_date format if provided
        if expiry_date:
            self._validate_date_format(expiry_date)
            filters = f"t={stock_name.upper()}&p=d&ty=oc&e={expiry_date}"
        else:
            if strike_price <= 0:
                raise ValueError("strike_price must be positive")
            filters = f"t={stock_name.upper()}&p=d&ty=oc&ov=chain_strike&s={strike_price}"

        url = f"{self.option_url}?{filters}&auth={self.auth_token}"

        try:
            logger.info(f"Downloading options data for {stock_name}...")
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()

            if response.status_code == 200:
                logger.info("Option data downloaded successfully")
                return csv.DictReader(io.StringIO(response.content.decode('utf-8')))
            else:
                logger.error(f"Unexpected status code: {response.status_code}")
                raise requests.RequestException(f"Unexpected status code: {response.status_code}")

        except requests.Timeout:
            logger.error(f"Request timed out after {self.REQUEST_TIMEOUT} seconds")
            raise
        except requests.RequestException as e:
            logger.error(f"Failed to download option data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            raise

    def _validate_date_format(self, date_str: str) -> None:
        """
        Validate date string format.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Raises:
            ValueError: If date format is invalid
        """
        from datetime import datetime
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    def calculate_call_option_roi(
        self,
        option_data: csv.DictReader,
        target_price: float
    ) -> List[Dict[str, Any]]:
        """
        Calculate ROI for call options based on target price.

        Args:
            option_data: CSV DictReader containing options data
            target_price: Target stock price for ROI calculation

        Returns:
            List of dictionaries containing ROI calculations, sorted by ROI (highest first)
        """
        if target_price <= 0:
            raise ValueError("target_price must be positive")

        results = []
        skipped_rows = 0

        for row_num, row in enumerate(option_data, start=1):
            try:
                # Skip put options
                if row.get('Type', '').lower() == 'put':
                    continue

                # Parse and validate data
                bid = self._parse_float(row.get('Bid'), 'Bid')
                ask = self._parse_float(row.get('Ask'), 'Ask')
                strike = self._parse_float(row.get('Strike'), 'Strike')
                delta = self._parse_float(row.get('Delta'), 'Delta', default=None)
                iv = self._parse_float(row.get('IV'), 'IV', default=None)

                # Skip rows with invalid critical data
                if bid is None or ask is None or strike is None:
                    logger.debug(f"Row {row_num}: Missing critical data, skipping")
                    skipped_rows += 1
                    continue

                # Validate bid/ask spread
                if bid < 0 or ask < 0 or bid > ask:
                    logger.debug(f"Row {row_num}: Invalid bid/ask values, skipping")
                    skipped_rows += 1
                    continue

                # Calculate metrics
                premium = round((bid + ask) / 2, 3)

                # Skip if premium is zero or negative
                if premium <= 0:
                    logger.debug(f"Row {row_num}: Invalid premium ({premium}), skipping")
                    skipped_rows += 1
                    continue

                breakeven_price = round(strike + premium, 3)
                gain = round(target_price - breakeven_price, 3)
                roi = round((gain / premium) * 100, 3)

                results.append({
                    'Strike': strike,
                    'Premium': premium,
                    'Breakeven Price': breakeven_price,
                    'Gain': gain,
                    'ROI': roi,
                    'Delta': round(delta, 3) if delta is not None else None,
                    'IV': round(iv, 3) if iv is not None else None
                })

            except (ValueError, KeyError) as e:
                logger.debug(f"Row {row_num}: Error processing data - {e}")
                skipped_rows += 1
                continue

        if skipped_rows > 0:
            logger.info(f"Skipped {skipped_rows} rows due to invalid or missing data")

        logger.info(f"Calculated ROI for {len(results)} call options")
        return sorted(results, key=lambda x: x['ROI'], reverse=True)

    def _parse_float(
        self,
        value: Any,
        field_name: str,
        default: Optional[float] = None
    ) -> Optional[float]:
        """
        Parse a value to float with error handling.

        Args:
            value: Value to parse
            field_name: Name of the field (for logging)
            default: Default value if parsing fails

        Returns:
            Parsed float value or default
        """
        if not value or value == '':
            return default

        try:
            parsed = float(value)
            return parsed if parsed >= 0 or default is None else default
        except (ValueError, TypeError):
            logger.debug(f"Could not parse {field_name}: {value}")
            return default

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit - close session."""
        self.session.close()


def main():
    """Main entry point for command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download options data from Finviz Elite and calculate ROI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate ROI and display results
  python finviz_options.py -a auth.yaml -t GOOG -e 2024-10-18 -p 150.50

  # Filter by strike price
  python finviz_options.py -a auth.yaml -t GOOG -s 100 -p 150.50
        """
    )

    parser.add_argument(
        "-a", "--auth_file",
        help="Path to the YAML file containing authentication token.",
        required=True
    )
    parser.add_argument(
        "-t", "--stock",
        help="Stock symbol (e.g., GOOG).",
        required=True
    )
    parser.add_argument(
        "-e", "--expiry_date",
        help="Expiry date in format YYYY-MM-DD."
    )
    parser.add_argument(
        "-s", "--strike_price",
        type=float,
        help="Strike price."
    )
    parser.add_argument(
        "-p", "--target_price",
        type=float,
        help="Target price for calculating ROI.",
        required=True
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate arguments
    if not args.expiry_date and not args.strike_price:
        parser.error("Either --expiry_date or --strike_price must be provided.")

    try:
        with StockDownloader(args.auth_file) as downloader:
            data = downloader.download_option_data(
                args.stock,
                args.expiry_date,
                args.strike_price
            )

            roi_results = downloader.calculate_call_option_roi(data, args.target_price)

            if roi_results:
                print("\n" + "="*80)
                print(f"Call Options ROI Analysis for {args.stock.upper()}")
                if args.expiry_date:
                    print(f"Expiry Date: {args.expiry_date}")
                print(f"Target Price: ${args.target_price}")
                print("="*80 + "\n")

                # Print header
                print(f"{'Strike':>8} {'Premium':>10} {'Breakeven':>10} {'Gain':>10} "
                      f"{'ROI %':>10} {'Delta':>8} {'IV':>8}")
                print("-" * 80)

                # Print results
                for result in roi_results:
                    delta_str = f"{result['Delta']:.3f}" if result['Delta'] is not None else "N/A"
                    iv_str = f"{result['IV']:.3f}" if result['IV'] is not None else "N/A"

                    print(f"{result['Strike']:>8.2f} "
                          f"${result['Premium']:>9.2f} "
                          f"${result['Breakeven Price']:>9.2f} "
                          f"${result['Gain']:>9.2f} "
                          f"{result['ROI']:>9.2f}% "
                          f"{delta_str:>8} "
                          f"{iv_str:>8}")

                print("\n" + "="*80)
                print(f"Total options analyzed: {len(roi_results)}")
                print("="*80 + "\n")
            else:
                logger.warning("No valid call options found for analysis")

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
