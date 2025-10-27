#!/usr/bin/env python3
"""
SEC Filings Downloader using sec-api.io

A production-grade module for searching and downloading SEC filings (10-K, 10-Q)
as PDFs using the sec-api.io Query API and Render API.

Example config.yaml:
SEC_API_KEY: YOUR_API_KEY_HERE
timeout_sec: 30
retries: 3
concurrency: 4
user_agent: SECDownloader/1.0 (research@example.com)

Usage:
    # Download latest 10 filings (no date range needed)
    python sec_downloader.py \
      --ticker TSLA \
      --config ./config.yaml \
      --out ./downloads

    # Download filings within date range
    python sec_downloader.py \
      --ticker TSLA \
      --start 2024-01-01 \
      --end 2025-10-26 \
      --config ./config.yaml \
      --out ./downloads \
      --limit 10 \
      --verbose
"""

import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================================
# Configuration & Data Classes
# ============================================================================

@dataclass
class DownloadResult:
    """Result of a single filing download attempt."""
    ticker: str
    form_type: str
    filed_at: str
    accession: str
    success: bool
    skipped: bool = False
    error: Optional[str] = None
    file_path: Optional[Path] = None


@dataclass
class DownloadSummary:
    """Summary statistics for a download session."""
    found: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    errors: list[str]


# ============================================================================
# SECDownloader Class
# ============================================================================

class SECDownloader:
    """
    Downloads SEC filings (10-K, 10-Q) as PDFs using sec-api.io APIs.

    Attributes:
        api_key: sec-api.io API key
        timeout: Request timeout in seconds
        retries: Number of retry attempts for failed requests
        backoff: Exponential backoff multiplier
        concurrency: Max concurrent downloads
        user_agent: User-Agent header for requests
    """

    QUERY_API_URL = "https://api.sec-api.io"
    RENDER_API_URL = "https://api.sec-api.io/filing-reader"

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        retries: int = 3,
        backoff: float = 1.5,
        concurrency: int = 4,
        user_agent: str = "SECDownloader/1.0"
    ):
        """
        Initialize SEC Downloader.

        Args:
            api_key: sec-api.io API key
            timeout: Request timeout in seconds (default: 30)
            retries: Number of retry attempts (default: 3)
            backoff: Exponential backoff multiplier (default: 1.5)
            concurrency: Max concurrent downloads (default: 4)
            user_agent: User-Agent string (default: "SECDownloader/1.0")

        Raises:
            ValueError: If api_key is empty
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        self.api_key = api_key.strip()
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.concurrency = concurrency
        self.user_agent = user_agent
        self.logger = logging.getLogger(__name__)

        # Create session with retry logic
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry configuration."""
        session = requests.Session()

        # Configure retry strategy for HTTP errors
        retry_strategy = Retry(
            total=self.retries,
            backoff_factor=self.backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            "User-Agent": self.user_agent,
            "Authorization": self.api_key
        })

        return session

    def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic and error handling.

        Args:
            method: HTTP method (GET, POST)
            url: Request URL
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.RequestException: On request failure after retries
        """
        kwargs.setdefault('timeout', self.timeout)

        attempt = 0
        last_exception = None

        while attempt < self.retries:
            try:
                self.logger.debug(f"Request attempt {attempt + 1}/{self.retries}: {method} {url}")
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                # Don't retry client errors (except 429)
                if status_code and 400 <= status_code < 500 and status_code != 429:
                    self.logger.error(f"Client error {status_code}: {e}")
                    raise

                last_exception = e
                attempt += 1

                if attempt < self.retries:
                    sleep_time = self.backoff ** attempt
                    self.logger.warning(
                        f"Request failed (HTTP {status_code}), "
                        f"retrying in {sleep_time:.1f}s... ({attempt}/{self.retries})"
                    )
                    time.sleep(sleep_time)

            except requests.exceptions.RequestException as e:
                last_exception = e
                attempt += 1

                if attempt < self.retries:
                    sleep_time = self.backoff ** attempt
                    self.logger.warning(
                        f"Request failed: {e}, "
                        f"retrying in {sleep_time:.1f}s... ({attempt}/{self.retries})"
                    )
                    time.sleep(sleep_time)

        # All retries exhausted
        self.logger.error(f"Request failed after {self.retries} attempts")
        raise last_exception

    def search_filings(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10
    ) -> list[dict]:
        """
        Search for SEC filings using Query API.

        Args:
            ticker: Stock ticker symbol (e.g., 'MSFT')
            start: Start date in ISO format (YYYY-MM-DD), optional
                   If not provided, defaults to 10 years ago
            end: End date in ISO format (YYYY-MM-DD), optional
                 If not provided, defaults to today
            limit: Maximum number of filings to return (default: 10)

        Returns:
            List of filing dictionaries with keys: formType, filedAt, accessionNo,
            linkToFilingDetails, companyName, ticker. Ordered by filedAt descending.

        Raises:
            ValueError: If inputs are invalid
            requests.RequestException: On API request failure
        """
        # Validate inputs
        self._validate_ticker(ticker)

        if limit <= 0:
            raise ValueError(f"Limit must be positive, got {limit}")

        ticker = ticker.upper().strip()

        # Set default date range if not provided
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        if start is None:
            # Default to 10 years ago
            start = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        # Validate date range
        self._validate_date_range(start, end)

        # Build Query API query
        query = {
            "query": {
                "query_string": {
                    "query": (
                        f'ticker:{ticker} AND '
                        f'(formType:"10-K" OR formType:"10-Q") AND '
                        f'filedAt:[{start} TO {end}]'
                    )
                }
            },
            "from": 0,
            "size": min(limit, 100),  # API typically caps at 50-100 per page
            "sort": [{"filedAt": {"order": "desc"}}]
        }

        self.logger.info(
            f"Searching filings for {ticker} from {start} to {end} (limit: {limit})"
        )
        self.logger.debug(f"Query: {query}")

        filings = []
        from_offset = 0

        # Paginate until we have enough results
        while len(filings) < limit:
            query["from"] = from_offset
            query["size"] = min(limit - len(filings), 100)

            try:
                response = self._make_request(
                    "POST",
                    self.QUERY_API_URL,
                    json=query,
                    headers={"Content-Type": "application/json"}
                )

                data = response.json()

                # Extract filings from response
                hits = data.get("filings", [])
                if not hits:
                    self.logger.debug("No more filings in response")
                    break

                for filing in hits:
                    if len(filings) >= limit:
                        break

                    # Extract required fields
                    filing_data = {
                        "formType": filing.get("formType", ""),
                        "filedAt": filing.get("filedAt", ""),
                        "accessionNo": filing.get("accessionNo", ""),
                        "linkToFilingDetails": filing.get("linkToFilingDetails", ""),
                        "companyName": filing.get("companyName", ""),
                        "ticker": filing.get("ticker", ticker)
                    }

                    # Validate required fields
                    if not filing_data["linkToFilingDetails"]:
                        self.logger.warning(
                            f"Skipping filing without linkToFilingDetails: "
                            f"{filing_data['formType']} {filing_data['filedAt']}"
                        )
                        continue

                    filings.append(filing_data)
                    self.logger.debug(
                        f"Found: {filing_data['formType']} filed {filing_data['filedAt']}"
                    )

                # Check if we have more pages
                total = data.get("total", {}).get("value", 0)
                if from_offset + len(hits) >= total:
                    self.logger.debug("Reached end of results")
                    break

                from_offset += len(hits)

            except requests.RequestException as e:
                self.logger.error(f"Failed to search filings: {e}")
                raise

        self.logger.info(f"Found {len(filings)} filings")
        return filings[:limit]

    def render_pdf(self, filing_details_url: str, out_path: Path) -> None:
        """
        Download filing as PDF using Render API.

        Args:
            filing_details_url: URL to filing details page (linkToFilingDetails)
            out_path: Output file path for PDF

        Raises:
            requests.RequestException: On API request failure
            ValueError: If PDF is empty or invalid
        """
        self.logger.debug(f"Rendering PDF from {filing_details_url}")

        # Build Render API request
        params = {
            "url": filing_details_url,
            "token": self.api_key,
            "type": "pdf"
        }

        try:
            response = self._make_request(
                "GET",
                self.RENDER_API_URL,
                params=params
            )

            # Validate PDF content
            content = response.content
            if not content or len(content) < 100:
                raise ValueError(f"Invalid or empty PDF (size: {len(content)} bytes)")

            # Verify PDF signature
            if not content.startswith(b'%PDF'):
                raise ValueError("Response is not a valid PDF file")

            # Save to file
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)

            self.logger.debug(f"Saved PDF ({len(content):,} bytes) to {out_path}")

        except requests.RequestException as e:
            self.logger.error(f"Failed to render PDF: {e}")
            raise

    def download_filings(
        self,
        filings: list[dict],
        out_dir: Path,
        max_workers: Optional[int] = None
    ) -> DownloadSummary:
        """
        Download filings as PDFs concurrently.

        Args:
            filings: List of filing dictionaries from search_filings()
            out_dir: Output directory for PDFs
            max_workers: Max concurrent downloads (default: self.concurrency)

        Returns:
            DownloadSummary with statistics
        """
        if not filings:
            self.logger.warning("No filings to download")
            return DownloadSummary(
                found=0,
                attempted=0,
                succeeded=0,
                failed=0,
                skipped=0,
                errors=[]
            )

        # Ensure output directory exists
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Verify write permissions
        if not out_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {out_dir}")
        if not self._is_writable(out_dir):
            raise PermissionError(f"No write permission for directory: {out_dir}")

        workers = max_workers or self.concurrency
        results = []

        self.logger.info(
            f"Starting download of {len(filings)} filings "
            f"with {workers} concurrent workers"
        )

        # Download filings concurrently
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all download tasks
            future_to_filing = {
                executor.submit(
                    self._download_single_filing,
                    filing,
                    out_dir
                ): filing
                for filing in filings
            }

            # Process completed downloads
            for future in as_completed(future_to_filing):
                filing = future_to_filing[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Don't log here if already skipped (logged in _download_single_filing)
                    if result.skipped:
                        pass  # Already logged in _download_single_filing
                    elif result.success:
                        self.logger.info(
                            f"✓ Downloaded {result.ticker} {result.form_type} "
                            f"{result.filed_at} → {result.file_path.name}"
                        )
                    else:
                        self.logger.error(
                            f"✗ Failed {result.ticker} {result.form_type} "
                            f"{result.filed_at}: {result.error}"
                        )

                except Exception as e:
                    # Unexpected error in download task
                    self.logger.error(
                        f"Unexpected error downloading "
                        f"{filing.get('formType')} {filing.get('filedAt')}: {e}"
                    )
                    results.append(DownloadResult(
                        ticker=filing.get('ticker', 'UNKNOWN'),
                        form_type=filing.get('formType', 'UNKNOWN'),
                        filed_at=filing.get('filedAt', 'UNKNOWN'),
                        accession=filing.get('accessionNo', 'UNKNOWN'),
                        success=False,
                        error=str(e)
                    ))

        # Build summary
        succeeded = sum(1 for r in results if r.success and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)
        failed = sum(1 for r in results if not r.success)
        errors = [r.error for r in results if not r.success and r.error]

        summary = DownloadSummary(
            found=len(filings),
            attempted=len(results),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            errors=errors
        )

        return summary

    def _download_single_filing(
        self,
        filing: dict,
        out_dir: Path
    ) -> DownloadResult:
        """
        Download a single filing as PDF.

        Checks if file already exists and skips download if present.

        Args:
            filing: Filing dictionary from search_filings()
            out_dir: Output directory

        Returns:
            DownloadResult
        """
        ticker = filing.get('ticker', 'UNKNOWN')
        form_type = filing.get('formType', 'UNKNOWN')
        filed_at = filing.get('filedAt', 'UNKNOWN')
        accession = filing.get('accessionNo', 'UNKNOWN')
        link = filing.get('linkToFilingDetails', '')

        # Generate safe filename
        filename = self._generate_filename(
            ticker=ticker,
            form_type=form_type,
            filed_at=filed_at,
            accession=accession
        )
        out_path = out_dir / filename

        # Check if file already exists
        if out_path.exists() and out_path.stat().st_size > 0:
            self.logger.info(
                f"⊙ Skipped {ticker} {form_type} {filed_at} → {filename} "
                f"(already exists, {out_path.stat().st_size:,} bytes)"
            )
            return DownloadResult(
                ticker=ticker,
                form_type=form_type,
                filed_at=filed_at,
                accession=accession,
                success=True,
                skipped=True,
                file_path=out_path
            )

        try:
            # Download PDF
            self.render_pdf(link, out_path)

            return DownloadResult(
                ticker=ticker,
                form_type=form_type,
                filed_at=filed_at,
                accession=accession,
                success=True,
                file_path=out_path
            )

        except Exception as e:
            return DownloadResult(
                ticker=ticker,
                form_type=form_type,
                filed_at=filed_at,
                accession=accession,
                success=False,
                error=str(e)
            )

    @staticmethod
    def _generate_filename(
        ticker: str,
        form_type: str,
        filed_at: str,
        accession: str
    ) -> str:
        """
        Generate safe filename for PDF.

        Format: <ticker>_<formType>_<reportDate>.pdf
        Example: MSFT_10-K_20250730.pdf

        Args:
            ticker: Stock ticker
            form_type: Form type (10-K, 10-Q)
            filed_at: Filing date (ISO format or with timezone)
            accession: Accession number (unused, kept for compatibility)

        Returns:
            Safe filename string
        """
        # Sanitize components
        ticker = re.sub(r'[^\w.-]', '', ticker).upper()
        form_type = re.sub(r'[^\w-]', '', form_type)

        # Extract date from filed_at (handle ISO format or datetime with timezone)
        # Examples: "2025-07-30", "2025-07-30T16:11:40-04:00"
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filed_at)
        if date_match:
            # Format as YYYYMMDD
            report_date = ''.join(date_match.groups())
        else:
            # Fallback to sanitized version
            report_date = re.sub(r'[^\d]', '', filed_at)[:8]

        # Build filename: <ticker>_<formType>_<YYYYMMDD>.pdf
        filename = f"{ticker}_{form_type}_{report_date}.pdf"

        return filename

    @staticmethod
    def _validate_ticker(ticker: str) -> None:
        """Validate ticker symbol format."""
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")

        ticker = ticker.strip().upper()

        if len(ticker) > 10:
            raise ValueError(f"Ticker too long (max 10 chars): {ticker}")

        if not re.match(r'^[A-Z.]+$', ticker):
            raise ValueError(
                f"Invalid ticker format (must be uppercase letters/dots): {ticker}"
            )

    @staticmethod
    def _validate_date_range(start: str, end: str) -> None:
        """Validate date range in ISO format."""
        try:
            start_dt = datetime.fromisoformat(start)
        except ValueError as e:
            raise ValueError(f"Invalid start date format (expected YYYY-MM-DD): {start}") from e

        try:
            end_dt = datetime.fromisoformat(end)
        except ValueError as e:
            raise ValueError(f"Invalid end date format (expected YYYY-MM-DD): {end}") from e

        if start_dt > end_dt:
            raise ValueError(f"Start date {start} is after end date {end}")

    @staticmethod
    def _is_writable(path: Path) -> bool:
        """Check if directory is writable."""
        try:
            test_file = path / '.write_test'
            test_file.touch()
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e

    # Validate required fields
    if 'SEC_API_KEY' not in config:
        raise ValueError("Missing required field 'SEC_API_KEY' in config")

    return config


# ============================================================================
# CLI
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Download SEC filings (10-K, 10-Q) as PDFs using sec-api.io',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download latest 10 TSLA filings (no date range needed)
  python sec_downloader.py --ticker TSLA --config ./config.yaml --out ./downloads

  # Download up to 10 TSLA filings from 2024
  python sec_downloader.py --ticker TSLA --start 2024-01-01 --end 2024-12-31 \\
    --config ./config.yaml --out ./downloads

  # Download 5 MSFT filings with debug logging
  python sec_downloader.py --ticker MSFT --start 2023-01-01 --end 2025-10-26 \\
    --config ./config.yaml --out ./downloads --limit 5 --verbose

Config file format (YAML):
  SEC_API_KEY: YOUR_API_KEY_HERE
  timeout_sec: 30
  retries: 3
  concurrency: 4
  user_agent: SECDownloader/1.0 (research@example.com)
        '''
    )

    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol (e.g., TSLA, MSFT)'
    )

    parser.add_argument(
        '--start',
        type=str,
        help='Start date in YYYY-MM-DD format (default: 10 years ago)'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='End date in YYYY-MM-DD format (default: today)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to config file (YAML)'
    )

    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output directory for PDFs'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=1,
        help='Maximum number of filings to download (default: 1)'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        help='Max concurrent downloads (overrides config)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)

        # Create downloader
        downloader = SECDownloader(
            api_key=config['SEC_API_KEY'],
            timeout=config.get('timeout_sec', 30),
            retries=config.get('retries', 3),
            backoff=config.get('backoff', 1.5),
            concurrency=args.concurrency or config.get('concurrency', 4),
            user_agent=config.get('user_agent', 'SECDownloader/1.0')
        )

        # Search for filings
        logger.info("=" * 70)
        logger.info("SEC Filings Downloader")
        logger.info("=" * 70)

        filings = downloader.search_filings(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            limit=args.limit
        )

        if not filings:
            logger.warning("No filings found matching criteria")
            return 0

        # Download PDFs
        logger.info("=" * 70)
        summary = downloader.download_filings(
            filings=filings,
            out_dir=args.out
        )

        # Print summary
        logger.info("=" * 70)
        logger.info("Download Summary")
        logger.info("=" * 70)
        logger.info(f"Found:     {summary.found}")
        logger.info(f"Attempted: {summary.attempted}")
        logger.info(f"Succeeded: {summary.succeeded}")
        logger.info(f"Skipped:   {summary.skipped}")
        logger.info(f"Failed:    {summary.failed}")

        if summary.errors:
            logger.info("\nErrors:")
            for i, error in enumerate(summary.errors[:5], 1):
                logger.info(f"  {i}. {error}")
            if len(summary.errors) > 5:
                logger.info(f"  ... and {len(summary.errors) - 5} more")

        logger.info("=" * 70)

        # Exit code based on results
        # Only fail if there were actual failures (not just skips)
        if summary.failed > 0 and summary.succeeded == 0:
            logger.error("All downloads failed")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
