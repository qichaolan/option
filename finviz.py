"""
Finviz Elite API Client for downloading stock and option data.

This module provides a client for interacting with the Finviz Elite API
to download option price data, option volatility data, and stock details.
"""

import csv
import io
import os
from typing import Optional, Union
import requests
import yaml


class StockDownloader:
    """
    Client for downloading stock and option data from Finviz Elite.

    Attributes:
        base_url (str): Base URL for the Finviz Elite API
        option_url (str): URL endpoint for option data exports
        auth_token (str): Authentication token for API access
        output_file (str, optional): Path to save downloaded data
    """

    DEFAULT_TIMEOUT = 300  # seconds

    def __init__(self, auth_file: str, output_file: Optional[str] = None):
        """
        Initialize the StockDownloader with authentication credentials.

        Args:
            auth_file: Path to YAML file containing 'auth_token' key
            output_file: Optional path to save downloaded data to file

        Raises:
            FileNotFoundError: If auth_file does not exist
            KeyError: If auth_file does not contain 'auth_token' key
        """
        self.base_url = "https://elite.finviz.com/"
        self.option_url = self.base_url + "export/options"
        self.auth_token = self._load_auth_token(auth_file)
        self.output_file = output_file

    def _load_auth_token(self, auth_file: str) -> str:
        """
        Load authentication token from YAML file.

        Args:
            auth_file: Path to YAML file containing authentication token

        Returns:
            Authentication token string

        Raises:
            FileNotFoundError: If auth_file does not exist
            KeyError: If auth_file does not contain 'auth_token' key
        """
        try:
            with open(auth_file, 'r') as file:
                auth_data = yaml.safe_load(file)
                return auth_data['auth_token']
        except FileNotFoundError:
            print(f"Error: Authentication file '{auth_file}' not found.")
            raise
        except KeyError:
            print("Error: Invalid authentication file format. Expected 'auth_token' key.")
            raise

    def _make_request(self, url: str, data_type: str = "data") -> Optional[Union[csv.DictReader, bool]]:
        """
        Make HTTP request to Finviz API with proper error handling.

        Args:
            url: Complete URL to request (should already include auth token)
            data_type: Type of data being downloaded (for error messages)

        Returns:
            csv.DictReader if no output_file is set, True if file written successfully,
            None if request failed
        """
        headers = {
            "User-Agent": "StockDownloader/1.0"
        }

        try:
            response = requests.get(url, headers=headers, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()
            return self._handle_response(response, data_type)
        except requests.exceptions.Timeout:
            print(f"Error: Request timed out after {self.DEFAULT_TIMEOUT} seconds.")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Error: Failed to connect to {self.base_url}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"Error: HTTP {response.status_code} - {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to download {data_type}: {e}")
            return None

    def _handle_response(self, response: requests.Response, data_type: str, allow_html: bool = False) -> Union[csv.DictReader, bool, str, None]:
        """
        Handle successful API response by saving to file or returning CSV reader.

        Args:
            response: Successful HTTP response object
            data_type: Type of data being downloaded (for success messages)
            allow_html: If True, return HTML content instead of treating it as error

        Returns:
            csv.DictReader if no output_file is set and content is CSV,
            True if file written successfully,
            str (HTML content) if allow_html=True and response is HTML,
            None if response contains HTML instead of CSV (when allow_html=False)
        """
        content = response.content.decode('utf-8')

        # Check if response is HTML instead of CSV
        if content.strip().startswith('<!DOCTYPE html>') or content.strip().startswith('<html'):
            if allow_html:
                # For filings, HTML is expected - return it directly
                print(f"Success: {data_type.capitalize()} downloaded (HTML format).")
                return content
            else:
                # For CSV endpoints, HTML indicates an error
                print(f"Error: Received HTML response instead of CSV data for {data_type}")
                print("This usually indicates:")
                print("  - Invalid or expired authentication token")
                print("  - Inactive Finviz Elite subscription")
                print("  - Incorrect API endpoint or parameters")
                return None

        if self.output_file:
            if os.path.exists(self.output_file):
                print(f"Warning: {self.output_file} already exists and will be replaced.")
            with open(self.output_file, "wb") as file:
                file.write(response.content)
            print(f"Success: {data_type.capitalize()} downloaded to {self.output_file}.")
            return True
        else:
            return csv.DictReader(io.StringIO(content))

    def download_option_price_data(
        self,
        stock_name: str,
        expiry_date: Optional[str] = None,
        strike_price: Optional[str] = None
    ) -> Optional[Union[csv.DictReader, bool]]:
        """
        Download option price data for a given stock.

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL')
            expiry_date: Option expiry date (format depends on Finviz API)
            strike_price: Option strike price

        Returns:
            csv.DictReader if no output_file is set, True if file written successfully,
            None if request failed

        Raises:
            ValueError: If neither expiry_date nor strike_price is provided,
                       or if stock_name is empty
        """
        if not stock_name or not isinstance(stock_name, str) or not stock_name.strip():
            raise ValueError("stock_name must be a non-empty string.")

        if expiry_date:
            filters = f"t={stock_name}&p=d&ty=oc&e={expiry_date}&auth={self.auth_token}"
        elif strike_price:
            filters = f"t={stock_name}&p=d&ty=oc&ov=chain_strike&s={strike_price}&auth={self.auth_token}"
        else:
            raise ValueError("Either expiry_date or strike_price must be provided.")

        url = f"{self.option_url}?{filters}"
        return self._make_request(url, "option price data")

    def download_option_volatility_data(
        self,
        stock_name: str,
        expiry_date: Optional[str] = None,
        strike_price: Optional[str] = None
    ) -> Optional[Union[csv.DictReader, bool]]:
        """
        Download option volatility data for a given stock.

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL')
            expiry_date: Option expiry date (format depends on Finviz API)
            strike_price: Option strike price

        Returns:
            csv.DictReader if no output_file is set, True if file written successfully,
            None if request failed

        Raises:
            ValueError: If neither expiry_date nor strike_price is provided,
                       or if stock_name is empty
        """
        if not stock_name or not isinstance(stock_name, str) or not stock_name.strip():
            raise ValueError("stock_name must be a non-empty string.")

        if expiry_date:
            filters = f"t={stock_name}&p=d&ty=ocv&e={expiry_date}&auth={self.auth_token}"
        elif strike_price:
            filters = f"t={stock_name}&p=d&ty=ocv&ov=chain_strike&s={strike_price}&auth={self.auth_token}"
        else:
            raise ValueError("Either expiry_date or strike_price must be provided.")

        url = f"{self.option_url}?{filters}"
        return self._make_request(url, "option volatility data")

    def download_stock_detail_data(self, stock_name: str) -> Optional[Union[csv.DictReader, bool]]:
        """
        Download detailed stock information.

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            csv.DictReader if no output_file is set, True if file written successfully,
            None if request failed

        Raises:
            ValueError: If stock_name is empty or invalid
        """
        if not stock_name or not isinstance(stock_name, str) or not stock_name.strip():
            raise ValueError("stock_name must be a non-empty string.")

        url = f"{self.base_url}quote_export.ashx?t={stock_name}&p=d&auth={self.auth_token}"

        # Mask auth token in debug output for security
        masked_url = url.replace(self.auth_token, "***")
        print(f"Downloading data from {masked_url} ...")

        return self._make_request(url, "stock detail data")

    def download_latest_filings(
        self,
        stock_name: str
    ) -> Optional[csv.DictReader]:
        """
        Download company's latest SEC filings list from Finviz and return as CSV.

        This method downloads an HTML page from Finviz Elite that contains
        a list of all SEC filings (10-K, 10-Q, 8-K, etc.) for the specified ticker,
        parses the embedded JSON data, and returns it in CSV format with columns:
        - filing_date: Date the filing was submitted to SEC
        - report_date: Period end date (if applicable)
        - form: Form type (10-K, 10-Q, 8-K, etc.)
        - document_url: Full SEC.gov URL to the filing document

        Args:
            stock_name: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

        Returns:
            csv.DictReader with filing data, or None if request failed

        Raises:
            ValueError: If stock_name is empty or invalid

        Example:
            >>> downloader = StockDownloader('config/finviz_auth.yaml')
            >>> filings = downloader.download_latest_filings('AAPL')
            >>> for filing in filings:
            >>>     print(f"{filing['form']} - {filing['filing_date']} - {filing['document_url']}")
        """
        if not stock_name or not isinstance(stock_name, str) or not stock_name.strip():
            raise ValueError("stock_name must be a non-empty string.")

        import re
        import json

        # Construct URL for latest filings (all types: annual, quarterly, current)
        # f=annual-quarterly-current includes 10-K, 10-Q, and 8-K filings
        url = f"{self.base_url}quote.ashx?t={stock_name}&p=d&ty=lf&f=annual-quarterly-current&auth={self.auth_token}"

        # Mask auth token in debug output for security
        masked_url = url.replace(self.auth_token, "***")
        print(f"Downloading latest filings list from {masked_url} ...")

        # Filings endpoint returns HTML with embedded JSON data
        headers = {
            "User-Agent": "StockDownloader/1.0"
        }

        try:
            response = requests.get(url, headers=headers, timeout=self.DEFAULT_TIMEOUT)
            response.raise_for_status()

            html_content = response.content.decode('utf-8')

            # Parse JSON from HTML
            match = re.search(
                r'<script id="route-init-data" type="application/json">(.+?)</script>',
                html_content,
                re.DOTALL
            )

            if not match:
                print("Error: Could not find filing data in Finviz response")
                return None

            json_str = match.group(1)
            data = json.loads(json_str)

            items = data.get('items', [])
            if not items:
                print(f"Warning: No filings found for {stock_name}")
                return None

            # Convert JSON items to CSV format
            csv_rows = []
            for item in items:
                # Extract relevant fields
                filing_date = item.get('filingDate', '')[:10]  # Extract YYYY-MM-DD
                report_date = item.get('reportDate', '')[:10] if 'reportDate' in item else ''
                form = item.get('form', '')
                primary_doc_url = item.get('primaryDocumentUrl', '')

                # Construct full SEC.gov URL
                document_url = f"https://www.sec.gov/Archives/edgar/data/{primary_doc_url}" if primary_doc_url else ''

                csv_rows.append({
                    'filing_date': filing_date,
                    'report_date': report_date,
                    'form': form,
                    'document_url': document_url
                })

            # Convert to CSV format using DictReader
            csv_string = io.StringIO()
            if csv_rows:
                fieldnames = ['filing_date', 'report_date', 'form', 'document_url']
                writer = csv.DictWriter(csv_string, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

                # Reset to beginning for reading
                csv_string.seek(0)

                print(f"Success: Latest filings list downloaded ({len(csv_rows)} filings).")
                return csv.DictReader(csv_string)
            else:
                return None

        except requests.exceptions.Timeout:
            print(f"Error: Request timed out after {self.DEFAULT_TIMEOUT} seconds.")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Error: Failed to connect to {self.base_url}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"Error: HTTP {response.status_code} - {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from Finviz response: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to download latest filings list: {e}")
            return None
