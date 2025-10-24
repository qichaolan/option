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
            url: Complete URL to request
            data_type: Type of data being downloaded (for error messages)

        Returns:
            csv.DictReader if no output_file is set, True if file written successfully,
            None if request failed
        """
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
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

    def _handle_response(self, response: requests.Response, data_type: str) -> Union[csv.DictReader, bool]:
        """
        Handle successful API response by saving to file or returning CSV reader.

        Args:
            response: Successful HTTP response object
            data_type: Type of data being downloaded (for success messages)

        Returns:
            csv.DictReader if no output_file is set, True if file written successfully
        """
        if self.output_file:
            if os.path.exists(self.output_file):
                print(f"Warning: {self.output_file} already exists and will be replaced.")
            with open(self.output_file, "wb") as file:
                file.write(response.content)
            print(f"Success: {data_type.capitalize()} downloaded to {self.output_file}.")
            return True
        else:
            return csv.DictReader(io.StringIO(response.content.decode('utf-8')))

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
            filters = f"t={stock_name}&p=d&ty=oc&e={expiry_date}"
        elif strike_price:
            filters = f"t={stock_name}&p=d&ty=oc&ov=chain_strike&s={strike_price}"
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
            filters = f"t={stock_name}&p=d&ty=ocv&e={expiry_date}"
        elif strike_price:
            filters = f"t={stock_name}&p=d&ty=ocv&ov=chain_strike&s={strike_price}"
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

        url = f"{self.base_url}quote.ashx?t={stock_name}&p=d"
        return self._make_request(url, "stock detail data")
