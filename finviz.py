import requests
import yaml
import os
import csv
import io

class StockDownloader:
    def __init__(self, auth_file, output_file=None):
        self.base_url = "https://elite.finviz.com/"
        self.option_url = self.base_url + "export/options"
        self.auth_token = self.load_auth_token(auth_file)
        self.output_file = output_file

    def load_auth_token(self, auth_file):
        try:
            with open(auth_file, 'r') as file:
                auth_data = yaml.safe_load(file)
                return auth_data['auth_token']
        except FileNotFoundError:
            print("Authentication file not found.")
            raise
        except KeyError:
            print("Invalid authentication file format. Expected 'auth_token'.")
            raise

    def download_option_price_data(self, stock_name, expiry_date=None, strike_price=None):
        if expiry_date:
            filters = f"t={stock_name}&p=d&ty=oc&e={expiry_date}"
        elif strike_price:
            filters = f"t={stock_name}&p=d&ty=oc&ov=chain_strike&s={strike_price}"
        else:
            raise ValueError("Either expiry_date or strike_price must be provided.")

        url = f"{self.option_url}?{filters}&auth={self.auth_token}"
        response = requests.get(url)

        if response.status_code == 200:
            if self.output_file:
                if os.path.exists(self.output_file):
                    print(f"Warning: {self.output_file} already exists and will be replaced.")
                with open(self.output_file, "wb") as file:
                    file.write(response.content)
                print(f"Option data downloaded successfully to {self.output_file}.")
            else:
                return csv.DictReader(io.StringIO(response.content.decode('utf-8')))
        else:
            print(f"Failed to download option data. Status code: {response.status_code}")
            return None

    def download_option_volatility_data(self, stock_name, expiry_date=None, strike_price=None):
        if expiry_date:
            filters = f"t={stock_name}&p=d&ty=ocv&e={expiry_date}"
        elif strike_price:
            filters = f"t={stock_name}&p=d&ty=ocv&ov=chain_strike&s={strike_price}"
        else:
            raise ValueError("Either expiry_date or strike_price must be provided.")

        url = f"{self.option_url}?{filters}&auth={self.auth_token}"
        response = requests.get(url)

        if response.status_code == 200:
            if self.output_file:
                if os.path.exists(self.output_file):
                    print(f"Warning: {self.output_file} already exists and will be replaced.")
                with open(self.output_file, "wb") as file:
                    file.write(response.content)
                print(f"Option data downloaded successfully to {self.output_file}.")
            else:
                return csv.DictReader(io.StringIO(response.content.decode('utf-8')))
        else:
            print(f"Failed to download option data. Status code: {response.status_code}")
            return None

    def download_stock_detail_data(self, stock_name):
        if stock_name:
            filters = f"quote.ashx?t={stock_name}&p=d"
        else:
            raise ValueError("stock_name must be provided.")

        url = f"{self.base_url}{filters}&auth={self.auth_token}"
        response = requests.get(url)

        if response.status_code == 200:
            if self.output_file:
                if os.path.exists(self.output_file):
                    print(f"Warning: {self.output_file} already exists and will be replaced.")
                with open(self.output_file, "wb") as file:
                    file.write(response.content)
                print(f"Option data downloaded successfully to {self.output_file}.")
            else:
                return csv.DictReader(io.StringIO(response.content.decode('utf-8')))
        else:
            print(f"Failed to download option data. Status code: {response.status_code}")
            return None