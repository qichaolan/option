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

    def download_option_data(self, stock_name, expiry_date=None, strike_price=None):
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

    def calculate_call_option_roi(self, option_data, target_price):
        results = []
        for row in option_data:
            try:
                if row['Type'].lower() == 'put':
                    continue
                bid = float(row['Bid']) if row['Bid'] else -1
                ask = float(row['Ask']) if row['Ask'] else -1
                strike = float(row['Strike']) if row['Strike'] else -1
                delta = round(float(row['Delta']) if row['Delta'] else -1, 3)
                iv = round(float(row['IV']) if row['IV'] else -1, 3)
                
                if bid == -1 or ask == -1 or strike == -1:
                    print("Missing or invalid data in row, skipping...")
                    continue

                premium =round((bid + ask) / 2, 3)
                breakeven_price = round(strike + premium, 3)
                gain = round(target_price - breakeven_price, 3)
                roi = round((gain / premium) * 100, 3)
                results.append({
                    'Strike': strike,
                    'Premium': premium,
                    'Breakeven Price': breakeven_price,
                    'Gain': gain,
                    'ROI': roi,
                    'Delta': delta,
                    'IV': iv
                })
            except ValueError:
                print("Invalid data in row, skipping...")
        return sorted(results, key=lambda x: x['ROI'], reverse=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download stock data from finviz.\n\nUsage Example:\npython stock_option_downloader.py -a auth.yaml -o output.csv -t GOOG -d option --expiry_date 2024-10-18\npython stock_option_downloader.py -a auth.yaml -t GOOG -d option --strike_price 100")
    parser.add_argument("-a", "--auth_file", help="Path to the YAML file containing authentication token.", required=True)
    parser.add_argument("-o", "--output_file", help="Output file path.")
    parser.add_argument("-t", "--stock", help="Stock symbol (e.g., GOOG).")
    parser.add_argument("-e", "--expiry_date", help="Expiry date in format YYYY-MM-DD.")
    parser.add_argument("-s", "--strike_price", type=int, help="Strike price.")
    parser.add_argument("-d", "--data_type", type=str, choices=['option', 'groups', 'screener'], help="type of data.", required=True)
    parser.add_argument("-p", "--target_price", type=float, help="Target price for calculating ROI.", required=True)

    args = parser.parse_args()

    downloader = StockDownloader(args.auth_file, args.output_file)

    if args.data_type.lower() == "option":
        if not args.stock:
            parser.error("Stock symbol must be provided.")
        if not args.expiry_date and not args.strike_price:
            parser.error("Either --expiry_date or --strike_price must be provided.")
        
        data = downloader.download_option_data(args.stock, args.expiry_date, args.strike_price)
        if data and not args.output_file:
            roi_results = downloader.calculate_call_option_roi(data, args.target_price)
            for result in roi_results:
                print(result)
