from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import pandas as pd
import time
import requests
import certifi
import json
import os
from dateutil import parser  
from datetime import datetime, timedelta  # Make sure both are imported


class CryptoPriceData:
    def __init__(self, config_path='config.json'):
        """
        Initialize with API credentials from config file
        
        Parameters:
        config_path (str): Path to the config JSON file containing API credentials
        """
        api_key, api_secret = self._load_config(config_path)
        
        if api_key and api_secret:
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                requests_params={'verify': certifi.where()}
            )
        else:
            print("Warning: No API keys provided. Some futures endpoints may not be accessible.")
            self.client = Client(
                requests_params={'verify': certifi.where()}
            )
        requests.packages.urllib3.disable_warnings()
    
    def _load_config(self, config_path):
        """Load API credentials from config file"""
        try:
            if not os.path.exists(config_path):
                print(f"Config file not found at {config_path}")
                return None, None
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            return config['binance']['api_key'], config['binance']['api_secret']
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return None, None

    def get_all_perpetual_symbols(self):
        """Get all available perpetual futures trading pairs"""
        try:
            exchange_info = self.client.futures_exchange_info()
            perpetual_symbols = [
                info['symbol'] for info in exchange_info['symbols']
                if info['contractType'] == 'PERPETUAL' and info['status'] == 'TRADING'
            ]
            return perpetual_symbols
        except BinanceAPIException as e:
            print(f"Error fetching perpetual symbols: {e}")
            return None

    def get_historical_perpetual_data(self, symbols=None, interval='1h', start_str='1 year ago UTC', end_str=None):
        """
        Fetch historical data for multiple perpetual futures.

        Parameters:
        -----------
        symbols : list
            List of trading pair symbols. If None, fetches all available perpetuals.
        interval : str
            Kline interval.
        start_str : str
            Start time in UTC format or relative time.
        end_str : str, optional
            End time in UTC format or relative time. If None, fetches up to the current time.

        Returns:
        --------
        pd.DataFrame or None
            DataFrame with historical OHLCV data or None if no data is fetched.
        """
        try:
            if symbols is None:
                symbols = self.get_all_perpetual_symbols()

            # Dictionary to store DataFrames for each symbol
            price_data = {}

            for symbol in symbols:
                print(f"Fetching data for {symbol}...")

                # Convert start_str and end_str to timestamps
                start_ts = self._convert_time_str_to_timestamp(start_str)
                end_ts = self._convert_time_str_to_timestamp(end_str) if end_str else int(datetime.now().timestamp() * 1000)

                # Initialize empty list to store all klines
                all_klines = []

                # Fetch data in chunks
                while start_ts < end_ts:
                    # Fetch the klines/candlestick data (max 1000 candles per request)
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=start_ts,
                        endTime=end_ts,
                        limit=1000
                    )

                    if not klines:
                        break

                    all_klines.extend(klines)

                    # Update start_ts for next iteration
                    start_ts = klines[-1][0] + 1  # Start from next candle

                    # Add small delay to avoid rate limits
                    time.sleep(0.1)

                if all_klines:
                    # Convert to DataFrame
                    df = pd.DataFrame(all_klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])

                    # Convert timestamp to datetime and set as index
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    # Convert string values to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)

                    # Store the OHLCV data
                    price_data[symbol] = df[['open', 'high', 'low', 'close', 'volume']]

                    print(f"Retrieved {len(df)} candles for {symbol}")

            # Combine all DataFrames
            if len(symbols) == 1:
                return price_data[symbols[0]]
            else:
                # For multiple symbols, create MultiIndex DataFrame
                combined_df = pd.concat(price_data.values(), keys=symbols, axis=1)
                combined_df.columns = pd.MultiIndex.from_product([symbols, ['open', 'high', 'low', 'close', 'volume']])
                return combined_df

        except BinanceAPIException as e:
            print(f"Error fetching historical data: {e}")
            return None

    def get_open_interest_data(self, symbols=None, interval='1d', start_str='1 year ago UTC'):
        """
        Fetch historical open interest data for perpetual futures
        
        Parameters:
        symbols (list): List of trading pair symbols. If None, fetches all available perpetuals
        interval (str): Data interval. Options: '5m','15m','30m','1h','2h','4h','6h','12h','1d'
        start_str (str): Start time in UTC format or relative time
        """
        try:
            if symbols is None:
                symbols = self.get_all_perpetual_symbols()
            
            # Dictionary to store DataFrames for each symbol
            oi_data = {}
            
            for symbol in symbols:
                print(f"Fetching OI data for {symbol}...")
                
                # Fetch the open interest data
                oi = self.client.futures_open_interest_hist(
                    symbol=symbol,
                    period=interval,
                    limit=500,  # Maximum limit per request
                    startTime=self._convert_time_str_to_timestamp(start_str)
                )
                
                if not oi:
                    print(f"No OI data available for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(oi)
                
                # Convert timestamp to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Convert string values to float
                df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
                
                # Store the open interest data
                oi_data[symbol] = df['sumOpenInterest']
            
            # Combine all series into a single DataFrame
            combined_df = pd.concat(oi_data, axis=1)
            
            # Save combined data to parquet
            combined_df.to_parquet(f"open_interest_data_{interval}_{start_str.replace(' ', '_')}.parquet")
            
            return combined_df
            
        except BinanceAPIException as e:
            print(f"Error fetching open interest data: {e}")
            return None

    # In data_loader.py, replace the _convert_time_str_to_timestamp method with this:

    def _convert_time_str_to_timestamp(self, time_str):
        """Convert time string to millisecond timestamp"""
        if isinstance(time_str, str) and 'ago' in time_str:
            
            parts = time_str.split()
            num = int(parts[0])
            unit = parts[1].lower()
            
            now = datetime.utcnow()
            
            if unit.startswith('minute'):
                delta = timedelta(minutes=num)
            elif unit.startswith('hour'):
                delta = timedelta(hours=num)
            elif unit.startswith('day'):
                delta = timedelta(days=num)
            elif unit.startswith('week'):
                delta = timedelta(weeks=num)
            elif unit.startswith('month'):
                delta = timedelta(days=num * 30)  # approximate
            elif unit.startswith('year'):
                delta = timedelta(days=num * 365)  # approximate
            else:
                raise ValueError(f"Unsupported time unit: {unit}")
            
            start_time = now - delta
            return int(start_time.timestamp() * 1000)
        else:
            # Parse absolute time strings
            try:
                dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                return int(dt.timestamp() * 1000)
            except ValueError:
                try:
                    dt = datetime.strptime(time_str, '%Y-%m-%d')
                    return int(dt.timestamp() * 1000)
                except ValueError:
                    raise ValueError(f"Unsupported time format: {time_str}")

