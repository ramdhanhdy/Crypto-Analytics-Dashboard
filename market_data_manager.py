# Import necessary libraries
from data_loader import CryptoPriceData
import pandas as pd
import os
from datetime import datetime, timedelta
import time

class MarketDataManager:
    def __init__(self, base_interval='5m', cache_dir='market_data_cache'):
        self.crypto_data = CryptoPriceData(config_path='config.json')
        self.cache_dir = cache_dir
        self.base_interval = base_interval
        self.cached_data = {}  # Store data by interval
        self.last_fetch_time = {}  # Track last fetch time by interval
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_filename(self, interval, symbol=None):
        """Generate cache filename for either combined data or individual symbol data"""
        if symbol:
            return os.path.join(self.cache_dir, f"{symbol}_market_data_{interval}.csv")
        return os.path.join(self.cache_dir, f"combined_market_data_{interval}.csv")
    
    def _load_cached_data(self, interval, symbol=None):
        """Load cached data if it exists"""
        filename = self._get_cache_filename(interval, symbol)
        if os.path.exists(filename):
            try:
                # Read CSV with specific options to handle mixed types
                df = pd.read_csv(filename, low_memory=False)
                # Convert timestamp column back to datetime index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                if symbol:
                    # For individual symbol files, just return the DataFrame
                    return df
                
                # For combined file, reconstruct MultiIndex columns
                symbols = df.columns.tolist()
                new_df = pd.concat({sym: df[sym] for sym in symbols}, axis=1)
                new_df.columns = pd.MultiIndex.from_tuples([(sym, 'close') for sym in symbols], names=['symbol', 'field'])
                
                return new_df
            except Exception as e:
                print(f"Error loading cached data: {e}")
                return None
        return None
    
    def _save_to_cache(self, df, interval, symbol=None):
        """Save data to cache"""
        filename = self._get_cache_filename(interval, symbol)
        try:
            # Create a copy to avoid modifying the original DataFrame
            df_to_save = df.copy()
            
            if not symbol:
                # Flatten MultiIndex columns to single level for combined data
                df_to_save.columns = [sym for sym in df.columns.get_level_values(0)]
            
            # Reset index to save timestamp as a column
            df_to_save.reset_index().to_csv(filename, index=False)
            print(f"Successfully saved {'combined' if not symbol else symbol} market data")
        except Exception as e:
            print(f"Error saving data: {e}")

    def initial_data_fetch(self, symbols=None, interval='5m', days=3):
        """Perform initial data fetch and save to cache"""
        if symbols is None:
            symbols = self.crypto_data.get_all_perpetual_symbols()
            print(f"Found {len(symbols)} symbols")

        all_data = {}
        
        for symbol in symbols:
            try:
                print(f"Fetching initial data for {symbol}...")
                df = self.crypto_data.get_historical_perpetual_data(
                    symbols=[symbol],
                    interval=interval,
                    start_str=f"{days} days ago UTC"
                )
                
                if df is not None:
                    df = df[['close']]
                    all_data[symbol] = df
                    print(f"Successfully fetched data for {symbol}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(
                {symbol: df for symbol, df in all_data.items()},
                axis=1,
                names=['symbol', 'field']  # Explicitly name the levels
            )
            self.cached_data = combined_df
            return combined_df
        return None

    def update_market_data(self, symbols=None, interval='5m', min_days=2):
        """Update market data with new data"""
        if symbols is None:
            symbols = self.crypto_data.get_all_perpetual_symbols()
        
        # Load existing cached data
        cached_df = self._load_cached_data(interval)
        current_time = datetime.now()
        symbol_data_dict = {}
        
        for symbol in symbols:
            try:
                print(f"Updating data for {symbol}...")
                
                if cached_df is not None and symbol in cached_df.columns.get_level_values(0):
                    symbol_data = cached_df[symbol]
                    symbol_data = symbol_data[['close']]
                    last_timestamp = symbol_data.index[-1]
                    # Add one interval to avoid duplicate data
                    minutes_to_add = int(interval[:-1]) if interval.endswith('m') else int(interval[:-1]) * 60
                    start_time = last_timestamp + timedelta(minutes=minutes_to_add)
                    
                    # If cached data is too old, fetch everything again
                    if current_time - last_timestamp > timedelta(days=min_days):
                        print(f"Cached data too old for {symbol}, fetching full history...")
                        start_str = f"{min_days} days ago UTC"
                        symbol_data = None
                    else:
                        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"Fetching new data from {start_str}")
                else:
                    print(f"No cached data found for {symbol}, fetching full history...")
                    start_str = f"{min_days} days ago UTC"
                    symbol_data = None
                
                # Fetch new data
                new_data = self.crypto_data.get_historical_perpetual_data(
                    symbols=[symbol],
                    interval=interval,
                    start_str=start_str
                )
                
                if new_data is not None:
                    new_data = new_data[['close']]
                    if symbol_data is not None:
                        # Combine cached and new data
                        final_data = pd.concat([symbol_data, new_data])
                        final_data = final_data[~final_data.index.duplicated(keep='last')]
                    else:
                        final_data = new_data
                    
                    # Sort and trim old data
                    final_data.sort_index(inplace=True)
                    cutoff_time = current_time - timedelta(days=min_days)
                    final_data = final_data[final_data.index >= cutoff_time]
                    
                    symbol_data_dict[symbol] = final_data
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error updating {symbol}: {e}")
                continue
        
        if symbol_data_dict:
            combined_df = pd.concat(symbol_data_dict.values(), 
                                    keys=symbol_data_dict.keys(), 
                                    axis=1,
                                    names=['symbol'],
                                    copy=True)
            
            # Update both disk and memory cache
            self._save_to_cache(combined_df, interval)
            self.cached_data = combined_df
            return combined_df
        return None

    def get_current_market_data(self, symbols=None, interval='5m', min_days=2):
        """
        Get the most up-to-date market data, including cached data
        """
        return self.update_market_data(symbols, interval, min_days)

    def _fetch_market_data(self, interval='5m', days=2):
        """Internal method to fetch market data"""
        # Your existing fetch logic here
        # ...

    def get_cached_data(self, symbols=None, interval='5m', min_days=15):
        """Get the currently cached market data without updating"""
        # Try memory cache first
        if hasattr(self, 'cached_data') and self.cached_data is not None:
            return self.cached_data
        
        # Try loading from disk if not in memory
        cached_df = self._load_cached_data(interval)
        if cached_df is not None:
            self.cached_data = cached_df  # Store in memory for next time
            return cached_df
        
        print("No cached data available. Please initialize data first.")
        return None