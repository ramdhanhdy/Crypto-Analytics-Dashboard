import pandas as pd
import numpy as np
from market_data_manager import MarketDataManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class AlphaBetaCalculator:
    def __init__(self, market_data_manager):
        self.mdm = market_data_manager
        self._setup_logging()

    def _setup_logging(self):
        """Initialize logger for this class"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def calculate_alpha_beta(self, symbols=None, interval='5m', min_days=15):
        """
        Calculate the static alpha and beta of cryptocurrencies against BTC using 5-minute data.
        Exclude BTC and ETHBTC from the calculation.
        """
        data = self.mdm.get_cached_data(symbols=symbols, interval=interval, min_days=min_days)
        if data is None:
            print("No data available.")
            return None

        # Exclude BTC and ETHBTC
        exclude_symbols = ['BTCUSDT', 'ETHBTC']
        symbols = [sym for sym in data.columns.get_level_values('symbol').unique() if sym not in exclude_symbols]

        # Get BTC close prices
        btc_symbol = 'BTCUSDT'
        if btc_symbol not in data.columns.get_level_values('symbol'):
            print(f"{btc_symbol} data is not available.")
            return None
        
        btc_close = data[btc_symbol]['close']
        btc_returns = btc_close.pct_change().dropna()

        results = {}
        for symbol in symbols:
            try:
                sym_close = data[symbol]['close']
                sym_returns = sym_close.pct_change().dropna()

                # Align returns
                aligned_data = pd.concat([sym_returns, btc_returns], axis=1).dropna()
                aligned_data.columns = ['Asset', 'Benchmark']

                # Covariance-based regression
                cov = np.cov(aligned_data['Asset'], aligned_data['Benchmark'])
                beta = cov[0, 1] / cov[1, 1]
                alpha = aligned_data['Asset'].mean() - beta * aligned_data['Benchmark'].mean()

                results[symbol] = {'alpha': alpha, 'beta': beta}
            except Exception as e:
                print(f"Error calculating alpha/beta for {symbol}: {e}")
                continue

        return results

    def calculate_rolling_alpha_beta(self, window=4320, symbols=None, interval='5m', min_days=15, progress_callback=None):
        """
        Calculate the rolling alpha and beta with BTC as benchmark.
        Excludes BTCUSDT and ETHBTC from the set of 'symbols'.
        """
        data = self.mdm.get_cached_data(symbols=symbols, interval=interval, min_days=min_days)
        if data is None:
            print("No data available.")
            return None, None
        
        exclude_symbols = ['BTCUSDT', 'ETHBTC']
        all_symbols = data.columns.get_level_values('symbol').unique()
        symbols = [sym for sym in all_symbols if sym not in exclude_symbols]
        total_symbols = len(symbols)

        # BTC is the benchmark
        btc_symbol = 'BTCUSDT'
        if btc_symbol not in all_symbols:
            print(f"{btc_symbol} data is not available.")
            return None, None

        btc_data = data[btc_symbol]['close']
        btc_returns = btc_data.pct_change().fillna(0)

        beta_dict = {}
        alpha_dict = {}

        # Rolling stats for each symbol
        for idx, symbol in enumerate(symbols):
            try:
                sym_data = data[symbol]['close']
                sym_returns = sym_data.pct_change().fillna(0)

                aligned_returns = pd.concat([sym_returns, btc_returns], axis=1).dropna()
                aligned_returns.columns = ['Asset', 'Benchmark']

                X = aligned_returns['Benchmark']
                y = aligned_returns['Asset']

                roll_X_mean = X.rolling(window=window).mean()
                roll_y_mean = y.rolling(window=window).mean()

                roll_cov = (X * y).rolling(window=window).mean() - (roll_X_mean * roll_y_mean)
                roll_var = (X * X).rolling(window=window).mean() - (roll_X_mean * roll_X_mean)

                betas = roll_cov / roll_var
                alphas = roll_y_mean - betas * roll_X_mean

                beta_dict[symbol] = betas[window:]
                alpha_dict[symbol] = alphas[window:]

                if progress_callback:
                    progress_callback((idx + 1) / total_symbols)
            except Exception as e:
                print(f"Error calculating rolling alpha/beta for {symbol}: {e}")
                continue

        rolling_betas = pd.concat(beta_dict, axis=1)
        rolling_alphas = pd.concat(alpha_dict, axis=1)
        return rolling_betas, rolling_alphas

    def calculate_rolling_alpha_beta_btcdom(self, window=4320, symbols=None, interval='5m', min_days=15, progress_callback=None):
        """
        Calculate the rolling alpha and beta using BTCDOMUSDT as the benchmark.
        Excludes only ETHBTC from the set of symbols. 
        BTCDOMUSDT itself is included and calculated via standard regression (vs. itself),
        so you'll see alpha ~ 0, beta ~ 1 in practice.
        """
        data = self.mdm.get_cached_data(symbols=symbols, interval=interval, min_days=min_days)
        if data is None:
            print("No data available.")
            return None, None
        
        # Exclude ETHBTC only
        exclude_symbols = ['ETHBTC']
        all_symbols = data.columns.get_level_values('symbol').unique()
        symbols = [sym for sym in all_symbols if sym not in exclude_symbols]
        total_symbols = len(symbols)

        btcdom_symbol = 'BTCDOMUSDT'
        if btcdom_symbol not in all_symbols:
            print(f"{btcdom_symbol} data is not available.")
            return None, None

        # BTCDOM is the benchmark
        btcdom_data = data[btcdom_symbol]['close']
        btcdom_returns = btcdom_data.pct_change().fillna(0)

        beta_dict = {}
        alpha_dict = {}

        # -- NO MORE FORCED ALPHA=0, BETA=1 FOR BTCDOM --
        # We'll let the standard rolling regression handle BTCDOM vs. BTCDOM
        # which, in practice, yields alpha ~ 0 and beta ~ 1 from the data.

        for idx, symbol in enumerate(symbols):
            try:
                sym_data = data[symbol]['close']
                sym_returns = sym_data.pct_change().fillna(0)

                aligned_returns = pd.concat([sym_returns, btcdom_returns], axis=1).dropna()
                aligned_returns.columns = ['Asset', 'Benchmark']

                X = aligned_returns['Benchmark']
                y = aligned_returns['Asset']

                roll_X_mean = X.rolling(window=window).mean()
                roll_y_mean = y.rolling(window=window).mean()

                roll_cov = (X * y).rolling(window=window).mean() - (roll_X_mean * roll_y_mean)
                roll_var = (X * X).rolling(window=window).mean() - (roll_X_mean * roll_X_mean)

                betas = roll_cov / roll_var
                alphas = roll_y_mean - betas * roll_X_mean

                beta_dict[symbol] = betas[window:]
                alpha_dict[symbol] = alphas[window:]

                if progress_callback:
                    progress_callback((idx + 1) / total_symbols)
            except Exception as e:
                print(f"Error calculating rolling alpha/beta for {symbol}: {e}")
                continue

        rolling_betas = pd.concat(beta_dict, axis=1)
        rolling_alphas = pd.concat(alpha_dict, axis=1)
        return rolling_betas, rolling_alphas

    def calculate_performance_metrics(self, window=4320, symbols=None, interval='5m', min_days=15,
                                      risk_free_rate=0.0, progress_callback=None):
        """
        Calculate performance metrics using vectorized operations and parallel processing.
        
        Parameters:
        -----------
        window : int
            Rolling window size
        symbols : list, optional
            List of symbols to analyze
        interval : str
            Time interval for data
        min_days : int
            Minimum days of data required
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
        progress_callback : callable, optional
            Callback function for progress updates
        """
        data = self.mdm.get_cached_data(symbols=symbols, interval=interval, min_days=min_days)
        if data is None:
            self.logger.error("No data available")
            return None

        all_symbols = data.columns.get_level_values(0).unique()
        exclude_symbols = ['BTCUSDT', 'ETHBTC', 'BTCDOMUSDT']
        symbols = [sym for sym in all_symbols if sym not in exclude_symbols]
        total_symbols = len(symbols)

        def calculate_symbol_metrics(symbol):
            """Calculate all metrics for a single symbol using vectorized operations"""
            try:
                # Get price data and calculate returns once
                prices = data[symbol]['close']
                returns = prices.pct_change().fillna(0)
                
                # Calculate rolling returns and volatility once
                rolling_returns = returns.rolling(window=window).sum()
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(window)
                
                # Efficient downside volatility calculation
                downside_returns = returns.copy()
                downside_returns[downside_returns > 0] = 0
                rolling_downside_vol = downside_returns.rolling(window=window).std() * np.sqrt(window)
                
                # Efficient drawdown calculation using vectorized operations
                rolling_cum_returns = (1 + returns).rolling(window=window).apply(
                    lambda x: np.prod(1 + x) - 1,
                    raw=True
                )
                rolling_cum_max = rolling_cum_returns.expanding().max()
                rolling_drawdown = (rolling_cum_returns - rolling_cum_max) / rolling_cum_max
                
                # Calculate Sharpe and Sortino ratios
                excess_returns = rolling_returns - risk_free_rate
                rolling_sharpe = np.where(rolling_vol != 0, excess_returns / rolling_vol, 0)
                rolling_sortino = np.where(
                    rolling_downside_vol != 0,
                    excess_returns / rolling_downside_vol,
                    0
                )
                
                return {
                    'sharpe': pd.Series(rolling_sharpe, index=prices.index),
                    'sortino': pd.Series(rolling_sortino, index=prices.index),
                    'max_drawdown': rolling_drawdown,
                    'volatility': rolling_vol,
                    'return': rolling_returns
                }
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {symbol}: {e}")
                return None

        # Process symbols in parallel
        metrics_dict = {'sharpe': {}, 'sortino': {}, 'max_drawdown': {}, 'volatility': {}, 'return': {}}
        processed = 0
        
        with ThreadPoolExecutor(max_workers=min(10, total_symbols)) as executor:
            future_to_symbol = {
                executor.submit(calculate_symbol_metrics, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    metrics = future.result()
                    if metrics:
                        for metric, values in metrics.items():
                            metrics_dict[metric][symbol] = values
                    
                    processed += 1
                    if progress_callback:
                        progress_callback(processed / total_symbols)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
        
        # Combine results
        result = {}
        for metric in metrics_dict:
            if metrics_dict[metric]:
                df_metric = pd.concat(metrics_dict[metric], axis=1)
                df_metric = df_metric.replace([np.inf, -np.inf], np.nan).fillna(0)
                result[metric] = df_metric
        
        return result

    def detect_market_regimes(self, symbol: str, lookback: int, timeframe: str) -> pd.DataFrame:
        """
        Detect market regimes across multiple timeframes using existing 5m data
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        lookback : int
            Number of periods to analyze
        timeframe : str
            Timeframe for analysis ('5m', '15m', '1H', '4H', '1D')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing regime analysis results
        """
        try:
            data = self.mdm.get_cached_data()
            if data is None or symbol not in data.columns.get_level_values('symbol'):
                self.logger.error(f"No data available for {symbol}")
                return None

            # Get price data and resample to desired timeframe
            prices = data[symbol]['close']
            
            # Convert timeframe to pandas offset string
            offset_map = {
                '5m': '5T',
                '15m': '15T',
                '1H': '1H',
                '4H': '4H',
                '1D': '1D'
            }
            offset = offset_map.get(timeframe)
            if not offset:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Resample data
            resampled = pd.DataFrame()
            resampled['price'] = prices.resample(offset).last()
            resampled['returns'] = np.log(resampled['price']).diff()
            resampled['volatility'] = resampled['returns'].rolling(20).std() * np.sqrt(252)
            
            # Take the last n periods based on lookback
            resampled = resampled.tail(lookback)
            
            # Simple regime classification based on volatility and returns
            vol_threshold = resampled['volatility'].mean() + resampled['volatility'].std()
            ret_threshold = resampled['returns'].mean()
            
            def classify_regime(row):
                if row['volatility'] > vol_threshold:
                    return 'High Volatility'
                elif row['returns'] > ret_threshold:
                    return 'Bull Market'
                else:
                    return 'Bear Market'
            
            resampled['regime_label'] = resampled.apply(classify_regime, axis=1)
            
            # Add timeframe info
            resampled['timeframe'] = timeframe
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error detecting market regimes: {e}")
            return None
