import pandas as pd
import numpy as np
from market_data_manager import MarketDataManager

class AlphaBetaCalculator:
    def __init__(self, market_data_manager):
        self.mdm = market_data_manager

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
        Calculate performance metrics (Sharpe, Sortino, Max Drawdown, Volatility, Return) using rolling windows.
        """
        data = self.mdm.get_cached_data(symbols=symbols, interval=interval, min_days=min_days)
        if data is None:
            print("No data available.")
            return None
        
        if not isinstance(data.columns, pd.MultiIndex):
            print("Data structure is not correct.")
            return None
        
        all_symbols = data.columns.get_level_values(0).unique()
        exclude_symbols = ['BTCUSDT', 'ETHBTC', 'BTCDOMUSDT']
        symbols = [sym for sym in all_symbols if sym not in exclude_symbols]
        total_symbols = len(symbols)

        metric_dfs = {
            'sharpe': {},
            'sortino': {},
            'max_drawdown': {},
            'volatility': {},
            'return': {}
        }

        for idx, symbol in enumerate(symbols):
            try:
                sym_data = data[symbol]['close']
                returns = sym_data.pct_change().fillna(0)

                rolling_returns = returns.rolling(window=window).sum()
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(window)

                downside_returns = returns.copy()
                downside_returns[downside_returns > 0] = 0
                rolling_downside_vol = downside_returns.rolling(window=window).std() * np.sqrt(window)

                rolling_sharpe = rolling_returns / rolling_vol
                rolling_sortino = rolling_returns / rolling_downside_vol

                # Rolling max drawdown
                rolling_cum_returns = (1 + returns).rolling(window=window).apply(lambda x: np.prod(1 + x) - 1)
                # Expand is for the entire series, but we only want rolling window
                # We can track max within that rolling window:
                # For better clarity: We can compute within the rolling apply or a short approach is:
                # We'll do it simply with a second rolling:
                # Note: This part can be done in various ways. We'll keep your logic for now.
                expanding_max = rolling_cum_returns.expanding().max()
                rolling_drawdown = (rolling_cum_returns - expanding_max) / expanding_max

                metric_dfs['sharpe'][symbol] = rolling_sharpe
                metric_dfs['sortino'][symbol] = rolling_sortino
                metric_dfs['max_drawdown'][symbol] = rolling_drawdown
                metric_dfs['volatility'][symbol] = rolling_vol
                metric_dfs['return'][symbol] = rolling_returns

                if progress_callback:
                    progress_callback((idx + 1) / total_symbols)

            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {e}")
                continue

        result = {}
        for metric in metric_dfs:
            df_metric = pd.concat(metric_dfs[metric], axis=1)
            df_metric = df_metric.replace([np.inf, -np.inf], np.nan).fillna(0)
            result[metric] = df_metric

        return result
