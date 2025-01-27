---

# Crypto Analytics Dashboard

**Crypto Analytics Dashboard** is a Streamlit application that provides real-time analysis of cryptocurrency markets using **alpha-beta** calculations against benchmarks like **BTC** and **BTCDOM**. It includes rolling alpha-beta distributions, performance metrics (Sharpe, Sortino, etc.), and dynamic visualizations to help traders or analysts better understand the relative performance and risk characteristics of various crypto assets.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

1. **Rolling Alpha-Beta Calculation**  
- **Benchmark Comparison**: Compare asset returns against both **BTC** and **BTCDOM** (BTC Dominance) to quickly gauge altcoins’ relative performance.  
- **Time-Varying Trends**: Visualize how alpha and beta evolve over the specified window, helping identify potential outperformers during bullish phases.  
- **Market Health Indicator**: Track the proportion of altcoins with negative alpha to assess near-term market conditions.  
  - Historically, if **fewer than 20%** of altcoins have negative alpha, there is approximately an **80%** chance the broader altcoin market stays strong over the next **6–12 hours**.  
  - Conversely, if **over 60%** of altcoins exhibit negative alpha, there is an **80%** likelihood of a short-term market downturn.

2. **Performance Metrics**  
   - Calculates Sharpe ratio, Sortino ratio, max drawdown, rolling returns, and volatility for each asset.  
   - Dynamically highlights top-performing assets based on user-defined time windows.

3. **Interactive Visualizations**  
   - Charts using Matplotlib and Seaborn, integrated within Streamlit.  
   - Distribution plots of alpha-beta across different assets, annotated with asset labels.  
   - Customizable color gradients and tooltips for clarity.

4. **Live Updates & Caching**  
   - Caching mechanism to speed up data retrieval from the Binance Futures API.  
   - One-click button to fetch fresh data or re-run calculations.

5. **User-Friendly UI**  
   - Intuitive sidebar controls for configuring date range, rolling window size, and advanced settings.  
   - Enhanced styling using custom CSS for a polished look (dark theme, gradient tabs, hover effects).

---

## Demo

[![Watch the video](https://raw.githubusercontent.com/ramdhanhdy/CryptoDashboard/master/assets/dashboard-demo.png)](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FLearn_Data_Science%2FUKVN4hl83a.mp4?alt=media&token=5c444d90-f7d2-4b50-a72d-ea667400343e)

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YourUsername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment (recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Make sure your `requirements.txt` includes packages like `streamlit`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `python-binance` (or whichever libraries are used in `data_loader.py`).

4. **Add Binance credentials**:

   - Create or update `config.json` in the project root:
     ```json
     {
       "binance": {
         "api_key": "YOUR_BINANCE_API_KEY",
         "api_secret": "YOUR_BINANCE_API_SECRET"
       }
     }
     ```
   - If you do not have API keys, some features may be limited.

---

## Usage

1. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```
   By default, Streamlit will open a new tab at `http://localhost:8501` in your browser.

2. **Navigate the UI**:
   - **Initialize Data**: Press the “Initialize Data” button to fetch historical Binance Futures data.  
   - **Update Data**: Fetch and merge additional data to keep everything up-to-date.  
   - **Calculate & Visualize**: Compute rolling alpha-beta and performance metrics, then view the charts and metrics in real time.  
   - **Tabs**: Switch between BTC Beta, BTCDOM Beta, and Performance Metrics tabs for different perspectives on the data.

3. **Configuration**:
   - Use the **sidebar** to adjust:
     - The number of **days** of data to fetch.  
     - The **analysis window** (in hours) for alpha-beta calculation.  
   - (Optional) **Advanced Settings** expander for upcoming, more granular configurations.

---

## Project Structure

```
├── app.py                     # Main Streamlit app
├── data_loader.py             # Handles data fetching from Binance
├── market_data_manager.py     # Manages caching & updates for market data
├── alpha_beta_calculator.py   # Contains logic to compute alpha-beta, metrics
├── visualization.py           # Plotting functions (matplotlib/seaborn)
├── config.json                # API keys (not committed by default)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Configuration

### Environment Variables / Secrets

- If you prefer not to store credentials in `config.json`, you can load them via environment variables or Streamlit’s secrets management:
  ```bash
  export BINANCE_API_KEY="YOUR_API_KEY"
  export BINANCE_API_SECRET="YOUR_API_SECRET"
  ```
  And then update `data_loader.py` to read from these environment variables.

### Changing Intervals or Default Parameters

- The default fetching interval is **5m** candles; you can change this in `app.py` or in `MarketDataManager` as needed.  
- The max number of days, default time windows, etc., can also be adjusted.

> This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


### Disclaimer

> This application is for **educational and informational purposes only** and is **not** financial advice. Always do your own research and consider your personal risk tolerance before making investment decisions.

---

That’s it! Happy analyzing! If you have any questions or run into issues, feel free to open an issue or contact us.