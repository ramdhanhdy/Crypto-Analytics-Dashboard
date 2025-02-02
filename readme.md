# Crypto Analytics Dashboard

![Dashboard Screenshot](https://github.com/user-attachments/assets/b3e77ec0-ec57-4114-b893-3d5bbd61b714)

**Crypto Analytics Dashboard** is a Streamlit application that provides real-time analysis of cryptocurrency markets using **alpha-beta** calculations and advanced market regime detection. It combines traditional financial metrics with machine learning models to deliver unique crypto market insights.

## Table of Contents

- [Key Features](#key-features)
- [Live Demo](#live-demo)
- [Installation](#installation)
- [Usage](#usage)
- [Advanced Features](#advanced-features)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Key Features

### 1. Market Regime Analysis 🔄
- **Multi-Timeframe Detection**: Identify bull/bear/high-volatility regimes across 5m, 15m, 1H, 4H, and 1D timeframes
- **Hidden Markov Models**: Machine learning-powered regime classification
- **Volatility Clustering**: Visualize volatility patterns and regime durations
- **Real-time Probability Estimates**: Bayesian change point detection

### 2. Alpha-Beta Matrix 📊
- **Dual Benchmarking**: Compare against BTC and BTCDOM simultaneously
- **Rolling Window Analysis**: 6h/12h/24h performance windows
- **Crowding Risk Alerts**: Detect position clustering in alpha-beta space

### 3. Performance Diagnostics 💹
- Advanced metrics (Sharpe/Sortino ratios)
- Drawdown heatmaps
- Return/volatility percentiles
- Cross-asset correlation matrices

### 4. Smart Visualization 🎨
- Interactive 3D regime probability surfaces
- Institutional-grade chart styling
- Mobile-optimized views
- Dark/light mode support

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-analytics-dashboard.git
cd crypto-analytics-dashboard

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**Required Packages**:
```text
streamlit==1.29.0
hmmlearn==0.3.0
ta==0.11.0
python-binance==1.0.19
```

---

## Usage

### Basic Operation
```bash
streamlit run app.py
```

### Deep Analysis Workflow
1. Navigate to "Deep Analysis" tab
2. Select asset and timeframe
3. Adjust lookback period (504 periods recommended)
4. Interpret regime visualization:
   - 🟢 Bull Market (Price > SMA50, Low Volatility)
   - 🔴 Bear Market (Price < SMA50, High Volatility)
   - 🟡 High Volatility (VIX > 30-day average)

![Regime Analysis Demo](https://github.com/user-attachments/regime-demo.gif)

---

## Advanced Features

### Custom Analysis Templates
Create JSON configs in `analysis_templates/` to:
- Set custom volatility thresholds
- Define regime combinations
- Backtest historical regimes

### API Webhook Integration
Configure in `config.json`:
```json
{
  "webhooks": {
    "telegram": "BOT_TOKEN",
    "discord": "WEBHOOK_URL"
  }
}
```

---

## Project Structure

```bash
├── app.py                     # Main application
├── data_loader.py             # Binance API client
├── market_data_manager.py     # Data pipeline & caching
├── alpha_beta_calculator.py   # Risk metrics & ML models
├── visualization.py           # Advanced charting
├── config/                    # Analysis templates
├── .venv/                     # Python environment
└── run_app.bat                # Windows launch script
```

---

## Troubleshooting

**Common Issues**:
- `hmmlearn` installation failures:
  ```bash
  conda install -c conda-forge hmmlearn
  ```
- Binance API limits: Reduce symbol count in `config.json`
- Memory errors: Decrease lookback period

---

## License

MIT License - See [LICENSE](LICENSE) for full text.

**Disclaimer**: This software is for research purposes only. Cryptocurrency trading carries substantial risk.