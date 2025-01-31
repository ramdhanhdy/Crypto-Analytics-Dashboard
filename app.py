import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

from alpha_beta_calculator import AlphaBetaCalculator
from market_data_manager import MarketDataManager
from visualization import plot_alpha_beta_map

# ====================
# Page configuration
# ====================
st.set_page_config(
    page_title="Crypto Alpha-Beta Map",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# Enhanced Custom CSS
# ====================
CUSTOM_CSS = """
<style>
    /* Main app styling */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background-color: #1A1D23 !important;
        border-right: 1px solid #2F333D;
    }
    
    /* Improved header styling */
    h1 {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin: 0.5rem 0 1.5rem 0 !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2F333D;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(145deg, #2F333D, #1A1D23);
        border: 1px solid #2F333D;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
        background: linear-gradient(145deg, #3A3F4B, #2F333D);
    }
    
    /* Progress bar enhancements */
    .stProgress > div > div {
        background-color: #FFD700;
        box-shadow: 0 2px 4px rgba(255, 215, 0, 0.2);
    }
    
    /* Card styling */
    .metric-card {
        background: #1A1D23;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #2F333D;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding: 4px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1A1D23 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        margin: 0 4px !important;
        border: 1px solid #2F333D !important;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #FFD700, #FFA500) !important;
        color: #000 !important;
        border-color: #FFD700 !important;
        font-weight: 700;
    }
    
    /* Data table styling */
    .stDataFrame {
        border: 1px solid #2F333D;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Enhanced slider styling */
    .stSlider .thumb {
        background-color: #FFD700 !important;
        border: 2px solid #1A1D23 !important;
    }
    
    .stSlider .track {
        background-color: #2F333D !important;
    }
    
    /* Section header styling */
    .section-header {
        font-size: 1.4rem !important;
        font-weight: 600;
        color: #FFD700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2F333D;
    }
    /* Compact form elements */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 0.5rem;
    }
    
    /* Dense input labels */
    .stWidget > label {
        font-size: 0.9em;
        margin-bottom: 0.2rem;
    }
    
    /* Tight expander padding */
    .stExpander .st-expanderHeader {
        padding: 0.5rem 1rem;
    }
    
    /* Small preset buttons */
    .stButton > button {
        padding: 0.25rem 0.75rem;
        font-size: 0.9em;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====================
# Helper Functions
# ====================

def calculate_alts_below_btcdom(alphas_df, betas_df):
    """
    Calculate the percentage of alts that are below BTCDOM
    """
    latest_alphas = alphas_df.iloc[-1]
    latest_betas = betas_df.iloc[-1]
    
    total_pairs = len(latest_alphas)
    negative_pairs = sum((latest_alphas < 0) & (latest_betas < 0))
    percentage = (negative_pairs / total_pairs) * 100 if total_pairs > 0 else 0
    
    return negative_pairs, total_pairs, percentage

def calculate_alts_relative_to_btcdom(alphas_df, betas_df, alphas_dom_df, betas_dom_df):
    """
    Calculate the percentage of alts that have alpha and beta below BTCDOM's values
    """
    latest_alphas = alphas_df.iloc[-1]
    latest_betas = betas_df.iloc[-1]

    latest_dom_alpha = alphas_dom_df.iloc[-1]['BTCDOMUSDT']
    latest_dom_beta = betas_dom_df.iloc[-1]['BTCDOMUSDT']

    # Exclude BTCDOMUSDT 
    latest_alphas = latest_alphas.drop('BTCDOMUSDT', errors='ignore')
    latest_betas = latest_betas.drop('BTCDOMUSDT', errors='ignore')

    total_pairs = len(latest_alphas)
    pairs_below = sum((latest_alphas < latest_dom_alpha) & (latest_betas < latest_dom_beta))
    percentage = (pairs_below / total_pairs) * 100 if total_pairs > 0 else 0

    return pairs_below, total_pairs, percentage

def calculate_percentage_negative_alpha(alphas_df, exclude_symbol=None):
    """
    Returns the count, total, and percentage of assets whose latest alpha is < 0.
    exclude_symbol: Optionally exclude a symbol (like BTCUSDT or BTCDOMUSDT) 
                    if it's present in the DataFrame and shouldn't be counted.
    """
    # Get the most recent row of alpha data
    latest_alphas = alphas_df.iloc[-1]
    
    # Optionally exclude the benchmark if present
    if exclude_symbol and exclude_symbol in latest_alphas.index:
        latest_alphas = latest_alphas.drop(exclude_symbol)
    
    negative_count = (latest_alphas < 0).sum()
    total_count = len(latest_alphas)
    percentage = (negative_count / total_count * 100) if total_count > 0 else 0
    
    return negative_count, total_count, percentage


def create_metric_card(title, value, delta=None, is_date=False):
    """Enhanced metric card with hover effects"""
    delta_html = ""
    if delta:
        # Attempt to parse delta as something containing a percentage for up/down arrow
        try:
            # If 'delta' is something like "3.5%" or "-1.2%"
            delta_value = float(delta.replace("%", ""))  # Remove the '%' sign if present
            color = "#28C76F" if delta_value >= 0 else "#EA5455"
            icon = "▲" if delta_value >= 0 else "▼"
            delta_html = f'<span style="color: {color}; font-size: 0.9rem;">{icon} {delta}</span>'
        except:
            # If delta cannot be parsed as float or doesn't have '%'
            delta_html = f'<span style="color: #7F7F7F; font-size: 0.9rem;">{delta}</span>'

    html = f"""
    <div class="metric-card">
        <div style="margin-bottom: 0.5rem; color: #FFD700; font-size: 0.95rem; font-weight: 500;">{title}</div>
        <div style="display: flex; align-items: baseline; gap: 0.75rem;">
            <div style="font-size: 1.8rem; font-weight: 700;">{value}</div>
            {delta_html}
        </div>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)

# ====================
# Session Initialization
# ====================

def initialize_session_state():
    """Initialize session state objects if not already present."""
    if 'mdm' not in st.session_state:
        st.session_state.mdm = MarketDataManager()
        st.session_state.abc = AlphaBetaCalculator(st.session_state.mdm)
        st.session_state.initialized = False

# ====================
# Sidebar
# ====================

def render_sidebar():
    """Compact configuration panel with smart organization"""
    st.sidebar.markdown("## ⚙️ Configuration Hub")
    
    # Configuration Tabs
    tab_config, tab_presets = st.sidebar.tabs(["Settings", "Presets"])
    
    with tab_config:
        # Data Settings Accordion
        with st.expander("📦 Data Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                days_to_fetch = st.number_input(
                    "History (Days)",
                    min_value=1,
                    max_value=7,
                    value=2,
                    help="Days of historical data to load"
                )
            with col2:
                update_freq = st.selectbox(
                    "Auto-Refresh",
                    ["Off", "5m", "15m", "30m"],
                    index=1,
                    help="Automatic data refresh interval"
                )
        
        # Analysis Parameters
        with st.expander("📈 Analysis Settings", expanded=True):
            hours_to_analyze = st.slider(
                "Window Size (Hours)",
                min_value=1,
                max_value=24,
                value=4,
                step=1,
                help="Analysis time window"
            )
            
    return days_to_fetch, hours_to_analyze

# ====================
# Main Controls
# ====================

def render_main_controls(days_to_fetch, hours_to_analyze):
    st.markdown("## 🕹️ Control Center")
    
    # Reduced width => ratio 2:3 is effectively 40% vs 60% in the main layout
    if st.button("📥 Initialize Data", disabled=st.session_state.initialized):
        try:
            with st.spinner("🔄 Fetching initial data..."):
                start_time = time.time()
                data = st.session_state.mdm.initial_data_fetch(interval='5m', days=days_to_fetch)
                if data is not None:
                    st.session_state.initialized = True
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Fetched {days_to_fetch} days of data in {elapsed_time:.1f}s")
                else:
                    st.error("Failed to fetch initial data")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    if st.button("🔄 Live Update", disabled=not st.session_state.initialized):
        try:
            with st.spinner("🔄 Updating market data..."):
                start_time = time.time()
                data = st.session_state.mdm.update_market_data(interval='5m', min_days=days_to_fetch)
                if data is not None:
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Updated in {elapsed_time:.1f}s")
                else:
                    st.error("Failed to update data")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    if st.button("🚀 Calculate & Visualize", disabled=not st.session_state.initialized):
        try:
            rolling_window = 12 * hours_to_analyze
            progress_container = st.container()
            with progress_container:
                st.markdown("#### 📊 Calculation Progress")
                
                # Each label above its bar
                btc_status = st.markdown("🔵 **BTC Beta:** 0%")
                btc_progress = st.progress(0)
                
                dom_status = st.markdown("🟡 **BTCDOM Beta:** 0%")
                dom_progress = st.progress(0)
                
                metrics_status = st.markdown("📊 **Metrics:** 0%")
                metrics_progress = st.progress(0)
            
            with st.spinner("🧮 Calculating..."):
                start_time = time.time()
                
                def update_btc_progress(progress):
                    btc_status.markdown(f"🔵 **BTC Beta:** {progress:.0%}")
                    btc_progress.progress(progress)
                
                def update_dom_progress(progress):
                    dom_status.markdown(f"🟡 **BTCDOM Beta:** {progress:.0%}")
                    dom_progress.progress(progress)
                
                def update_metrics_progress(progress):
                    metrics_status.markdown(f"📊 **Metrics:** {progress:.0%}")
                    metrics_progress.progress(progress)
                
                betas, alphas = st.session_state.abc.calculate_rolling_alpha_beta(
                    window=rolling_window,
                    interval='5m',
                    min_days=days_to_fetch,
                    progress_callback=update_btc_progress
                )
                
                betas_dom, alphas_dom = st.session_state.abc.calculate_rolling_alpha_beta_btcdom(
                    window=rolling_window,
                    interval='5m',
                    min_days=days_to_fetch,
                    progress_callback=update_dom_progress
                )
                
                metrics = st.session_state.abc.calculate_performance_metrics(
                    window=rolling_window,
                    interval='5m',
                    min_days=days_to_fetch,
                    progress_callback=update_metrics_progress
                )
                
                st.session_state.last_betas = betas
                st.session_state.last_alphas = alphas
                st.session_state.last_betas_dom = betas_dom
                st.session_state.last_alphas_dom = alphas_dom
                st.session_state.last_metrics = metrics
                st.session_state.last_update = pd.Timestamp.now()
                st.session_state.show_plots = True
                
                elapsed_time = time.time() - start_time
            
            progress_container.empty()
            st.success(f"✅ Calculated in {elapsed_time:.1f}s")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # 🔄 Reset
    st.markdown("---")
    if st.button('🛑 Full Reset', 
                 type="secondary", 
                 use_container_width=True,
                 help="Clear all data and restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# ====================
# Status / Metrics
# ====================

def render_status_metrics(hours_to_analyze, days_to_fetch):
    """Render a row of status metric cards if we have a last update time."""
    if hasattr(st.session_state, 'last_update'):
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            create_metric_card(
                "Last Update",
                st.session_state.last_update.strftime('%H:%M:%S'),
                st.session_state.last_update.strftime('%Y-%m-%d')
            )
        with metrics_col2:
            create_metric_card(
                "Analysis Window",
                f"{hours_to_analyze}h",
                f"{days_to_fetch}d"
            )
        with metrics_col3:
            if hasattr(st.session_state, 'last_betas'):
                create_metric_card(
                    "Assets Analyzed",
                    len(st.session_state.last_betas.columns),
                    "Pairs"
                )

# ====================
# Tabs & Charts
# ====================

def render_btc_beta_tab(hours_to_analyze):
    """Improved tab layout with a responsive split for charts vs metrics."""
    if not (hasattr(st.session_state, 'last_alphas') and hasattr(st.session_state, 'last_betas')):
        st.warning("No BTC alpha-beta data available. Please calculate first.")
        return
    
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        st.markdown("### 📈 Alpha-Beta Distribution")
        fig1 = plot_alpha_beta_map(
            st.session_state.last_alphas,
            st.session_state.last_betas,
            title=f"BTC as Benchmark: Alpha & Beta Distribution\nLast {hours_to_analyze} Hours"
        )
        if fig1:
            st.pyplot(fig1)
            plt.close(fig1)
    
    with col2:
        st.markdown("### 📊 Key Metrics")
        
        # 1) Negative alpha
        neg_count, total_count, pct_neg = calculate_percentage_negative_alpha(
            st.session_state.last_alphas,
            exclude_symbol='BTCUSDT'  # Optionally exclude the benchmark if it appears
        )
        create_metric_card(
            "Alts with Negative Alpha",
            f"{neg_count}/{total_count}",
            f"{pct_neg:.1f}%"
        )
        
        # 2) (Optional) If you still want the “relative to BTCDOM” metric, keep it:
        if (hasattr(st.session_state, 'last_alphas_dom') 
            and hasattr(st.session_state, 'last_betas_dom')):
            from_below, total_pairs, perc_below = calculate_alts_relative_to_btcdom(
                st.session_state.last_alphas,
                st.session_state.last_betas,
                st.session_state.last_alphas_dom,
                st.session_state.last_betas_dom
            )
            create_metric_card(
                "Below BTCDOM Values (α & β)",
                f"{from_below}/{total_pairs}",
                f"{perc_below:.1f}%"
            )

def render_btcdom_beta_tab(hours_to_analyze):
    if not (hasattr(st.session_state, 'last_alphas_dom') and hasattr(st.session_state, 'last_betas_dom')):
        st.warning("No BTCDOM alpha-beta data available. Please calculate first.")
        return
    
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        st.markdown("### BTCDOM Alpha-Beta Distribution")
        fig2 = plot_alpha_beta_map(
            st.session_state.last_alphas_dom,
            st.session_state.last_betas_dom,
            title=f"BTCDOM as Benchmark: Alpha & Beta Distribution\nLast {hours_to_analyze} Hours"
        )
        if fig2:
            st.pyplot(fig2)
            plt.close(fig2)
    
    with col2:
        st.markdown("### Key Metrics")
        # Keep only one metric card instead of duplicating
        negative_pairs, total_pairs, percentage_below = calculate_alts_below_btcdom(
            st.session_state.last_alphas_dom,
            st.session_state.last_betas_dom
        )
        create_metric_card(
            "Alts with α<0 & β<0",
            f"{negative_pairs}/{total_pairs}",
            f"{percentage_below:.1f}%"
        )



def render_performance_metrics_tab():
    """Enhanced performance metrics tab with sub-tabs for top performers & full metrics."""
    if not hasattr(st.session_state, 'last_metrics') or st.session_state.last_metrics is None:
        st.warning("No performance metrics available. Please calculate first.")
        return

    metrics_data = st.session_state.last_metrics
    # Extract latest metrics for each symbol
    latest_metrics = {
        metric: df.iloc[-1].sort_values(ascending=False)
        for metric, df in metrics_data.items()
    }

    tab1, tab2, tab3 = st.tabs(["🏆 Top Performers", "📋 Full Metrics", "🔍 Deep Analysis"])
    
    with tab1:
        st.markdown("#### 🥇 Best Overall")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Highest Sharpe Ratio")
            top_sharpe = latest_metrics['sharpe'].head(5)
            for symbol, value in top_sharpe.items():
                create_metric_card(
                    symbol,
                    f"{value:.2f}",
                    f"Sortino: {latest_metrics['sortino'][symbol]:.2f}"
                )
        
        with col2:
            st.markdown("##### Best Returns")
            top_returns = latest_metrics['return'].head(5)
            for symbol, value in top_returns.items():
                create_metric_card(
                    symbol,
                    f"{value*100:.1f}%",
                    f"Vol: {latest_metrics['volatility'][symbol]*100:.1f}%"
                )
        
        with col3:
            st.markdown("##### Lowest Drawdown")
            top_dd = latest_metrics['max_drawdown'].sort_values().head(5)
            for symbol, value in top_dd.items():
                create_metric_card(
                    symbol,
                    f"{value*100:.1f}%",
                    f"Sharpe: {latest_metrics['sharpe'][symbol]:.2f}"
                )

    with tab2:
        st.markdown("#### 📜 Complete Dataset")
        detailed_df = pd.DataFrame({
            'Sharpe': latest_metrics['sharpe'],
            'Sortino': latest_metrics['sortino'],
            'Return (%)': latest_metrics['return'] * 100,
            'Volatility (%)': latest_metrics['volatility'] * 100,
            'Max Drawdown (%)': latest_metrics['max_drawdown'] * 100
        }).round(2)
        
        st.dataframe(detailed_df.style.background_gradient(cmap='RdYlGn', axis=0), height=400)

    with tab3:
        st.markdown("#### 🔎 Detailed Analysis")
        st.warning("Advanced analytical features coming in next release")

def render_tabs(hours_to_analyze):
    """Render the three main tabs for BTC Beta, BTCDOM Beta, and Performance Metrics."""
    tab1, tab2, tab3 = st.tabs(["📈 BTC Beta", "🔶 BTCDOM Beta", "📊 Performance Metrics"])
    
    with tab1:
        render_btc_beta_tab(hours_to_analyze)
    
    with tab2:
        render_btcdom_beta_tab(hours_to_analyze)
    
    with tab3:
        render_performance_metrics_tab()

# ====================
# Main Application
# ====================

# def main():
#     st.title("Crypto Alpha-Beta Map")
#     st.markdown("_Real-time cryptocurrency analytics dashboard_")
#     initialize_session_state()

#     days_to_fetch, hours_to_analyze = render_sidebar()
    
#     # Reduce first column to 40% width, second to 60% => ratio 2:3
#     col1, col2 = st.columns([2, 3], gap="large")
#     with col1:
#         render_main_controls(days_to_fetch, hours_to_analyze)
#     with col2:
#         st.markdown("## 📍 Live Dashboard")
#         render_status_metrics(hours_to_analyze, days_to_fetch)
        
#         if hasattr(st.session_state, 'show_plots') and st.session_state.show_plots:
#             render_tabs(hours_to_analyze)

def main():
    st.title("Crypto Market Analytics Dashboard")
    st.markdown("_Real-time cryptocurrency analytics dashboard_")

    # Initialize session state
    initialize_session_state()

    # Render sidebar to get user input
    days_to_fetch, hours_to_analyze = render_sidebar()

    # Two-column layout
    col1, col2 = st.columns([1, 3], gap="small")

    with col1:
        render_main_controls(days_to_fetch, hours_to_analyze)

    with col2:
        st.markdown("## 📍 Live Dashboard")
        render_status_metrics(hours_to_analyze, days_to_fetch)

        # Show main tabs (charts/metrics) if calculations have been done
        if hasattr(st.session_state, 'show_plots') and st.session_state.show_plots:
            render_tabs(hours_to_analyze)



if __name__ == "__main__":
    main()
