import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_alpha_beta_map(alphas, betas, title='Alpha & Beta: Distribution Across Assets', show_labels=True):
    """
    Creates an enhanced 2D density or scatter plot of alpha vs. beta for crypto assets
    
    Parameters:
    -----------
    alphas : pd.DataFrame
        DataFrame containing alpha values
    betas : pd.DataFrame
        DataFrame containing beta values
    title : str
        Plot title
    show_labels : bool
        Whether to show asset labels on the plot
    """
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        
        # Get the latest alpha and beta
        alpha_latest = alphas.iloc[-1].to_frame().T
        beta_latest = betas.iloc[-1].to_frame().T
        
        df_ab_latest = pd.DataFrame({
            'asset': alpha_latest.columns,
            'alpha': alpha_latest.iloc[0].values,
            'beta': beta_latest.iloc[0].values
        })
        df_ab_latest.set_index('asset', inplace=True)
        
        # Style
        plt.style.use('dark_background')
        ax.set_facecolor('#1C1C1C')
        fig.patch.set_facecolor('#1C1C1C')
        
        # Main scatter
        scatter = sns.scatterplot(
            data=df_ab_latest,
            x='alpha',
            y='beta',
            color='#FFD700',
            edgecolor='white',
            s=150,
            alpha=0.6,
            marker='o',
            ax=ax,
            label='All Assets'
        )
        
        # Highlight BTCDOMUSDT & USDCUSDT
        special_symbols = df_ab_latest[df_ab_latest.index.isin(['BTCDOMUSDT', 'USDCUSDT'])]
        if not special_symbols.empty:
            sns.scatterplot(
                data=special_symbols,
                x='alpha',
                y='beta',
                color='red',
                edgecolor='white',
                s=300,         # bigger marker
                alpha=0.9,
                marker='o',
                ax=ax,
                label='BTCDOM & USDC'
            )
        
        # Improved label placement strategy
        if show_labels:
            # Calculate point density for each point
            for idx, row in df_ab_latest.iterrows():
                # Count nearby points within a radius
                nearby_points = df_ab_latest[
                    (abs(df_ab_latest['alpha'] - row['alpha']) < 0.0005) &  # Reduced radius
                    (abs(df_ab_latest['beta'] - row['beta']) < 0.05)  # Adjusted for beta scale
                ]
                
                # Always label if:
                # 1. Point is isolated (no nearby points)
                # 2. Point is special (BTCDOM/USDC)
                # 3. Point is in a small cluster (2-3 points)
                should_label = (
                    len(nearby_points) <= 3 or  # Small cluster
                    len(nearby_points) == 1 or  # Isolated point
                    idx in ['BTCDOMUSDT', 'USDCUSDT']  # Special symbols
                )
                
                if should_label:
                    # Calculate label position to avoid overlaps
                    xytext = (5, 5)  # Default offset
                    if row['beta'] > df_ab_latest['beta'].mean():
                        xytext = (5, -10)  # Place below for points above mean
                    
                    ax.annotate(
                        idx,
                        (row['alpha'], row['beta']),
                        xytext=xytext,
                        textcoords='offset points',
                        color='white',
                        fontsize=9,
                        alpha=0.9,
                        bbox=dict(
                            facecolor='black',
                            alpha=0.4,
                            edgecolor='none',
                            pad=1
                        )
                    )
        
        # Zero lines
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        
        ax.set_title(title, pad=20, fontsize=16, color='white', fontweight='bold')
        ax.set_xlabel('Alpha (α)', fontsize=14, color='white', labelpad=10)
        ax.set_ylabel('Beta (β)', fontsize=14, color='white', labelpad=10)
        ax.tick_params(colors='white', labelsize=10)
        
        ax.legend(facecolor='#1C1C1C', edgecolor='white', fontsize=10)
        
        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None




def plot_market_regimes(regime_df):
    """Plot market regimes with probability heatmap"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Price with regime shading
    regimes = regime_df['regime_label']
    colors = {'High Volatility': '#FF6B6B', 'Bull Market': '#4ECDC4', 'Bear Market': '#556270'}
    
    last_regime = None
    for i, (date, row) in enumerate(regime_df.iterrows()):
        if row['regime_label'] != last_regime:
            ax1.axvspan(date, regime_df.index[i+1] if i+1 < len(regime_df) else date, 
                       color=colors[row['regime_label']], alpha=0.3)
            last_regime = row['regime_label']
    ax1.plot(regime_df['price'], color='#2E294E', lw=2)
    
    # Volatility surface
    x = regime_df.index
    y = regime_df['volatility']
    ax2.fill_between(x, y, color='#C5D86D', alpha=0.3)
    ax2.plot(x, y, color='#1B5299', lw=2)
    
    # Styling
    ax1.set_title('Market Regime Detection', color='white', fontsize=16)
    ax2.set_xlabel('Date', color='white')
    ax1.set_facecolor('#1C1C1C')
    ax2.set_facecolor('#1C1C1C')
    fig.patch.set_facecolor('#1C1C1C')
    
    return fig 