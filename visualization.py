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

def plot_alpha_beta_map(alphas, betas, title='Alpha & Beta: Distribution Across Assets'):
    """Creates an enhanced 2D density or scatter plot of alpha vs. beta for crypto assets"""
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
        
        # Annotate each point
        for idx, row in df_ab_latest.iterrows():
            ax.annotate(
                idx,
                (row['alpha'], row['beta']),
                xytext=(5, 5),
                textcoords='offset points',
                color='white',
                fontsize=9,
                alpha=0.9,
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1)
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


# def plot_alpha_beta_map(alphas, betas, title='Alpha & Beta: Distribution Across Assets'):
#     """Creates an enhanced 2D density plot of alpha vs. beta for crypto assets"""
#     try:
#         # Create figure with gridspec for better control
#         fig = plt.figure(figsize=(16, 10))
#         gs = GridSpec(1, 1)
#         ax = fig.add_subplot(gs[0, 0])
        
#         # Get the latest alpha and beta values - modified to handle index correctly
#         alpha_latest = alphas.iloc[-1].to_frame().T
#         beta_latest = betas.iloc[-1].to_frame().T
        
#         # Create a DataFrame with both alpha and beta
#         df_ab_latest = pd.DataFrame({
#             'asset': alpha_latest.columns,
#             'alpha': alpha_latest.iloc[0].values,
#             'beta': beta_latest.iloc[0].values
#         })
#         df_ab_latest.set_index('asset', inplace=True)
        
#         # Set style
#         plt.style.use('dark_background')
#         ax.set_facecolor('#1C1C1C')
#         fig.patch.set_facecolor('#1C1C1C')
        
#         # Create the density plot
#         kde = sns.kdeplot(
#             data=df_ab_latest,
#             x='alpha',
#             y='beta',
#             cmap='rocket_r',
#             fill=True,
#             alpha=0.7,
#             levels=30,
#             thresh=.2,
#             linewidths=0.5,
#             ax=ax
#         )
        
#         # Add colorbar
#         cbar = plt.colorbar(kde.collections[-1], label='Density')
#         cbar.ax.tick_params(labelsize=10)
#         cbar.set_label('Density', size=12, color='white')
        
#         # Add scatter points
#         scatter = sns.scatterplot(
#             data=df_ab_latest,
#             x='alpha',
#             y='beta',
#             color='#FFD700',
#             edgecolor='white',
#             s=150,
#             alpha=0.6,
#             marker='o',
#             ax=ax
#         )
        
#         # Add asset labels
#         for idx, row in df_ab_latest.iterrows():
#             ax.annotate(
#                 idx,
#                 (row['alpha'], row['beta']),
#                 xytext=(5, 5),
#                 textcoords='offset points',
#                 color='white',
#                 fontsize=9,
#                 alpha=0.9,
#                 bbox=dict(
#                     facecolor='black',
#                     alpha=0.4,
#                     edgecolor='none',
#                     pad=1
#                 )
#             )
        
#         # Add zero lines
#         ax.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
#         ax.axvline(x=0, color='white', linestyle='--', alpha=0.3, linewidth=1)
        
#         # Get axis limits and add padding
#         xmin, xmax = ax.get_xlim()
#         ymin, ymax = ax.get_ylim()
#         padding = 0.05
#         x_padding = (xmax - xmin) * padding
#         y_padding = (ymax - ymin) * padding
        
#         # Add quadrant labels
#         quadrant_coords = {
#             'High α\nHigh β': (xmax - x_padding, ymax - y_padding),
#             'High α\nLow β': (xmax - x_padding, ymin + y_padding),
#             'Low α\nHigh β': (xmin + x_padding, ymax - y_padding),
#             'Low α\nLow β': (xmin + x_padding, ymin + y_padding)
#         }
        
#         for label, (x, y) in quadrant_coords.items():
#             ha = 'right' if x > 0 else 'left'
#             va = 'top' if y > 0 else 'bottom'
#             ax.text(
#                 x, y, label,
#                 color='white',
#                 alpha=0.6,
#                 fontsize=10,
#                 ha=ha,
#                 va=va,
#                 bbox=dict(
#                     facecolor='black',
#                     alpha=0.3,
#                     edgecolor='white',
#                     linewidth=0.5,
#                     pad=5
#                 )
#             )
        
#         # Enhance grid and styling
#         ax.grid(True, alpha=0.2)
#         for spine in ax.spines.values():
#             spine.set_color('white')
#             spine.set_alpha(0.3)
        
#         # Set title and labels
#         ax.set_title(title, pad=20, fontsize=16, color='white', fontweight='bold')
#         ax.set_xlabel('Alpha (α)', fontsize=14, color='white', labelpad=10)
#         ax.set_ylabel('Beta (β)', fontsize=14, color='white', labelpad=10)
#         ax.tick_params(colors='white', which='both', labelsize=10)
        
#         # Adjust layout
#         plt.tight_layout()
        
#         return fig
        
#     except Exception as e:
#         st.error(f"Error creating plot: {str(e)}")
#         return None 