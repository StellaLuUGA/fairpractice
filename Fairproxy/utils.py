"""Helper utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_leakage_distribution(df_laft, save_path='./results/leakage_dist.png'):
    """Plot distribution of leakage scores"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Ratio distribution
    axes[0].hist(df_laft['ratio'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(1.0, color='red', linestyle='--', label='No leakage (ratio=1)')
    axes[0].set_xlabel('Distance Ratio')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Distance Ratios')
    axes[0].legend()
    
    # Leakage score distribution
    axes[1].hist(df_laft['leakage_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='No leakage (score=0)')
    axes[1].set_xlabel('Leakage Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Leakage Scores')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")