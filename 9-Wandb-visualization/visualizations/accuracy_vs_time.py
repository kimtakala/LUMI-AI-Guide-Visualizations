#!/usr/bin/env python3
"""
Matplotlib accuracy vs time scatter plot for LUMI vs Laptop benchmarking.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from pathlib import Path
from extract_data import create_manual_data
from visualization_colors import COLORS

# Set up output directory
OUTPUT_DIR = Path(__file__).parent / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure matplotlib for better quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def load_data():
    return create_manual_data()

def create_accuracy_vs_time_mp4():
    """Create accuracy vs time scatter plot as MP4"""
    print("🎬 Creating accuracy vs time MP4...")
    
    df = load_data()
    
    # Prepare data
    df_with_acc = df[df['accuracy'].notna()].copy()
    df_with_acc = df_with_acc.sort_values('training_time_hours')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xscale('log')
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Training Time (Hours, Log Scale)')
    ax.set_ylabel('Final Training Accuracy (%)')
    ax.set_title('Training Accuracy vs Time: LUMI vs Laptop')
    ax.grid(True, alpha=0.3)
    
    # Add annotations outside the plot area
    ax.text(1.15, 0.7, 'Efficiency Insights\n• LUMI achieves higher accuracy\n• In dramatically less time\n• Laptop requires hours for similar accuracy', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.text(1.15, 0.3, 'About This Visualization\n• X-axis: Training time (log scale)\n• Y-axis: Final accuracy achieved\n• Ideal: High accuracy, low time\n• (top-left corner is best)', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.subplots_adjust(right=0.75)  # Make room for annotations
    
    def animate(frame):
        ax.clear()
        ax.set_xscale('log')
        ax.set_xlim(0.01, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Training Time (Hours, Log Scale)')
        ax.set_ylabel('Final Training Accuracy (%)')
        ax.set_title('Training Accuracy vs Time: LUMI vs Laptop')
        ax.grid(True, alpha=0.3)
        
        # Show first (frame+1) points
        points_to_show = min(frame + 1, len(df_with_acc))
        current_df = df_with_acc.iloc[:points_to_show]
        
        for _, row in current_df.iterrows():
            color = COLORS.get(row['clean_name'], '#333')
            ax.scatter(row['training_time_hours'], row['accuracy'], 
                      s=150, color=color, alpha=0.8, edgecolors='black', linewidth=2)
            ax.text(row['training_time_hours'], row['accuracy'], 
                   f"  {row['clean_name']}", va='center', ha='left', fontsize=10)
        
        return ax.collections + ax.texts
    
    # Create animation
    frames = len(df_with_acc)
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                 interval=1000, blit=False, repeat=True)
    
    # Save as MP4
    output_file = OUTPUT_DIR / 'accuracy_vs_time.mp4'
    anim.save(output_file, writer='ffmpeg', fps=30, bitrate=1800)
    plt.close()
    print(f"✅ Saved accuracy vs time MP4 to {output_file}")

if __name__ == "__main__":
    create_accuracy_vs_time_mp4()
