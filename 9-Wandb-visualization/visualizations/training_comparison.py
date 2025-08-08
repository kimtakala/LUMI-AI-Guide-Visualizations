#!/usr/bin/env python3
"""
Matplotlib training time comparison animation for LUMI vs Laptop benchmarking.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from pathlib import Path
from visualization_colors import COLORS

# Set up output directory
OUTPUT_DIR = Path(__file__).parent / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure matplotlib for better quality
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14


def load_data():
    """Load benchmark data from CSV file"""
    csv_path = Path(__file__).parent.parent / "charts" / "benchmark_data.csv"

    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
        print("Run extract_data.py first to generate benchmark data.")
        return None

    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def format_time_human_readable(hours):
    """Format time in human-readable format"""
    if hours >= 10:
        # 10+ hours: just show hours
        return f"{hours:.0f}h"
    elif hours >= 1:
        # 1-10 hours: show hours and minutes
        h = int(hours)
        m = int((hours - h) * 60)
        if m == 0:
            return f"{h}h"
        else:
            return f"{h}h {m}m"
    else:
        # Less than 1 hour: show minutes
        minutes = hours * 60
        if minutes >= 1:
            return f"{minutes:.0f}m"
        else:
            # Less than 1 minute: show seconds
            seconds = minutes * 60
            return f"{seconds:.0f}s"


def create_training_comparison_mp4():
    """Create training time comparison animation as MP4"""
    print("🎬 Creating training comparison MP4...")

    df = load_data()
    if df is None:
        return

    # Create clean_name field based on GPU count for LUMI runs, simplify laptop names
    def create_clean_name(row):
        if row["is_laptop"]:
            # Simplify laptop names to match color keys
            if "Multicore" in row["run_name"]:
                return "Laptop (Multicore)"
            elif "Benchmark" in row["run_name"]:
                return "Laptop (Single)"
            else:
                return "Laptop"
        else:
            # LUMI runs: use GPU count instead of node count
            gpu_count = row["total_gpus"]
            if pd.notna(gpu_count) and gpu_count > 0:
                return f"LUMI {int(gpu_count)} GPU"
            else:
                # Fallback to original name if no GPU count
                return row["run_name"]

    df["clean_name"] = df.apply(create_clean_name, axis=1)

    # Include all runs (no filtering by node count)
    df_filtered = df[df["training_time_hours"].notna()].copy()

    if len(df_filtered) == 0:
        print("❌ No data found for comparison!")
        return

    # Prepare data
    df_sorted = df_filtered.sort_values("training_time_hours")
    names = df_sorted["clean_name"].tolist()
    times = df_sorted["training_time_hours"].tolist()
    colors = [COLORS.get(name, "#333") for name in names]

    print(f"📊 Comparing {len(names)} configurations")

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_yscale("log")  # Use logarithmic scale for y-axis
    ax.set_ylim(min(times) * 0.5, max(times) * 2)
    ax.set_xlabel("Hardware Configuration")
    ax.set_ylabel("Training Time - Log Scale")
    ax.set_title("LUMI vs Laptop: Training Time Comparison")

    # Create bars
    bars = ax.bar(names, [min(times) * 0.1] * len(names), color=colors, alpha=0.8)

    # Rotate x-axis labels for better readability and ensure they fit
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    # Add extra space at bottom for rotated labels
    plt.subplots_adjust(bottom=0.2)

    # Add value labels
    value_texts = []
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        text = ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(times) * 0.1,
            "",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        value_texts.append(text)

    # Add legend box positioned in top right corner of the plot area
    ax.text(
        0.02,
        0.98,
        "Performance Insights\n• LUMI GPUs are orders of magnitude faster\n• Laptop training takes days vs hours\n• Log scale shows dramatic differences\n• GPU scaling shows clear benefits",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Animation configuration for constant speed and 10 second duration
    min_time = min(times)
    max_time = max(times)

    # Target 10 seconds at 30 fps = 300 frames
    total_frames = 300
    print(f"Animation will be {total_frames} frames ({total_frames/30:.1f} seconds)")

    def animate(frame):
        # Log scale progression: each frame represents a constant portion of log space
        progress = frame / (total_frames - 1)  # 0 to 1

        # Calculate current threshold using logarithmic interpolation
        log_min = np.log10(min_time)
        log_max = np.log10(max_time)
        current_log_threshold = log_min + progress * (log_max - log_min)
        current_max_time = 10**current_log_threshold

        for i, (bar, time_val, text) in enumerate(zip(bars, times, value_texts)):
            # Each bar grows until it reaches its final height or the current time limit
            current_height = max(min(time_val, current_max_time), min(times) * 0.1)
            bar.set_height(current_height)

            # Update text
            if current_height > min(times) * 0.2:
                text.set_text(format_time_human_readable(current_height))
                text.set_position(
                    (
                        bar.get_x() + bar.get_width() / 2,
                        current_height * 1.1,
                    )
                )
            else:
                text.set_text("")

        return list(bars) + value_texts

    # Create animation with constant speed
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=33, blit=True, repeat=True
    )

    # Save as MP4 with proper error handling
    output_file = OUTPUT_DIR / "training_comparison.mp4"
    try:
        anim.save(output_file, writer="ffmpeg", fps=30, bitrate=1800)
        plt.close()
        print(f"✅ Saved training comparison MP4 to {output_file}")
    except KeyboardInterrupt:
        plt.close()
        print("\n🛑 Animation generation cancelled by user")
        if output_file.exists():
            output_file.unlink()  # Remove incomplete file
        return
    except Exception as e:
        plt.close()
        print(f"❌ Error saving animation: {e}")
        return


if __name__ == "__main__":
    try:
        create_training_comparison_mp4()
    except KeyboardInterrupt:
        print("\n🛑 Animation generation cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
