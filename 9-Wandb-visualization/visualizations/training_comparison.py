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
    return


def create_training_comparison_mp4():
    """Create training time comparison animation as MP4"""
    print("🎬 Creating training comparison MP4...")

    df = load_data()

    # Filter to only include up to 4 nodes (32 GPUs) and laptop
    df_filtered = df[
        (df["is_laptop"]) | ((~df["is_laptop"]) & (df["nodes"] <= 4))
    ].copy()

    # Prepare data
    df_sorted = df_filtered.sort_values("training_time_hours")
    names = df_sorted["clean_name"].tolist()
    times = df_sorted["training_time_hours"].tolist()
    colors = [COLORS.get(name, "#333") for name in names]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_yscale("log")  # Use logarithmic scale for y-axis
    ax.set_ylim(min(times) * 0.5, max(times) * 2)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Training Time (Hours) - Log Scale")
    ax.set_title("LUMI vs Laptop: Training Time Comparison (Up to 4 Nodes)")

    # Create bars
    bars = ax.bar(names, [min(times) * 0.1] * len(names), color=colors, alpha=0.8)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

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

    # Add legend box to top right outside of graph
    ax.text(
        1.05,
        0.95,
        "Performance Insights\n• LUMI GPUs are orders of magnitude faster\n• Laptop training takes days vs hours\n• Log scale shows dramatic differences\n• Tested up to 32 GPUs (4 nodes)",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.subplots_adjust(right=0.75)  # Make room for legend box

    # Calculate animation timing based on completion count
    # No hard frame limit - animation ends when last bar completes
    # Individual speed multipliers for each completion + overall speed scaling
    min_time = min(times)
    max_time = max(times)

    # Configuration: Targeted speed control
    # The key insight: slow progression for first 3 bars, then speed up for final 2
    base_time_per_frame = 0.002  # Very slow base progression (hours per frame)
    speedup_after_third = (
        200.0  # MAIN CONTROL: Speed multiplier after 3rd bar completes
    )
    overall_speed_factor = 1.0  # Global speed adjustment (keep at 1.0 unless needed)

    # Calculate total frames needed for this configuration
    def calculate_total_frames():
        """Calculate how many frames needed to complete with current settings"""
        effective_base_rate = base_time_per_frame * overall_speed_factor

        sorted_times = sorted(times)
        current_time = 0
        frames_used = 0

        while current_time < max_time:
            # Count completed bars at current time
            completed_count = sum(1 for t in sorted_times if t <= current_time)

            # Simple rule: slow until 3rd completion, then fast
            if completed_count >= 3:
                current_rate = effective_base_rate * speedup_after_third
            else:
                current_rate = effective_base_rate

            # Advance one frame
            current_time += current_rate
            frames_used += 1

            # Safety check
            if frames_used > 5000:  # Max 5000 frames
                break

        return frames_used

    total_frames = calculate_total_frames()
    print(f"Animation will be {total_frames} frames ({total_frames/30:.1f} seconds)")

    # Create time mapping for variable speed animation
    def get_current_time_limit(frame):
        """Calculate time limit for given frame"""
        effective_base_rate = base_time_per_frame * overall_speed_factor

        sorted_times = sorted(times)
        current_time = 0
        current_frame = 0

        while current_frame < frame and current_time < max_time:
            # Count completed bars at current time
            completed_count = sum(1 for t in sorted_times if t <= current_time)

            # Simple rule: slow until 3rd completion, then fast
            if completed_count >= 3:
                current_rate = effective_base_rate * speedup_after_third
            else:
                current_rate = effective_base_rate

            # Advance one frame
            current_time += current_rate
            current_frame += 1

        return min(current_time, max_time)

    def animate(frame):
        current_max_time = get_current_time_limit(frame)

        for i, (bar, time_val, text) in enumerate(zip(bars, times, value_texts)):
            # Each bar grows until it reaches its final height or the current time limit
            current_height = max(min(time_val, current_max_time), min(times) * 0.1)
            bar.set_height(current_height)

            # Update text
            if current_height > min(times) * 0.2:
                if current_height < 1:
                    text.set_text(f"{current_height:.3f}h")
                else:
                    text.set_text(f"{current_height:.2f}h")
                text.set_position(
                    (
                        bar.get_x() + bar.get_width() / 2,
                        current_height * 1.1,
                    )
                )
            else:
                text.set_text("")

        return list(bars) + value_texts

    # Create animation with dynamic frame count
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
