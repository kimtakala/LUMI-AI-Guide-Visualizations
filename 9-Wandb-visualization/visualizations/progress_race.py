#!/usr/bin/env python3
"""
Matplotlib progress race animation for LUMI vs Laptop benchmarking.
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

        # Create clean_name field based on GPU count for LUMI runs, keep original for laptops
        def create_clean_name(row):
            if row["is_laptop"]:
                # Keep laptop names as they are
                return row["run_name"]
            else:
                # LUMI runs: use GPU count instead of node count
                gpu_count = row["total_gpus"]
                if pd.notna(gpu_count) and gpu_count > 0:
                    return f"LUMI {int(gpu_count)} GPU"
                else:
                    # Fallback to original name if no GPU count
                    return row["run_name"]

        df["clean_name"] = df.apply(create_clean_name, axis=1)

        # Debug: print the clean names being generated
        print("Clean names generated:")
        for name in sorted(df["clean_name"].unique()):
            print(f"  '{name}'")

        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def create_progress_race_mp4():
    """Create progress race animation as MP4"""
    print("🎬 Creating progress race MP4...")

    df = load_data()

    # Prepare data - use training_time_hours and sort by it
    df_sorted = df.sort_values("training_time_hours")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel("Time (Log Scale)")
    ax.set_ylabel("Training Progress (%)")
    ax.set_title("LUMI vs Laptop: Training Progress Race")
    ax.set_xscale("log")
    ax.set_xlim(0.001, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Set custom tick labels for human-readable time
    def format_time_label(hours):
        if hours < 1 / 360:  # Less than 10 seconds
            return "start"
        elif hours < 1 / 60:  # Less than 1 minute, show in seconds
            seconds = hours * 3600
            if seconds < 10:
                return f"{seconds:.0f} seconds"
            else:
                return f"{int(seconds//10)*10} seconds"
        elif hours < 1:  # Less than 1 hour, show in minutes
            minutes = hours * 60
            return f"{int(minutes)} minutes"
        elif hours < 24:  # Less than 1 day, show in hours
            if hours == 1:
                return f"{hours:.0f} hour"
            else:
                return f"{hours:.0f} hours"
        else:  # 1 day or more
            days = hours / 24
            return f"{days:.0f} days"

    # Define tick positions and labels
    tick_positions = [0.001, 0.01, 0.1, 1, 10, 100]
    tick_labels = [format_time_label(pos) for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Generate time points for animation (start at 0 for true zero progress)
    num_frames = 150  # Reduce frames for faster rendering (5 seconds at 30fps)
    max_time = df_sorted["training_time_hours"].max()
    time_points = np.logspace(np.log10(0.001), np.log10(max_time), num_frames)

    # Pre-calculate the complete curves for each training run
    training_curves = {}
    for _, row in df_sorted.iterrows():
        name = row["clean_name"]
        total_time = row["training_time_hours"]

        # Generate the complete curve with many points for smooth tracing
        num_curve_points = 200
        # Create a proper logarithmic distribution from 0.001 to total_time
        time_curve = np.logspace(
            np.log10(0.001), np.log10(total_time), num_curve_points
        )
        progress_curve = []

        for i, t in enumerate(time_curve):
            t_ratio = t / total_time  # 0 to 1 for full training
            # Pure logarithmic curve from 0% to 100%
            # Use a simple linear mapping: progress = 100 * t_ratio
            p = 100 * t_ratio
            progress_curve.append(p)

        training_curves[name] = {
            "time": np.array(time_curve),
            "progress": np.array(progress_curve),
            "total_time": total_time,
        }

    def animate(frame):
        ax.clear()
        ax.set_xlabel("Time (Log Scale)")
        ax.set_ylabel("Training Progress (%)")
        ax.set_title("LUMI vs Laptop: Training Progress Race")
        ax.set_xscale("log")
        ax.set_xlim(0.001, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Reset custom tick labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        current_time = time_points[frame]

        for _, row in df_sorted.iterrows():
            name = row["clean_name"]
            color = COLORS.get(name, "#333")
            curve_data = training_curves[name]

            # Find how much of the curve to show based on current time
            if current_time < curve_data["total_time"]:
                # Training in progress - show curve up to current time
                mask = curve_data["time"] <= current_time
                if np.any(mask):
                    visible_times = curve_data["time"][mask]
                    visible_progress = curve_data["progress"][mask]
                    ax.plot(
                        visible_times,
                        visible_progress,
                        color=color,
                        linewidth=3,
                        label=name,
                        alpha=0.8,
                    )

                    # Add a dot at the current position for emphasis
                    if len(visible_times) > 0:
                        current_progress = visible_progress[-1]
                        ax.plot(
                            visible_times[-1],
                            current_progress,
                            "o",
                            color=color,
                            markersize=1,
                            alpha=0.9,
                        )
            else:
                # Training complete - show full curve
                ax.plot(
                    curve_data["time"],
                    curve_data["progress"],
                    color=color,
                    linewidth=3,
                    label=name,
                    alpha=0.8,
                )
                # Add completion marker - use larger size for emphasis
                ax.plot(
                    curve_data["time"][-1],
                    curve_data["progress"][-1],
                    "o",
                    color=color,
                    markersize=8,
                    alpha=0.9,
                    markeredgecolor="white",
                    markeredgewidth=1,
                )

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return ax.lines

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames, interval=33, blit=False, repeat=True
    )

    # Save as MP4
    output_file = OUTPUT_DIR / "progress_race.mp4"
    anim.save(output_file, writer="ffmpeg", fps=30, bitrate=1800)
    plt.close()
    print(f"✅ Saved progress race MP4 to {output_file}")


if __name__ == "__main__":
    create_progress_race_mp4()
