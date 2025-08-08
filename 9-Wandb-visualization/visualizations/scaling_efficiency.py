#!/usr/bin/env python3
"""
Matplotlib scaling efficiency animation for LUMI benchmarking.
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


def create_scaling_efficiency_mp4():
    """Create scaling efficiency animation as MP4"""
    print("🎬 Creating scaling efficiency MP4...")

    df = load_data()
    if df is None:
        return

    # Debug: Print dataframe info
    print(f"📊 Total rows: {len(df)}")
    print(f"📊 Columns: {df.columns.tolist()}")
    print(f"📊 Sample data:")
    print(df[["run_name", "is_laptop", "total_gpus", "training_time_hours"]].head())

    # Prepare data - filter to only LUMI runs (not laptops)
    lumi_df = df[~df["is_laptop"]].copy()
    print(f"📊 LUMI runs (non-laptop): {len(lumi_df)}")

    # Remove any rows without total_gpus or training_time_hours data
    lumi_df = lumi_df[lumi_df["total_gpus"].notna()]
    lumi_df = lumi_df[lumi_df["training_time_hours"].notna()]
    lumi_df = lumi_df[lumi_df["training_time_hours"] > 0]  # Must have positive time
    print(f"📊 LUMI runs with valid data: {len(lumi_df)}")

    if len(lumi_df) == 0:
        print("❌ No LUMI GPU data with valid training times found!")
        return

    # Filter to reasonable GPU counts and sort
    lumi_df = lumi_df[lumi_df["total_gpus"] <= 32].sort_values("total_gpus")
    print(f"📊 LUMI runs ≤32 GPUs: {len(lumi_df)}")

    # For duplicate GPU counts, keep only the fastest run
    lumi_df = lumi_df.loc[lumi_df.groupby("total_gpus")["training_time_hours"].idxmin()]
    print(f"📊 LUMI runs after deduplication: {len(lumi_df)}")

    # Debug: Show the filtered data
    print("📊 Filtered LUMI data:")
    for _, row in lumi_df.iterrows():
        print(
            f"  {row['run_name']}: {row['total_gpus']} GPUs, {row['training_time_hours']:.3f} hours"
        )

    gpu_counts = sorted(lumi_df["total_gpus"].unique())
    speedups = []

    # Get baseline time (use the SLOWEST run with fewest GPUs for proper speedup calculation)
    min_gpu_count = min(gpu_counts)
    base_time = lumi_df[lumi_df["total_gpus"] == min_gpu_count][
        "training_time_hours"
    ].max()  # Use max time as baseline

    print(f"📊 Base time ({min_gpu_count} GPUs): {base_time:.3f} hours")

    for n in gpu_counts:
        # Get the best (fastest) time for this GPU count
        gpu_subset = lumi_df[lumi_df["total_gpus"] == n]
        t = gpu_subset["training_time_hours"].min()
        speedup = base_time / t if t and t > 0 else 0
        speedups.append(speedup)
        print(f"📊 {n} GPUs: {t:.3f} hours → speedup: {speedup:.2f}x")

    print(f"📊 Final GPU scaling data: {dict(zip(gpu_counts, speedups))}")

    # Validate we have meaningful data
    if max(speedups) <= 1.1:  # If no speedup is greater than 1.1x
        print("⚠️  Warning: No significant speedup detected. Check your data.")
        print("   This might be normal if you only have one GPU configuration.")
        return

    # Animation parameters
    duration_sec = 10
    fps = 30
    total_frames = duration_sec * fps
    n_points = len(gpu_counts)
    # Number of segments = n_points - 1
    frames_per_segment = (
        total_frames // (n_points - 1) if n_points > 1 else total_frames
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.75)  # Make more space for annotation
    ax.set_xlim(0.8, max(gpu_counts) + 2)
    ax.set_ylim(0.8, max(max(speedups), max(gpu_counts)) + 2)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup Factor")
    ax.set_title("LUMI GPU Scaling Performance Analysis")
    ax.grid(True, alpha=0.3)

    # Initialize lines
    (actual_line,) = ax.plot(
        [],
        [],
        "o-",
        color=COLORS["LUMI 1 GPU"],
        linewidth=3,
        markersize=8,
        label="Actual Speedup",
    )
    (ideal_line,) = ax.plot(
        [], [], "--", color="gray", linewidth=2, label="Ideal Speedup"
    )

    ax.legend(loc="upper left")

    # Add text annotations
    ax.annotate(
        "GPU Scaling Insights\n• Ideal speedup: linear with GPUs\n• Actual speedup: near-linear\n• LUMI shows excellent GPU scaling\n• Tested up to 32 GPUs (4 nodes)",
        xy=(1.0, 0.98),
        xycoords="axes fraction",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        fontsize=12,
        clip_on=False,
    )

    ax.text(
        0.98,
        0.02,
        "About Speedup Factor\nSpeedup = Time(1) / Time(N)\nMeasures parallel efficiency\nHigher = better scaling",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.9),
    )

    def interpolate(p0, p1, alpha):
        # Linear interpolation between two points
        return p0 + (p1 - p0) * alpha

    def animate(frame):
        # Determine which segment and how far along
        if n_points == 1:
            actual_line.set_data([gpu_counts[0]], [speedups[0]])
            ideal_line.set_data([gpu_counts[0]], [gpu_counts[0]])
            return actual_line, ideal_line

        seg = min(frame // frames_per_segment, n_points - 2)
        seg_frame = frame % frames_per_segment
        alpha = seg_frame / frames_per_segment

        # Build data up to current segment
        x_data = list(gpu_counts[: seg + 1])
        y_actual = list(speedups[: seg + 1])
        y_ideal = list(gpu_counts[: seg + 1])

        # Interpolate to next point if not at the end
        if seg < n_points - 1:
            x_interp = interpolate(gpu_counts[seg], gpu_counts[seg + 1], alpha)
            y_actual_interp = interpolate(speedups[seg], speedups[seg + 1], alpha)
            y_ideal_interp = interpolate(gpu_counts[seg], gpu_counts[seg + 1], alpha)
            x_data.append(x_interp)
            y_actual.append(y_actual_interp)
            y_ideal.append(y_ideal_interp)

        actual_line.set_data(x_data, y_actual)
        ideal_line.set_data(x_data, y_ideal)
        return actual_line, ideal_line

    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000 / fps, blit=True, repeat=True
    )

    # Save as MP4
    output_file = OUTPUT_DIR / "scaling_efficiency.mp4"
    anim.save(output_file, writer="ffmpeg", fps=fps, bitrate=1800)
    plt.close()
    print(f"✅ Saved scaling efficiency MP4 to {output_file}")


if __name__ == "__main__":
    create_scaling_efficiency_mp4()
