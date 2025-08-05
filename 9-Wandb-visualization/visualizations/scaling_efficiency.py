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
from extract_data import create_manual_data
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
    return create_manual_data()


def create_scaling_efficiency_mp4():
    """Create scaling efficiency animation as MP4"""
    print("🎬 Creating scaling efficiency MP4...")

    df = load_data()

    # Prepare data - filter to only up to 32 GPUs
    lumi_df = df[~df["is_laptop"]].copy()
    lumi_df = lumi_df[lumi_df["gpus"] <= 32].sort_values("gpus")
    gpu_counts = sorted(lumi_df["gpus"].unique())
    speedups = []
    base_time = lumi_df[lumi_df["gpus"] == min(gpu_counts)]["training_time_hours"].min()
    for n in gpu_counts:
        t = lumi_df[lumi_df["gpus"] == n]["training_time_hours"].min()
        speedup = base_time / t if t else 0
        speedups.append(speedup)

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
    # Move the main annotation to the right outside the graph
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
