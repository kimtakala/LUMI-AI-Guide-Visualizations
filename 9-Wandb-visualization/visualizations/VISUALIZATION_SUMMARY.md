# LUMI vs Laptop: Visualization Suite

## Overview

This suite visualizes the performance gap between LUMI distributed training and local laptop runs using clear, presentation-ready charts and animations. All visualizations are reproducible and based on identical model, dataset, and hyperparameters.

## Visualizations

- **Static Comparison**: Training time and accuracy comparisons. `.png` files in `charts/` are exported directly from Wandb (`1-32GPU_batch_loss.png`, `1-32GPU_epochs.png`, `1-32GPU_learning_rate.png`, `1-32GPU_running_accuracy.png`, `1-32GPU_train_loss.png`).
- **Animated comparison**: `training_comparison.mp4`.
- **Animated Progress Race**: Real-time race between LUMI and laptop training (`progress_race.mp4`).
- **Accuracy vs Time**: Animated accuracy and loss curves (`1-32GPU_running_accuracy.png`, `1-32GPU_train_loss.png`).
- **Scaling Efficiency**: Static and animated scaling analyses (`scaling_efficiency.mp4`, `1-32GPU_epochs.png`).

## Key Metrics

- **LUMI 8-Node**: 3.3 min | **Laptop**: 54–76 hrs
- **Max Speedup**: 1,393x (LUMI vs laptop)
- **Best Accuracy**: 98.4% (LUMI 8-GPU, 16 min) vs 96% (laptop, 76 hrs [extrapolated])

## Usage

1. Open PNG for static charts (image viewer)
2. Open MP4 for animated charts (video player)
3. To extract data:

   ```bash
   cd 9-Wandb-visualization
   source visualization_env/bin/activate
   python extract_data.py
   ```

4. To visualize data:

   _note:_
   make sure you have ffmpeg installed for animation output.

   ```
   Install ffmpeg:
   Ubuntu/Debian: sudo apt install ffmpeg
   macOS: brew install ffmpeg
   ```

   ```bash
   cd 9-Wandb-visualization
   source visualization_env/bin/activate
   ./visualizations/generate_animations.sh
   # or alternatively:
   python visualizations/progress_race.py  # etc.
   ```

   output will be generated to: `visualizations/charts`

## Notes

- `.png` files in `charts/` are direct exports from Wandb.

---

**Frameworks**: PyTorch, Wandb, Matplotlib, Pandas, H5PY
**Environment**: LUMI supercomputer, Singularity
**Dataset**: ImageNet subset
