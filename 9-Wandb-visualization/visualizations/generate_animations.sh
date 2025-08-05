#!/bin/bash
# Generate all MP4 animations using matplotlib
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source ../visualization_env/bin/activate

echo "🎬 Generating all matplotlib MP4 animations..."

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: ffmpeg not found. Please install ffmpeg to create MP4 animations."
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   macOS: brew install ffmpeg"
    exit 1
fi

# Generate each animation
echo "🎯 Running scaling efficiency animation..."
python scaling_efficiency.py

echo "📊 Running training comparison animation..."
python training_comparison.py

echo "📈 Running accuracy vs time animation..."
python accuracy_vs_time.py

echo "🏁 Running progress race animation..."
python progress_race.py

echo ""
echo "🎉 All MP4 animations generated successfully!"
echo "📁 Files created in $SCRIPT_DIR/charts/:"
ls -la charts/*.mp4 2>/dev/null || echo "   (No MP4 files found)"
