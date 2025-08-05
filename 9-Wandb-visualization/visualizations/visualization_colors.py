# Centralized color definitions for all visualizations
COLORS = {
    # GPU-based configurations (gradients from light to dark green/teal)
    "LUMI 1 GPU": "#90EE90",  # light green
    "LUMI 2 GPUs": "#7CCD7C",  # medium light green
    "LUMI 4 GPUs": "#6AB06A",  # medium green
    "LUMI 8 GPUs": "#008080",  # teal
    "LUMI 12 GPUs": "#006666",  # darker teal
    "LUMI 16 GPUs": "#004C4C",  # dark teal
    "LUMI 20 GPUs": "#003232",  # very dark teal
    "LUMI 24 GPUs": "#001919",  # almost black teal
    "LUMI 28 GPUs": "#000F0F",  # black teal
    "LUMI 32 GPUs": "#000000",  # black
    # Keep laptop colors the same
    "Laptop (Multicore)": "#FF8C00",  # dark yellow / orange
    "Laptop (Single)": "#8B0000",  # dark red
    # Legacy node-based colors for backward compatibility
    "LUMI 8-Node": "#008080",  # teal
    "LUMI 4-Node": "#90EE90",  # light green
    "LUMI 1-Node": "#006400",  # dark green
}
LINE_WIDTH = 3
