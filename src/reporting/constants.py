"""Constants for reporting and graph generation."""

# Constants for report formatting
MAX_FUNCTIONS_PER_CATEGORY: int = 5
MAX_DESCRIPTION_LENGTH: int = 80

# Graph configuration and styling constants
LARGE_FONT_THRESHOLD: int = 20
MEDIUM_FONT_THRESHOLD: int = 16
SVG_DPI: int = 72
START_NODE_DIMENSION: float = 0.5
DEFAULT_LINE_MAX_LENGTH: int = 55

# Color schemes for different depth levels in graphs
# Colors are defined as "fillcolor_gradient_start:fillcolor_gradient_end", "font_and_border_color"
COLOR_SCHEMES: dict[int, dict[str, str]] = {
    0: {"fillcolor": "#E3F2FD:#BBDEFB", "color": "#2196F3"},  # Blue theme
    1: {"fillcolor": "#E8F5E9:#C8E6C9", "color": "#4CAF50"},  # Green theme
    2: {"fillcolor": "#FFF8E1:#FFECB3", "color": "#FFC107"},  # Amber theme
    3: {"fillcolor": "#FFEBEE:#FFCDD2", "color": "#F44336"},  # Red theme
    4: {"fillcolor": "#F3E5F5:#E1BEE7", "color": "#9C27B0"},  # Purple theme
    5: {"fillcolor": "#E0F7FA:#B2EBF2", "color": "#00BCD4"},  # Cyan theme
    6: {"fillcolor": "#FFFDE7:#FFF9C4", "color": "#FFEB3B"},  # Yellow theme
    7: {"fillcolor": "#FBE9E7:#FFCCBC", "color": "#FF5722"},  # Deep Orange theme
    8: {"fillcolor": "#E8EAF6:#C5CAE9", "color": "#3F51B5"},  # Indigo theme
    9: {"fillcolor": "#F1F8E9:#DCEDC8", "color": "#8BC34A"},  # Light Green theme
}

# Layout configuration constants
LARGE_FONT_LAYOUT = {
    "pad": "0.3",
    "nodesep": "0.3",
    "ranksep": "0.5",
    "node_margin": "0.1,0.08",
    "splines": "ortho",
    "overlap": "compress",
}

COMPACT_LAYOUT = {
    "pad": "0.4",
    "nodesep": "0.4",
    "ranksep": "0.7",
    "node_margin": "0.15,0.1",
    "splines": "ortho",
    "overlap": "compress",
}
