"""Centralised visual identity helpers for experiment notebooks.

This module uses the Okabe-Ito colour palette, designed for accessibility and
optimised for readers with colour vision deficiencies (CVD).

Design System Notes
- Bluish Green `#009260` anchors positive and consensus outcomes.
- Dark Gray `#48535A`, Medium Gray `#7F8990`, and Light Gray `#EBEBE4` support typography, baselines, and gridlines.
- Figures target <=12" width, grid-on-y by default, and aligned bar/heatmap annotations.
- Preference colours map consistently across waves and cohorts for direct comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import scienceplots  # Required for v2.0.0+

ColorMap = Dict[str, str]
FontSizeMap = Dict[str, int]
FigureSizeMap = Dict[str, tuple[float, float]]

# Okabe-Ito Color Palette (Color-blind friendly)
# Source: Wong, B. (2011). Points of view: Color blindness. Nature Methods, 8(6), 441.
COLORS: ColorMap = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
    "white": "#FFFFFF",
    
    # Semantic aliases
    "primary": "#000000",
    "secondary": "#575757",
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    
    # Text colors
    "text_main": "#000000",
    "text_light": "#575757",
    
    # Plotting aliases for compatibility and semantic usage
    "primary_green": "#009E73",  # Mapped to Bluish Green
    "primary_blue": "#0072B2",   # Mapped to Blue
    "primary_orange": "#E69F00", # Mapped to Orange
    "accent_1": "#F0E442",       # Mapped to Yellow
    "dark_gray": "#333333",
    "medium_gray": "#777777",
    "light_gray": "#E0E0E0",
    
    # Specific outcomes
    "stayed": "#009E73",    # Bluish Green
    "switched": "#D55E00",  # Vermilion (High contrast to bluish green)
    "highlight": "#009E73", # Bluish Green
}

# Delegate color assignment to scienceplots default cycle
# Colors are automatically assigned from plt.rcParams['axes.prop_cycle']
# This provides colorblind-safe colors without manual mapping (IEEE compliant)
PRINCIPLE_COLORS: Optional[ColorMap] = None

PRINCIPLE_DISPLAY_NAMES: Dict[str, str] = {
    "Max Avg Income": "Max. Avg. Income",
    "Max Avg + Floor": "Max. Avg. + Floor",
    "Max Avg + Range": "Max. Avg. + Range",
    "Max Floor": "Max. Floor",
    "Failure": "Failure",
    "None": "None",
}

PRINCIPLE_ORDER: List[str] = [
    "Max Floor",
    "Max Avg Income",
    "Max Avg + Floor",
    "Max Avg + Range",
]

FONT_SIZES: FontSizeMap = {
    "title": 14,
    "subtitle": 12,
    "axis_label": 11,
    "tick_label": 10,
    "legend": 10,
    "annotation": 9,
}

# Use a standard font available on most systems to ensure portability
FONT_FAMILY = "sans-serif"


def _register_fonts() -> None:
    """Register custom fonts if available (placeholder)."""
    font_dir = Path(__file__).parent / "fonts"
    if not font_dir.exists():
        return

    for font_file in font_dir.glob("*.otf"):
        try:
            fm.fontManager.addfont(str(font_file))
        except Exception:
            pass


# IEEE Publication Standards
# Single-column: 3.5 inches | Double-column: 7.16 inches
# Aspect ratios: 4:3 (default), 3:2, or custom
FIG_SIZES: FigureSizeMap = {
    "single": (3.5, 2.625),      # IEEE single-column (4:3 ratio)
    "double": (7.16, 5.37),      # IEEE double-column (4:3 ratio)
    "triple": (7.16, 2.5),       # IEEE triple-panel (full width, compact)
    "single_tall": (3.5, 4.0),   # IEEE single-column portrait
    "double_wide": (7.16, 3.5),  # IEEE double-column landscape
    "tall": (3.5, 4.5),          # IEEE single-column extra tall
    "wide_single": (7.16, 3.5),  # Alias for double_wide
}

GRID_ALPHA = 0.3
GRID_LINEWIDTH = 0.8
GRID_LINESTYLE = ":"


def _palette_hex() -> List[str]:
    palette_order: Iterable[str] = [
        COLORS["primary_green"],
        COLORS["primary_blue"],
        COLORS["primary_orange"],
        COLORS["reddish_purple"],
    ]
    return list(dict.fromkeys(palette_order))


def apply_theme() -> None:
    """
    Apply SciencePlots 'science' + 'ieee' styles for IEEE journal compliance.

    IEEE Style Characteristics:
    - Single-column figure width (3.5" IEEE standard)
    - Black-and-white compatible color cycles
    - LaTeX rendering for professional typography
    - 600 DPI for publication quality

    Note: Requires LaTeX installation for full rendering.
    """
    _register_fonts()  # Keep custom fonts available for LaTeX compatibility

    # Apply scienceplots base styles (science + IEEE)
    plt.style.use(['science', 'ieee'])

    # Override specific settings for thesis needs
    plt.rcParams.update({
        "figure.dpi": 600,           # IEEE publication standard (600 DPI)
        "savefig.dpi": 600,          # Match figure DPI
        "figure.figsize": (3.5, 2.625),  # IEEE single-column width (4:3 ratio)
    })


def format_principle_label(name: str) -> str:
    """Return display-friendly label for a principle name."""
    return PRINCIPLE_DISPLAY_NAMES.get(name, name)


def format_principle_labels(labels: Iterable[str]) -> List[str]:
    """Vectorised helper to format a sequence of principle names."""
    return [format_principle_label(label) for label in labels]


__all__ = [
    "COLORS",
    "PRINCIPLE_COLORS",
    "PRINCIPLE_DISPLAY_NAMES",
    "PRINCIPLE_ORDER",
    "FONT_SIZES",
    "FIG_SIZES",
    "FONT_FAMILY",
    "apply_theme",
    "format_principle_label",
    "format_principle_labels",
]
