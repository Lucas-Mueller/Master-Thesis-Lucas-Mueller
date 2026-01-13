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
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import scienceplots  # Register scienceplots styles

# Configuration for scienceplots styles
# e.g. ['science', 'ieee'] for IEEE standard, or ['science', 'nature'] for Nature
# Styles to apply (in order). 
# 'std-colors' provides the vibrant color cycle (Blue, Green, Orange, Red, Purple...)
SCIENCE_PLOT_STYLES: List[str] = ["science", "ieee", "std-colors"]

ColorMap = Dict[str, str]
FontSizeMap = Dict[str, int]
FigureSizeMap = Dict[str, tuple[float, float]]

# Map semantic keys to the SciencePlots 'std-colors' cycle (C0, C1, etc.)
# C0: Blue (#0C5DA5) - Primary / Neutral
# C1: Green (#00B945) - Positive / Stayed
# C2: Orange (#FF9500) - Warning / Switched / Difference
# C3: Red (#FF2C00) - Error / Negative
# C4: Purple (#845B97) - Secondary
COLORS: ColorMap = {
    # Semantics
    "primary_blue": "C0",    # was #0072B2 (Okabe Blue)
    "primary_green": "C1",   # was #009E73 (Okabe Green)
    "primary_orange": "C2",  # New: Orange for warning/difference
    "primary_red": "C3",     # was #D55E00 (Okabe Red)
    "reddish_purple": "C4",  # New: Purple for secondary accent
    "light_gray": "#E6E6E6", # Keep grays hardcoded as they aren't in the cycle usually
    "medium_gray": "#999999",
    "dark_gray": "#333333",
    "black": "#000000",
    "white": "#FFFFFF",
    
    # Text/Grid
    "text_main": "#000000",
    "text_light": "#575757",
    "grid": "#E0E0E0",
    "background": "#FFFFFF",
    
    # Domain-specific
    "stayed": "C1",          # Green for stability/consensus
    "switched": "C2",        # Orange for change/instability (Visual distinction from C1)
}

PRINCIPLE_COLORS: ColorMap = {
    "Max Avg Income": COLORS["primary_green"],
    "Max Avg + Floor": COLORS["primary_blue"],
    "Max Avg + Range": COLORS["primary_orange"],
    "Max Floor": COLORS["reddish_purple"], # Distinct from others
}

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


FIG_SIZES: FigureSizeMap = {
    "single": (9, 5),
    "double": (12, 5),
    "triple": (14, 5),
    "wide_single": (11, 5),
    "tall": (9, 7),
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
    """Apply RC params from scienceplots and our custom overrides."""
    _register_fonts()
    
    # 1. Apply scienceplots styles first (if configured)
    # This sets strict academic standards (fonts, ticks, sizes)
    if SCIENCE_PLOT_STYLES:
        plt.style.use(SCIENCE_PLOT_STYLES)

    # 2. Apply project-specific identity (colors, accessibility)
    # We override only what's necessary to maintain our color identity
    # while respecting the structural styling of scienceplots.
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            # We keep our background colors? Scienceplots usually assumes white.
            # "figure.facecolor": COLORS["background"],
            # "axes.facecolor": COLORS["background"],
            
            "axes.edgecolor": COLORS["medium_gray"],
            "axes.linewidth": 0.8, # Slightly thinner to match science style usually
            
            # Keep our accessible font sizes if they don't conflict too largely
            # Scienceplots often sets these specifically for column widths.
            # We'll trust scienceplots for sizes unless we really need to force them.
            # "axes.labelsize": FONT_SIZES["axis_label"],
            
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["text_main"],
            
            # Scienceplots 'ieee' style usually has no grid. 
            # Uncomment if we explicitly want to force the grid back on.
            # "axes.grid": True,
            # "axes.grid.axis": "y",
            # "grid.alpha": GRID_ALPHA,
            # "grid.linewidth": GRID_LINEWIDTH,
            # "grid.linestyle": GRID_LINESTYLE,
            # "grid.color": COLORS["grid"],
            
            "xtick.color": COLORS["text_main"],
            "ytick.color": COLORS["text_main"],
            "text.color": COLORS["text_main"],
            
            # Ensure we don't accidentally override the font family scienceplots just set
            # "font.family": FONT_FAMILY,
        }
    )
    
    # Set seaborn theme but try to preserve the style we just set
    # sns.set_theme often resets matplotlib rcParams. 
    # NOTE: We use set_context instead of set_theme to avoid overriding the 
    # detailed rcParams set by scienceplots. We DO NOT set the palette here,
    # relying on the 'std-colors' style cycle instead.
    sns.set_context("paper", font_scale=1.0)


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
