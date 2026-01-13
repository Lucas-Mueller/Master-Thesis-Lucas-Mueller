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

# User requested Linux Libertine
FONT_FAMILY = "serif"


def _register_fonts() -> None:
    """Register custom fonts if available (placeholder)."""
    font_dir = Path(__file__).parent / "fonts"
    if not font_dir.exists():
        return

    for font_file in list(font_dir.glob("*.otf")) + list(font_dir.glob("*.ttf")):
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
    """Apply rcParams and seaborn defaults for an accessible visual identity."""
    # Apply seaborn theme first to avoid overwriting custom rcParams
    sns.set_theme(style="whitegrid", context="paper", palette=_palette_hex())
    
    _register_fonts()
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["medium_gray"],
            "axes.linewidth": 1.0,
            "axes.labelsize": FONT_SIZES["axis_label"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "axes.labelcolor": COLORS["text_main"],
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": GRID_ALPHA,
            "grid.linewidth": GRID_LINEWIDTH,
            "grid.linestyle": GRID_LINESTYLE,
            "grid.color": COLORS["grid"],
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "xtick.color": COLORS["text_main"],
            "ytick.color": COLORS["text_main"],
            "legend.fontsize": FONT_SIZES["legend"],
            "legend.framealpha": 0.95,
            "legend.edgecolor": COLORS["medium_gray"],
            "font.family": "serif",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{libertine}",
            "text.color": COLORS["text_main"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


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
