"""Centralised journal-friendly visual identity helpers for hypothesis notebooks."""

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
JOURNAL_COLORS: ColorMap = {
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
    "Max Avg Income": JOURNAL_COLORS["primary_green"],
    "Max Avg + Floor": JOURNAL_COLORS["primary_blue"],
    "Max Avg + Range": JOURNAL_COLORS["primary_orange"],
    "Max Floor": JOURNAL_COLORS["reddish_purple"], # Distinct from others
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

JOURNAL_FONT_SIZES: FontSizeMap = {
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


JOURNAL_FIG_SIZES: FigureSizeMap = {
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
        JOURNAL_COLORS["primary_green"],
        JOURNAL_COLORS["primary_blue"],
        JOURNAL_COLORS["primary_orange"],
        JOURNAL_COLORS["reddish_purple"],
    ]
    return list(dict.fromkeys(palette_order))


def apply_journal_theme() -> None:
    """Apply rcParams and seaborn defaults for the Journal identity."""
    _register_fonts()
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": JOURNAL_COLORS["background"],
            "axes.facecolor": JOURNAL_COLORS["background"],
            "axes.edgecolor": JOURNAL_COLORS["medium_gray"],
            "axes.linewidth": 1.0,
            "axes.labelsize": JOURNAL_FONT_SIZES["axis_label"],
            "axes.titlesize": JOURNAL_FONT_SIZES["title"],
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "axes.labelcolor": JOURNAL_COLORS["text_main"],
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": GRID_ALPHA,
            "grid.linewidth": GRID_LINEWIDTH,
            "grid.linestyle": GRID_LINESTYLE,
            "grid.color": JOURNAL_COLORS["grid"],
            "xtick.labelsize": JOURNAL_FONT_SIZES["tick_label"],
            "ytick.labelsize": JOURNAL_FONT_SIZES["tick_label"],
            "xtick.color": JOURNAL_COLORS["text_main"],
            "ytick.color": JOURNAL_COLORS["text_main"],
            "legend.fontsize": JOURNAL_FONT_SIZES["legend"],
            "legend.framealpha": 0.95,
            "legend.edgecolor": JOURNAL_COLORS["medium_gray"],
            "font.family": FONT_FAMILY,
            "text.color": JOURNAL_COLORS["text_main"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_theme(style="whitegrid", context="paper", palette=_palette_hex())


def format_principle_label(name: str) -> str:
    """Return display-friendly label for a principle name."""
    return PRINCIPLE_DISPLAY_NAMES.get(name, name)


def format_principle_labels(labels: Iterable[str]) -> List[str]:
    """Vectorised helper to format a sequence of principle names."""
    return [format_principle_label(label) for label in labels]


__all__ = [
    "JOURNAL_COLORS",
    "PRINCIPLE_COLORS",
    "PRINCIPLE_DISPLAY_NAMES",
    "PRINCIPLE_ORDER",
    "JOURNAL_FONT_SIZES",
    "JOURNAL_FIG_SIZES",
    "FONT_FAMILY",
    "apply_journal_theme",
    "format_principle_label",
    "format_principle_labels",
]
