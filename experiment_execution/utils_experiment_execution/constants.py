"""Shared constants for experiment analysis notebooks.

This module centralizes constants used across hypothesis analysis notebooks,
ensuring consistency in labeling, ordering, and scoring.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Principle Labels (API key -> Display label)
# -----------------------------------------------------------------------------
PRINCIPLE_LABELS: Dict[str | None, str] = {
    "maximizing_average": "Max Avg Income",
    "maximizing_average_floor_constraint": "Max Avg + Floor",
    "maximizing_average_range_constraint": "Max Avg + Range",
    "maximizing_floor": "Max Floor",
    "failure": "Failure",
    None: "None",
}

# Canonical ordering for consistent visualizations
PRINCIPLE_ORDER: List[str] = [
    "Max Floor",
    "Max Avg Income",
    "Max Avg + Floor",
    "Max Avg + Range",
]

# Outcome categories for result classification (API keys)
OUTCOME_CATEGORIES: List[str] = [
    "maximizing_floor",
    "maximizing_average",
    "maximizing_average_floor_constraint",
    "maximizing_average_range_constraint",
    "disagreement",
]


# -----------------------------------------------------------------------------
# Wave Definitions (Phase 1 ranking waves + Final Wave)
# -----------------------------------------------------------------------------
# Each tuple: (json_key, display_label, wave_position)
WAVE_DEFINITIONS: List[Tuple[str, str, int]] = [
    ("initial_ranking", "Wave 1 - Initial", 1),
    ("ranking_2", "Wave 2 - Post-Explanation", 2),
    ("ranking_3", "Wave 3 - Final Phase 1", 3),
]

FINAL_WAVE: Tuple[str, int] = ("Wave 4 - Post-Group", 4)

WAVE_ORDER: List[str] = [label for _, label, _ in WAVE_DEFINITIONS] + [FINAL_WAVE[0]]


# -----------------------------------------------------------------------------
# Certainty Scoring (certainty string -> numeric score)
# -----------------------------------------------------------------------------
CERTAINTY_TO_SCORE: Dict[str | None, float] = {
    "very_unsure": 1,
    "unsure": 1,
    "neutral": 2,
    "sure": 2,
    "very_sure": 3,
    "no_opinion": np.nan,
    None: np.nan,
}


__all__ = [
    "PRINCIPLE_LABELS",
    "PRINCIPLE_ORDER",
    "OUTCOME_CATEGORIES",
    "WAVE_DEFINITIONS",
    "FINAL_WAVE",
    "WAVE_ORDER",
    "CERTAINTY_TO_SCORE",
]
