"""Utilities for running Hypothesis testing batches.

This package provides helpers to discover configs and run them in parallel
with logging and deterministic output paths for later analysis.
"""

from .runner import (
    list_config_files,
    select_configs,
    run_configs_in_parallel,
)
from .statistics import (
    bias_corrected_cramers_v,
    bootstrap_cramers_v,
    cramers_v,
)
from .visualizations import (
    plot_floor_constraint_distribution,
    plot_income_composition,
    plot_income_preference_bars,
    plot_long_term_counts_grid,
    plot_long_term_margin,
    plot_long_term_stability,
    plot_long_term_stability_grid,
    plot_preference_stability,
    plot_rounds_to_outcome,
    plot_rounds_to_outcome_grouped,
    plot_floor_constraint_distribution_grouped,
    plot_transition_heatmaps,
    plot_voting_attempts_summary,
)
from .style import (
    COLORS,
    FIG_SIZES,
    FONT_SIZES,
    PRINCIPLE_COLORS,
    PRINCIPLE_DISPLAY_NAMES,
    apply_theme,
    format_principle_label,
    format_principle_labels,
)
from .constants import (
    CERTAINTY_TO_SCORE,
    FINAL_WAVE,
    OUTCOME_CATEGORIES,
    PRINCIPLE_LABELS,
    PRINCIPLE_ORDER,
    WAVE_DEFINITIONS,
    WAVE_ORDER,
)
from .data_loading import (
    load_experiment_runs,
    extract_run_metrics,
    extract_vote_rounds,
    extract_rankings,
    extract_income_classes,
    build_transition_data,
    create_transition_matrix,
    GroupDataset,
    build_group_dataset,
)

__all__ = [
    # runner
    "list_config_files",
    "select_configs",
    "run_configs_in_parallel",
    # statistics
    "cramers_v",
    "bias_corrected_cramers_v",
    "bootstrap_cramers_v",
    # visualizations
    "plot_income_preference_bars",
    "plot_income_composition",
    "plot_rounds_to_outcome",
    "plot_rounds_to_outcome_grouped",
    "plot_floor_constraint_distribution_grouped",
    "plot_floor_constraint_distribution",
    "plot_voting_attempts_summary",
    "plot_preference_stability",
    "plot_transition_heatmaps",
    "plot_long_term_stability",
    "plot_long_term_counts_grid",
    "plot_long_term_margin",
    "plot_long_term_stability_grid",
    # style
    "COLORS",
    "FIG_SIZES",
    "FONT_SIZES",
    "PRINCIPLE_COLORS",
    "PRINCIPLE_DISPLAY_NAMES",
    "apply_theme",
    "format_principle_label",
    "format_principle_labels",
    # constants
    "CERTAINTY_TO_SCORE",
    "FINAL_WAVE",
    "OUTCOME_CATEGORIES",
    "PRINCIPLE_LABELS",
    "PRINCIPLE_ORDER",
    "WAVE_DEFINITIONS",
    "WAVE_ORDER",
    # data_loading
    "load_experiment_runs",
    "extract_run_metrics",
    "extract_vote_rounds",
    "extract_rankings",
    "extract_income_classes",
    "build_transition_data",
    "create_transition_matrix",
    "GroupDataset",
    "build_group_dataset",
]

