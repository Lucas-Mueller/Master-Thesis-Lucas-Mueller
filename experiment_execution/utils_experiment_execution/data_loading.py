"""Data loading and extraction utilities for experiment analysis.

This module provides functions to load experiment result files and extract
structured data suitable for analysis in pandas DataFrames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import (
    CERTAINTY_TO_SCORE,
    FINAL_WAVE,
    OUTCOME_CATEGORIES,
    PRINCIPLE_LABELS,
    WAVE_DEFINITIONS,
    WAVE_ORDER,
)


# -----------------------------------------------------------------------------
# Result File Loading
# -----------------------------------------------------------------------------


def load_experiment_runs(
    results_dir: Path,
    pattern: str = "*_config_results.json",
    terminal_outputs_dir: Optional[Path] = None,
    log_pattern: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Load experiment result JSON files from a directory.

    Args:
        results_dir: Directory containing result JSON files.
        pattern: Glob pattern to match result files.
        terminal_outputs_dir: Optional directory with terminal logs to cross-reference.
        log_pattern: Optional glob pattern for log files.

    Returns:
        List of (run_id, run_data) tuples, sorted by run_id.
    """
    result_paths = {p.resolve() for p in results_dir.glob(pattern)}

    if terminal_outputs_dir and terminal_outputs_dir.exists() and log_pattern:
        for log_path in terminal_outputs_dir.glob(log_pattern):
            try:
                first_line = log_path.read_text(encoding="utf-8").splitlines()[0]
            except Exception:
                continue
            parts = first_line.strip().split()
            if parts[:1] == ["CMD:"] and parts[-1].endswith(".json"):
                candidate = Path(parts[-1])
                if not candidate.is_absolute():
                    # Try resolving relative to common parents
                    for parent in [results_dir, results_dir.parent, results_dir.parent.parent]:
                        resolved = (parent / candidate).resolve()
                        if resolved.exists():
                            result_paths.add(resolved)
                            break
                elif candidate.exists():
                    result_paths.add(candidate.resolve())

    runs: List[Tuple[str, Dict[str, Any]]] = []
    for file_path in sorted(result_paths):
        if not file_path.exists():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as f:
                runs.append((file_path.stem, json.load(f)))
        except (json.JSONDecodeError, IOError):
            continue

    runs.sort(key=lambda item: item[0])
    return runs


# -----------------------------------------------------------------------------
# Result Classification
# -----------------------------------------------------------------------------


def categorize_result(result_path: Path) -> str:
    """Categorize an experiment result file by its outcome.

    Args:
        result_path: Path to a *_results.json file.

    Returns:
        One of OUTCOME_CATEGORIES: the agreed principle API key or 'disagreement'.
    """
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        gi = data.get("general_information", {})
        consensus = gi.get("consensus_reached", False)
        principle = gi.get("consensus_principle")
        if consensus and principle in OUTCOME_CATEGORIES:
            return principle
        return "disagreement"
    except Exception:
        return "disagreement"


def fisher_freeman_halton_pvalue_r(
    contingency,
    timeout_sec: int = 30,
) -> Optional[float]:
    """Run Fisher-Freeman-Halton exact test via R's fisher.test.

    Args:
        contingency: 2D numpy array or list of lists contingency table.
        timeout_sec: Timeout in seconds for the R subprocess.

    Returns:
        p-value as float, or None if R is not available or test fails.
    """
    import shutil
    import subprocess

    import numpy as np

    if shutil.which("Rscript") is None:
        return None

    contingency = np.asarray(contingency)
    r_matrix = ",".join(str(int(x)) for x in contingency.flatten(order="C"))
    nrow, ncol = contingency.shape

    r_code = f"""m <- matrix(c({r_matrix}), nrow={nrow}, ncol={ncol}, byrow=TRUE);
f <- tryCatch(fisher.test(m), error=function(e) NA);
if (is.list(f)) {{ cat(f$p.value) }} else {{ cat('NA') }}
"""
    try:
        out = subprocess.check_output(
            ["Rscript", "-e", r_code],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
        out = out.strip()
        return float(out) if out and out != "NA" else None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Data Extraction Functions
# -----------------------------------------------------------------------------


def extract_run_metrics(run_id: str, run_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract high-level run metrics from experiment data.

    Returns a dict with keys: run_id, consensus_reached, consensus_principle,
    rounds_to_outcome, max_rounds, total_vote_attempts, successful_votes,
    total_vote_rounds.
    """
    general = run_data.get("general_information", {})
    voting = run_data.get("voting_history", {})
    return {
        "run_id": run_id,
        "consensus_reached": general.get("consensus_reached"),
        "consensus_principle": PRINCIPLE_LABELS.get(
            general.get("consensus_principle"), general.get("consensus_principle")
        ),
        "rounds_to_outcome": general.get("rounds_conducted_phase_2"),
        "max_rounds": general.get("max_rounds_phase_2"),
        "total_vote_attempts": voting.get("total_vote_attempts"),
        "successful_votes": voting.get("successful_votes"),
        "total_vote_rounds": len(voting.get("vote_rounds", [])),
    }


def extract_vote_rounds(run_id: str, run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract vote round details from experiment data.

    Returns a list of dicts, one per voting round.
    """
    rounds: List[Dict[str, Any]] = []
    for idx, round_info in enumerate(
        run_data.get("voting_history", {}).get("vote_rounds", []), start=1
    ):
        rounds.append({
            "run_id": run_id,
            "round_index": idx,
            "round_number": round_info.get("round_number"),
            "vote_type": round_info.get("vote_type"),
            "consensus_reached": round_info.get("consensus_reached", False),
            "agreed_principle": round_info.get("agreed_principle"),
            "agreed_principle_label": PRINCIPLE_LABELS.get(
                round_info.get("agreed_principle"), round_info.get("agreed_principle")
            ),
            "agreed_constraint": round_info.get("agreed_constraint"),
            "participant_count": len(round_info.get("participant_votes", [])),
        })
    return rounds


def _append_ranking_rows(
    *,
    target_top: List[Dict[str, Any]],
    target_long: List[Dict[str, Any]],
    run_id: str,
    agent_name: str,
    wave_label: str,
    wave_pos: int,
    ranking_data: Dict[str, Any],
) -> None:
    """Helper to append ranking data to collection lists."""
    rankings = ranking_data.get("rankings", [])
    if not rankings:
        return

    certainty = ranking_data.get("certainty")
    certainty_score = CERTAINTY_TO_SCORE.get(certainty, np.nan)
    top_principle = None

    for item in rankings:
        principle_name = item.get("principle")
        rank_value = item.get("rank")
        target_long.append({
            "run_id": run_id,
            "agent": agent_name,
            "wave_label": wave_label,
            "wave_position": wave_pos,
            "principle": principle_name,
            "principle_label": PRINCIPLE_LABELS.get(principle_name, principle_name),
            "rank": rank_value,
            "certainty": certainty,
            "certainty_score": certainty_score,
        })
        if rank_value == 1 and top_principle is None:
            top_principle = principle_name

    if top_principle is not None:
        target_top.append({
            "run_id": run_id,
            "agent": agent_name,
            "wave_label": wave_label,
            "wave_position": wave_pos,
            "top_principle": top_principle,
            "top_principle_label": PRINCIPLE_LABELS.get(top_principle, top_principle),
            "certainty": certainty,
            "certainty_score": certainty_score,
        })


def extract_rankings(
    run_id: str, run_data: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract agent ranking data from experiment results.

    Returns:
        Tuple of (top_choice_records, long_format_records).
    """
    top_choice_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    for agent in run_data.get("agents", []):
        agent_name = agent.get("name")
        phase1 = agent.get("phase_1", {})

        for key, wave_label, wave_pos in WAVE_DEFINITIONS:
            container = phase1.get(key)
            ranking_data = container.get("ranking_result") if isinstance(container, dict) else None
            if ranking_data:
                _append_ranking_rows(
                    target_top=top_choice_rows,
                    target_long=long_rows,
                    run_id=run_id,
                    agent_name=agent_name,
                    wave_label=wave_label,
                    wave_pos=wave_pos,
                    ranking_data=ranking_data,
                )

        # Final wave (post-group discussion)
        final_ranking = (
            agent.get("phase_2", {})
            .get("post_group_discussion", {})
            .get("final_ranking")
        )
        if final_ranking:
            _append_ranking_rows(
                target_top=top_choice_rows,
                target_long=long_rows,
                run_id=run_id,
                agent_name=agent_name,
                wave_label=FINAL_WAVE[0],
                wave_pos=FINAL_WAVE[1],
                ranking_data=final_ranking,
            )

    return top_choice_rows, long_rows


def extract_income_classes(
    runs: List[Tuple[str, Dict[str, Any]]]
) -> pd.DataFrame:
    """Extract income class assignments from Phase 2 data.

    Returns a DataFrame with columns: run_id, agent, income_class_raw, income_class.
    """
    income_data = []

    for run_id, run_data in runs:
        for agent in run_data.get("agents", []):
            agent_name = agent.get("name")
            phase2 = agent.get("phase_2", {})
            post_group = phase2.get("post_group_discussion", {})
            class_assigned = post_group.get("class_put_in", None)

            income_data.append({
                "run_id": run_id,
                "agent": agent_name,
                "income_class_raw": class_assigned,
            })

    df = pd.DataFrame(income_data)

    # Standardize labels
    class_map = {
        "low": "Low",
        "medium_low": "Medium-Low",
        "medium": "Medium",
        "medium_high": "Medium-High",
        "high": "High",
    }

    def standardize(raw):
        if raw is None or pd.isna(raw):
            return "Unknown"
        return class_map.get(str(raw).lower(), "Unknown")

    df["income_class"] = df["income_class_raw"].apply(standardize)
    return df


# -----------------------------------------------------------------------------
# Transition Data Builders
# -----------------------------------------------------------------------------


def build_transition_data(ranking_top_df: pd.DataFrame) -> pd.DataFrame:
    """Build transition data showing how agents' top choices change between waves.

    Args:
        ranking_top_df: DataFrame with columns run_id, agent, wave_position, top_principle_label.

    Returns:
        DataFrame with columns: run_id, agent, wave1, wave2, wave3, wave4.
    """
    agent_sessions = ranking_top_df[["run_id", "agent"]].drop_duplicates()
    transitions = []

    for _, row in agent_sessions.iterrows():
        run_id, agent = row["run_id"], row["agent"]
        agent_data = ranking_top_df[
            (ranking_top_df["run_id"] == run_id) & (ranking_top_df["agent"] == agent)
        ].sort_values("wave_position")

        if len(agent_data) >= 4:
            prefs = agent_data["top_principle_label"].values
            transitions.append({
                "run_id": run_id,
                "agent": agent,
                "wave1": prefs[0],
                "wave2": prefs[1],
                "wave3": prefs[2],
                "wave4": prefs[3],
            })

    return pd.DataFrame(transitions)


def create_transition_matrix(
    transition_data: pd.DataFrame, from_col: str, to_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create transition matrix showing flows between principles.

    Args:
        transition_data: DataFrame with wave columns.
        from_col: Source wave column name (e.g., "wave1").
        to_col: Target wave column name (e.g., "wave2").

    Returns:
        Tuple of (counts_matrix, percentage_matrix).
    """
    matrix = pd.crosstab(transition_data[from_col], transition_data[to_col], margins=False)
    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0) * 100
    return matrix, matrix_pct


# -----------------------------------------------------------------------------
# GroupDataset: Container for Multi-Group Analysis
# -----------------------------------------------------------------------------


@dataclass
class GroupDataset:
    """Container for a single experiment group's data.

    Attributes:
        label: Human-readable group name (e.g., "English").
        slug: Filesystem-safe identifier (e.g., "english").
        runs: Raw (run_id, run_data) tuples.
        run_metrics: DataFrame of run-level metrics.
        vote_rounds: DataFrame of voting round details.
        ranking_top: DataFrame of top-choice rankings per wave.
        ranking_long: DataFrame of full rankings in long format.
        transition_data: Optional DataFrame of wave transitions.
        income_df: Optional DataFrame of income class assignments.
        switcher_analysis: Optional DataFrame with switcher analysis.
    """
    label: str
    slug: str
    runs: List[Tuple[str, Dict[str, Any]]]
    run_metrics: pd.DataFrame
    vote_rounds: pd.DataFrame
    ranking_top: pd.DataFrame
    ranking_long: pd.DataFrame
    transition_data: Optional[pd.DataFrame] = None
    income_df: Optional[pd.DataFrame] = None
    switcher_analysis: Optional[pd.DataFrame] = None


def build_group_dataset(
    label: str,
    slug: str,
    runs: List[Tuple[str, Dict[str, Any]]],
) -> GroupDataset:
    """Build a complete GroupDataset from raw experiment runs.

    Args:
        label: Human-readable group name.
        slug: Filesystem-safe identifier.
        runs: List of (run_id, run_data) tuples.

    Returns:
        Populated GroupDataset instance.
    """
    run_metrics_records: List[Dict[str, Any]] = []
    vote_round_records: List[Dict[str, Any]] = []
    top_choice_records: List[Dict[str, Any]] = []
    long_ranking_records: List[Dict[str, Any]] = []

    for run_id, run_data in runs:
        run_metrics_records.append(extract_run_metrics(run_id, run_data))
        vote_round_records.extend(extract_vote_rounds(run_id, run_data))
        top_rows, long_rows = extract_rankings(run_id, run_data)
        top_choice_records.extend(top_rows)
        long_ranking_records.extend(long_rows)

    run_metrics = pd.DataFrame(run_metrics_records) if run_metrics_records else pd.DataFrame()
    vote_rounds_df = pd.DataFrame(vote_round_records) if vote_round_records else pd.DataFrame()
    ranking_top_df = pd.DataFrame(top_choice_records) if top_choice_records else pd.DataFrame()
    ranking_long_df = pd.DataFrame(long_ranking_records) if long_ranking_records else pd.DataFrame()

    # Add group label
    for df in [run_metrics, vote_rounds_df, ranking_top_df, ranking_long_df]:
        if not df.empty:
            df["group"] = label

    # Set categorical wave ordering
    if not ranking_top_df.empty and "wave_label" in ranking_top_df.columns:
        ranking_top_df["wave_label"] = pd.Categorical(
            ranking_top_df["wave_label"], categories=WAVE_ORDER, ordered=True
        )
    if not ranking_long_df.empty and "wave_label" in ranking_long_df.columns:
        ranking_long_df["wave_label"] = pd.Categorical(
            ranking_long_df["wave_label"], categories=WAVE_ORDER, ordered=True
        )

    return GroupDataset(
        label=label,
        slug=slug,
        runs=runs,
        run_metrics=run_metrics,
        vote_rounds=vote_rounds_df,
        ranking_top=ranking_top_df,
        ranking_long=ranking_long_df,
    )


# -----------------------------------------------------------------------------
# Preference Analysis Helpers
# -----------------------------------------------------------------------------


def create_preference_ordering_table(
    ranking_long_df: pd.DataFrame, wave_name: str
) -> pd.DataFrame:
    """Create a frequency table of complete preference orderings for a specific wave.

    Args:
        ranking_long_df: DataFrame with all ranking data (must have columns:
            wave_label, run_id, agent, principle_label, rank).
        wave_name: Name of the wave to analyze (e.g., "Wave 1 - Initial").

    Returns:
        DataFrame showing each unique ordering and its frequency, with columns:
        Preference Ordering, Count, Percentage.
    """
    # Filter to the specific wave
    wave_data = ranking_long_df[ranking_long_df["wave_label"] == wave_name].copy()

    if wave_data.empty:
        return pd.DataFrame({"Note": [f"No data available for {wave_name}"]})

    # Create ordering strings for each agent
    orderings = []
    for (run_id, agent), group in wave_data.groupby(["run_id", "agent"]):
        # Sort by rank to get the ordering
        sorted_group = group.sort_values("rank")
        ordering = tuple(sorted_group["principle_label"].tolist())
        orderings.append({
            "run_id": run_id,
            "agent": agent,
            "ordering": ordering,
        })

    ordering_df = pd.DataFrame(orderings)

    # Count frequency of each ordering
    freq_table = ordering_df.groupby("ordering").size().reset_index(name="Count")
    freq_table = freq_table.sort_values("Count", ascending=False).reset_index(drop=True)

    # Calculate percentage
    total = freq_table["Count"].sum()
    freq_table["Percentage"] = (freq_table["Count"] / total * 100).round(1)

    # Format the ordering as a readable string (1st > 2nd > 3rd > 4th)
    def format_ordering(ordering_tuple):
        if len(ordering_tuple) == 4:
            return f"{ordering_tuple[0]} > {ordering_tuple[1]} > {ordering_tuple[2]} > {ordering_tuple[3]}"
        return str(ordering_tuple)

    freq_table["Preference Ordering"] = freq_table["ordering"].apply(format_ordering)

    # Reorder columns for display
    freq_table = freq_table[["Preference Ordering", "Count", "Percentage"]]

    return freq_table


def prepare_switcher_analysis(
    transition_df: pd.DataFrame, income_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge transition and income data, categorize agents by switching behavior.

    Args:
        transition_df: DataFrame with columns: run_id, agent, wave1, wave2, wave3, wave4.
        income_df: DataFrame with columns: run_id, agent, income_class_raw, income_class.

    Returns:
        DataFrame with switcher analysis columns including:
        - switched_w1_w2, switched_w2_w3, switched_w3_w4: Boolean switches per transition
        - switched_any: Boolean indicating if agent switched at any point
        - switched_w1_w4: Boolean indicating if wave1 differs from wave4
        - num_switches: Count of total switches (0-3)
        - switcher_category: Category label based on switching frequency
    """
    # Merge income data with transition data
    switcher_analysis = transition_df.merge(income_df, on=["run_id", "agent"], how="left")

    # Identify switchers: agents who changed their top choice at ANY point
    switcher_analysis["switched_w1_w2"] = (
        switcher_analysis["wave1"] != switcher_analysis["wave2"]
    )
    switcher_analysis["switched_w2_w3"] = (
        switcher_analysis["wave2"] != switcher_analysis["wave3"]
    )
    switcher_analysis["switched_w3_w4"] = (
        switcher_analysis["wave3"] != switcher_analysis["wave4"]
    )
    switcher_analysis["switched_any"] = (
        switcher_analysis["switched_w1_w2"]
        | switcher_analysis["switched_w2_w3"]
        | switcher_analysis["switched_w3_w4"]
    )
    switcher_analysis["switched_w1_w4"] = (
        switcher_analysis["wave1"] != switcher_analysis["wave4"]
    )

    # Count switches per agent
    switcher_analysis["num_switches"] = (
        switcher_analysis["switched_w1_w2"].astype(int)
        + switcher_analysis["switched_w2_w3"].astype(int)
        + switcher_analysis["switched_w3_w4"].astype(int)
    )

    # Categorize agents
    def categorize_switcher(row):
        if row["num_switches"] == 0:
            return "Loyal (No switches)"
        elif row["num_switches"] == 1:
            return "Minor switcher (1 switch)"
        elif row["num_switches"] == 2:
            return "Moderate switcher (2 switches)"
        else:
            return "Major switcher (3+ switches)"

    switcher_analysis["switcher_category"] = switcher_analysis.apply(
        categorize_switcher, axis=1
    )

    return switcher_analysis


def summarize_income_preferences(
    switcher_analysis: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create summary tables comparing income distribution by preference stability.

    Args:
        switcher_analysis: DataFrame from prepare_switcher_analysis with columns:
            income_class, switched_any, and other switcher analysis columns.

    Returns:
        Tuple of (summary_df, count_long, share_long, composition_percent_long):
        - summary_df: Wide-format comparison table with counts and percentages
        - count_long: Long-format counts for plotting
        - share_long: Long-format within-group shares for plotting
        - composition_percent_long: Long-format within-income-class shares for plotting
    """
    # Create comparison table
    switchers = switcher_analysis[switcher_analysis["switched_any"] == True]
    maintainers = switcher_analysis[switcher_analysis["switched_any"] == False]

    income_comparison = pd.DataFrame({
        "Income Class": ["Low", "Medium-Low", "Medium", "Medium-High", "High"],
    })

    # Calculate counts and percentages for agents who changed preference
    switcher_income_counts = switchers["income_class"].value_counts()
    income_comparison["Changed Preference Count"] = (
        income_comparison["Income Class"]
        .map(switcher_income_counts)
        .fillna(0)
        .astype(int)
    )
    income_comparison["Changed Preference %"] = (
        income_comparison["Changed Preference Count"] / len(switchers) * 100
    ).round(1) if len(switchers) > 0 else 0.0

    # Calculate counts and percentages for agents who maintained preference
    maintainer_income_counts = maintainers["income_class"].value_counts()
    income_comparison["Maintained Preference Count"] = (
        income_comparison["Income Class"]
        .map(maintainer_income_counts)
        .fillna(0)
        .astype(int)
    )
    income_comparison["Maintained Preference %"] = (
        income_comparison["Maintained Preference Count"] / len(maintainers) * 100
    ).round(1) if len(maintainers) > 0 else 0.0

    # Calculate difference between groups (percentage points)
    income_comparison["Diff Changed vs Maintained (pp)"] = (
        income_comparison["Changed Preference %"]
        - income_comparison["Maintained Preference %"]
    ).round(1)

    # Create long-format data for plotting
    count_columns = {
        "Changed Preference Count": "Changed Preference",
        "Maintained Preference Count": "Maintained Preference",
    }
    share_columns = {
        "Changed Preference %": "Changed Preference",
        "Maintained Preference %": "Maintained Preference",
    }

    count_wide = income_comparison[["Income Class", *count_columns.keys()]].rename(
        columns=count_columns
    )
    share_wide = income_comparison[["Income Class", *share_columns.keys()]].rename(
        columns=share_columns
    )

    count_long = count_wide.melt(
        id_vars="Income Class",
        value_vars=list(count_columns.values()),
        var_name="Preference Group",
        value_name="Count",
    )

    share_long = share_wide.melt(
        id_vars="Income Class",
        value_vars=list(share_columns.values()),
        var_name="Preference Group",
        value_name="Share",
    )

    # Create within-income-class composition data
    composition_summary = income_comparison.copy()
    composition_summary["Total Agents"] = (
        composition_summary["Changed Preference Count"]
        + composition_summary["Maintained Preference Count"]
    )

    composition_long = composition_summary.melt(
        id_vars=["Income Class", "Total Agents"],
        value_vars=["Changed Preference Count", "Maintained Preference Count"],
        var_name="Preference Group",
        value_name="Count",
    )

    composition_long["Preference Group"] = composition_long["Preference Group"].replace({
        "Changed Preference Count": "Changed Preference",
        "Maintained Preference Count": "Maintained Preference",
    })
    composition_long["Percent"] = np.where(
        composition_long["Total Agents"] > 0,
        (composition_long["Count"] / composition_long["Total Agents"]) * 100,
        0,
    )

    composition_percent_long = composition_long[
        ["Income Class", "Preference Group", "Percent"]
    ].copy()

    return income_comparison, count_long, share_long, composition_percent_long


__all__ = [
    "load_experiment_runs",
    "categorize_result",
    "fisher_freeman_halton_pvalue_r",
    "extract_run_metrics",
    "extract_vote_rounds",
    "extract_rankings",
    "extract_income_classes",
    "build_transition_data",
    "create_transition_matrix",
    "GroupDataset",
    "build_group_dataset",
    "create_preference_ordering_table",
    "prepare_switcher_analysis",
    "summarize_income_preferences",
]
