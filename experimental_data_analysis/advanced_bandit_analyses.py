#!/usr/bin/env python3
"""Advanced summary analyses for bandit pilot data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_visualize_bandit_data_module():
    module_path = Path(__file__).with_name("visualize_bandit_data.py")
    spec = importlib.util.spec_from_file_location("visualize_bandit_data", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load visualize_bandit_data from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_viz = _load_visualize_bandit_data_module()


def _sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _coerce_reduce_name(reduce_name: str):
    allowed = {"mean", "median"}
    if reduce_name not in allowed:
        raise ValueError(f"Unsupported reduce function: {reduce_name}")
    return reduce_name


def _performance_group_order() -> Sequence[str]:
    return ("low", "middle", "high")


def _performance_group_colors() -> Dict[str, str]:
    return {
        "low": "tab:blue",
        "middle": "tab:orange",
        "high": "tab:green",
    }


def add_previous_reward_columns(
    df: pd.DataFrame,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    block_col: str = "block",
    trial_col: str = "trial",
    reward_col: str = "normalized_score",
    bin_edges: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Add previous-trial normalized reward columns and fixed reward bins."""
    if reward_col not in df.columns:
        raise ValueError(f"Reward column not found: {reward_col}")

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    work = df.copy()

    for col in [env_col, block_col, trial_col, reward_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.sort_values([pid_col, block_col, env_col, trial_col], kind="mergesort").copy()
    work["prev_normalized_score"] = (
        work.groupby([pid_col, block_col, env_col], dropna=False)[reward_col].shift(1)
    )

    edges = list(bin_edges or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if edges[-1] <= 1.0:
        edges[-1] = 1.0000001
    labels = [f"{edges[index]:.1f}–{min(edges[index + 1], 1.0):.1f}" for index in range(len(edges) - 1)]
    work["prev_reward_bin"] = pd.cut(
        work["prev_normalized_score"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    )
    work["prev_reward_bin_index"] = work["prev_reward_bin"].cat.codes.replace(-1, np.nan)
    return work


def _summarize_previous_reward_conditioned_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    block_col: str = "block",
    bin_col: str = "prev_reward_bin",
    bin_index_col: str = "prev_reward_bin_index",
    within_series_reduce: str = "mean",
    line_center_reduce: str = "mean",
    overall_center_reduce: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[str]]:
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")
    if bin_col not in df.columns or bin_index_col not in df.columns:
        raise ValueError("Previous reward columns are missing. Run add_previous_reward_columns(...) first.")

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    work = df[[pid_col, env_col, block_col, value_col, bin_col, bin_index_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[env_col] = pd.to_numeric(work[env_col], errors="coerce")
    work[block_col] = pd.to_numeric(work[block_col], errors="coerce")
    work = work.dropna(subset=[value_col, env_col, block_col, bin_col, bin_index_col])
    work["within_unit"] = work[env_col].astype(int).astype(str) + "_b" + work[block_col].astype(int).astype(str)

    reduced = (
        work.groupby([pid_col, bin_col, bin_index_col, "within_unit"], dropna=False, observed=True)[value_col]
        .agg(_coerce_reduce_name(within_series_reduce))
        .reset_index(name="value")
    )

    unit_summary = (
        reduced.groupby([pid_col, bin_col, bin_index_col], dropna=False, observed=True)["value"]
        .agg(mean=_coerce_reduce_name(line_center_reduce), sem=_sem, n="count")
        .reset_index()
        .sort_values([pid_col, bin_index_col], kind="mergesort")
    )
    overall_summary = (
        unit_summary.groupby([bin_col, bin_index_col], dropna=False, observed=True)["mean"]
        .agg(mean=_coerce_reduce_name(overall_center_reduce), sem=_sem, n="count")
        .reset_index()
        .sort_values(bin_index_col, kind="mergesort")
    )
    labels = [
        str(label)
        for label in unit_summary.sort_values(bin_index_col, kind="mergesort")[bin_col].dropna().drop_duplicates().tolist()
    ]
    return unit_summary, overall_summary, labels


def plot_previous_reward_conditioned_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    block_col: str = "block",
    within_series_reduce: str = "mean",
    line_center_reduce: str = "mean",
    overall_center_reduce: str = "mean",
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    show: bool = False,
):
    """Plot participant-level and grand-average trends conditioned on previous reward."""
    unit_summary, overall_summary, labels = _summarize_previous_reward_conditioned_trends(
        df,
        value_col,
        participant_id_col=participant_id_col,
        env_col=env_col,
        block_col=block_col,
        within_series_reduce=within_series_reduce,
        line_center_reduce=line_center_reduce,
        overall_center_reduce=overall_center_reduce,
    )

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=False)
    positions = np.arange(len(labels))

    for _, sub in unit_summary.groupby(pid_col, sort=True):
        sub = sub.sort_values("prev_reward_bin_index")
        x = sub["prev_reward_bin_index"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        sem = sub["sem"].to_numpy(dtype=float)
        axes[0].plot(
            x,
            y,
            linewidth=1.4,
            alpha=0.35,
            color="tab:blue",
            marker="o",
            markersize=3.5,
        )
        axes[0].fill_between(x, y - sem, y + sem, alpha=0.08, color="tab:blue")

    overall_summary = overall_summary.sort_values("prev_reward_bin_index")
    x = overall_summary["prev_reward_bin_index"].to_numpy(dtype=float)
    y = overall_summary["mean"].to_numpy(dtype=float)
    sem = overall_summary["sem"].to_numpy(dtype=float)
    axes[1].plot(
        x,
        y,
        linewidth=2.2,
        alpha=0.95,
        color="tab:purple",
        marker="o",
        markersize=5,
    )
    axes[1].fill_between(
        x,
        y - sem,
        y + sem,
        alpha=0.18,
        color="tab:purple",
    )

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Previous normalized reward")
        ax.set_xlim(-0.25, len(labels) - 0.75)

    axes[0].set_title("Participant means across envs and blocks")
    axes[0].set_ylabel(y_label or value_col.replace("_", " "))
    axes[1].set_title("Grand mean across participants")
    fig.suptitle(title or f"{value_col} by previous reward")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _summarize_previous_reward_conditioned_by_group(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    block_col: str = "block",
    group_col: str = "performance_group",
    bin_col: str = "prev_reward_bin",
    bin_index_col: str = "prev_reward_bin_index",
    within_series_reduce: str = "mean",
    line_center_reduce: str = "mean",
    overall_center_reduce: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence[str]]:
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")
    if group_col not in df.columns:
        raise ValueError("Performance groups are missing. Run assign_performance_groups(...) first.")
    if bin_col not in df.columns or bin_index_col not in df.columns:
        raise ValueError("Previous reward columns are missing. Run add_previous_reward_columns(...) first.")

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    work = df[[pid_col, env_col, block_col, value_col, group_col, bin_col, bin_index_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[env_col] = pd.to_numeric(work[env_col], errors="coerce")
    work[block_col] = pd.to_numeric(work[block_col], errors="coerce")
    work = work.dropna(subset=[pid_col, env_col, block_col, value_col, group_col, bin_col, bin_index_col])
    work["within_unit"] = work[env_col].astype(int).astype(str) + "_b" + work[block_col].astype(int).astype(str)

    reduced = (
        work.groupby([pid_col, group_col, bin_col, bin_index_col, "within_unit"], dropna=False, observed=True)[value_col]
        .agg(_coerce_reduce_name(within_series_reduce))
        .reset_index(name="value")
    )

    participant_summary = (
        reduced.groupby([pid_col, group_col, bin_col, bin_index_col], dropna=False, observed=True)["value"]
        .agg(mean=_coerce_reduce_name(line_center_reduce), sem=_sem, n="count")
        .reset_index()
        .sort_values([group_col, pid_col, bin_index_col], kind="mergesort")
    )
    overall_summary = (
        participant_summary.groupby([group_col, bin_col, bin_index_col], dropna=False, observed=True)["mean"]
        .agg(mean=_coerce_reduce_name(overall_center_reduce), sem=_sem, n="count")
        .reset_index()
        .sort_values([group_col, bin_index_col], kind="mergesort")
    )
    labels = [
        str(label)
        for label in participant_summary.sort_values(bin_index_col, kind="mergesort")[bin_col].dropna().drop_duplicates().tolist()
    ]
    return participant_summary, overall_summary, labels


def plot_previous_reward_conditioned_by_performance(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    block_col: str = "block",
    group_col: str = "performance_group",
    within_series_reduce: str = "mean",
    line_center_reduce: str = "mean",
    overall_center_reduce: str = "mean",
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    show: bool = False,
):
    """Plot previous-reward-conditioned trends separated by performance group."""
    participant_summary, overall_summary, labels = _summarize_previous_reward_conditioned_by_group(
        df,
        value_col,
        participant_id_col=participant_id_col,
        env_col=env_col,
        block_col=block_col,
        group_col=group_col,
        within_series_reduce=within_series_reduce,
        line_center_reduce=line_center_reduce,
        overall_center_reduce=overall_center_reduce,
    )

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    colors = _performance_group_colors()
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=False)

    for group_name in _performance_group_order():
        group_participants = participant_summary[participant_summary[group_col] == group_name]
        color = colors[group_name]
        for _, sub in group_participants.groupby(pid_col, sort=True):
            sub = sub.sort_values("prev_reward_bin_index")
            x = sub["prev_reward_bin_index"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            sem = sub["sem"].to_numpy(dtype=float)
            axes[0].plot(x, y, linewidth=1.3, alpha=0.28, color=color, marker="o", markersize=3.0)
            axes[0].fill_between(x, y - sem, y + sem, alpha=0.06, color=color)

        group_overall = overall_summary[overall_summary[group_col] == group_name].sort_values("prev_reward_bin_index")
        x = group_overall["prev_reward_bin_index"].to_numpy(dtype=float)
        y = group_overall["mean"].to_numpy(dtype=float)
        sem = group_overall["sem"].to_numpy(dtype=float)
        axes[1].plot(x, y, linewidth=2.2, alpha=0.95, color=color, marker="o", markersize=5, label=group_name)
        axes[1].fill_between(x, y - sem, y + sem, alpha=0.14, color=color)

    axes[0].set_title("Participant means within performance groups")
    axes[1].set_title("Grand means by performance group")
    axes[1].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Previous normalized reward")
        ax.set_xlim(-0.25, len(labels) - 0.75)

    axes[0].set_ylabel(y_label or value_col.replace("_", " "))
    fig.suptitle(title or f"{value_col} by previous reward and performance group")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def assign_performance_groups(
    df: pd.DataFrame,
    *,
    participant_id_col: Optional[str] = None,
    performance_col: str = "reward",
) -> pd.DataFrame:
    """Assign low/middle/high participant performance groups from overall reward."""
    if performance_col not in df.columns:
        raise ValueError(f"Performance column not found: {performance_col}")

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    participant_reward = (
        df[[pid_col, performance_col]]
        .copy()
        .assign(**{performance_col: lambda frame: pd.to_numeric(frame[performance_col], errors="coerce")})
        .dropna(subset=[performance_col])
        .groupby(pid_col, dropna=False)[performance_col]
        .first()
        .reset_index()
    )

    if participant_reward.empty:
        raise ValueError("No participant performance values available.")

    ranked = participant_reward[performance_col].rank(method="first")
    participant_reward["performance_group"] = pd.qcut(
        ranked,
        q=3,
        labels=list(_performance_group_order()),
    )

    merged = df.merge(participant_reward[[pid_col, "performance_group"]], on=pid_col, how="left")
    merged["performance_group"] = pd.Categorical(
        merged["performance_group"],
        categories=list(_performance_group_order()),
        ordered=True,
    )
    return merged


def _summarize_performance_group_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    x_col: str,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    group_col: str = "performance_group",
    within_series_reduce: str = "mean",
    line_center_reduce: str = "mean",
    overall_center_reduce: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")
    if group_col not in df.columns:
        raise ValueError("Performance groups are missing. Run assign_performance_groups(...) first.")

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    work = df[[pid_col, env_col, x_col, value_col, group_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[env_col] = pd.to_numeric(work[env_col], errors="coerce")
    work = work.dropna(subset=[pid_col, env_col, x_col, value_col, group_col])

    reduced = (
        work.groupby([pid_col, group_col, x_col, env_col], dropna=False, observed=False)[value_col]
        .agg(_coerce_reduce_name(within_series_reduce))
        .reset_index(name="value")
    )

    participant_summary = (
        reduced.groupby([pid_col, group_col, x_col], dropna=False, observed=False)["value"]
        .agg(mean=_coerce_reduce_name(line_center_reduce), sem=_sem, n="count")
        .reset_index()
        .sort_values([group_col, pid_col, x_col], kind="mergesort")
    )
    overall_summary = (
        participant_summary.groupby([group_col, x_col], dropna=False, observed=False)["mean"]
        .agg(mean=_coerce_reduce_name(overall_center_reduce), sem=_sem, n="count")
        .reset_index()
        .sort_values([group_col, x_col], kind="mergesort")
    )
    return participant_summary, overall_summary


def plot_performance_group_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    x_col: str = "trial",
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    group_col: str = "performance_group",
    within_series_reduce: str = "mean",
    line_center_reduce: str = "mean",
    overall_center_reduce: str = "mean",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    show: bool = False,
):
    """Plot participant-level lines and grouped grand averages by performance tercile."""
    participant_summary, overall_summary = _summarize_performance_group_trends(
        df,
        value_col,
        x_col=x_col,
        participant_id_col=participant_id_col,
        env_col=env_col,
        group_col=group_col,
        within_series_reduce=within_series_reduce,
        line_center_reduce=line_center_reduce,
        overall_center_reduce=overall_center_reduce,
    )

    pid_col = _viz.choose_participant_id_col(df, participant_id_col)
    colors = _performance_group_colors()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharex=True, sharey=False)

    for group_name in _performance_group_order():
        group_participants = participant_summary[participant_summary[group_col] == group_name]
        color = colors[group_name]
        for _, sub in group_participants.groupby(pid_col, sort=True):
            axes[0].plot(sub[x_col], sub["mean"], linewidth=1.0, alpha=0.55, color=color)
            axes[0].fill_between(sub[x_col], sub["mean"] - sub["sem"], sub["mean"] + sub["sem"], alpha=0.08, color=color)

        group_overall = overall_summary[overall_summary[group_col] == group_name]
        axes[1].plot(group_overall[x_col], group_overall["mean"], linewidth=1.5, alpha=0.95, color=color, label=group_name)
        axes[1].fill_between(
            group_overall[x_col],
            group_overall["mean"] - group_overall["sem"],
            group_overall["mean"] + group_overall["sem"],
            alpha=0.16,
            color=color,
        )

    axes[0].set_title("Participant means within performance groups")
    axes[1].set_title("Grand means by performance group")
    axes[1].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.set_xlabel(x_label or x_col.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(y_label or value_col.replace("_", " "))
    fig.suptitle(title or f"{value_col} by {x_col} and performance group")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def make_performance_group_visualizations(
    df: pd.DataFrame,
    *,
    participant_id_col: Optional[str] = None,
    rt_col: Optional[str] = None,
    group_col: str = "performance_group",
    show: bool = False,
) -> Dict[str, Dict[str, Tuple[plt.Figure, np.ndarray]]]:
    """Create trial- and block-level performance-group plots for common metrics."""
    figures: Dict[str, Dict[str, Tuple[plt.Figure, np.ndarray]]] = {"trial": {}, "block": {}}

    reward_col = "normalized_score" if "normalized_score" in df.columns else "score" if "score" in df.columns else None
    rt_col = rt_col or _viz.find_rt_column(df)

    if reward_col is not None:
        figures["trial"]["reward"] = plot_performance_group_trends(
            df,
            reward_col,
            x_col="trial",
            participant_id_col=participant_id_col,
            group_col=group_col,
            within_series_reduce="mean",
            line_center_reduce="mean",
            overall_center_reduce="mean",
            title="Reward by trial and performance group",
            x_label="Trial",
            y_label="Mean reward",
            show=False,
        )
        figures["block"]["reward"] = plot_performance_group_trends(
            df,
            reward_col,
            x_col="block",
            participant_id_col=participant_id_col,
            group_col=group_col,
            within_series_reduce="mean",
            line_center_reduce="mean",
            overall_center_reduce="mean",
            title="Reward by block and performance group",
            x_label="Block",
            y_label="Mean reward",
            show=False,
        )

    if rt_col is not None and rt_col in df.columns:
        figures["trial"]["rt"] = plot_performance_group_trends(
            df,
            rt_col,
            x_col="trial",
            participant_id_col=participant_id_col,
            group_col=group_col,
            within_series_reduce="mean",
            line_center_reduce="median",
            overall_center_reduce="mean",
            title="RT by trial and performance group",
            x_label="Trial",
            y_label="RT [ms]",
            show=False,
        )
        figures["block"]["rt"] = plot_performance_group_trends(
            df,
            rt_col,
            x_col="block",
            participant_id_col=participant_id_col,
            group_col=group_col,
            within_series_reduce="median",
            line_center_reduce="median",
            overall_center_reduce="mean",
            title="RT by block and performance group",
            x_label="Block",
            y_label="Median RT [ms]",
            show=False,
        )

    if "search_distance" in df.columns:
        figures["trial"]["search_distance"] = plot_performance_group_trends(
            df,
            "search_distance",
            x_col="trial",
            participant_id_col=participant_id_col,
            group_col=group_col,
            within_series_reduce="mean",
            line_center_reduce="mean",
            overall_center_reduce="mean",
            title="Search distance by trial and performance group",
            x_label="Trial",
            y_label="Mean search distance",
            show=False,
        )
        figures["block"]["search_distance"] = plot_performance_group_trends(
            df,
            "search_distance",
            x_col="block",
            participant_id_col=participant_id_col,
            group_col=group_col,
            within_series_reduce="median",
            line_center_reduce="median",
            overall_center_reduce="mean",
            title="Search distance by block and performance group",
            x_label="Block",
            y_label="Median search distance",
            show=False,
        )

    for value_col, label, title_prefix in [
        ("hit_global_max", "Global-max probability", "Global-max indicator"),
        ("hit_top_local_max", "Local-max probability", "Local-max indicator"),
    ]:
        if value_col in df.columns:
            figures["trial"][value_col] = plot_performance_group_trends(
                df,
                value_col,
                x_col="trial",
                participant_id_col=participant_id_col,
                group_col=group_col,
                within_series_reduce="mean",
                line_center_reduce="mean",
                overall_center_reduce="mean",
                title=f"{title_prefix} by trial and performance group",
                x_label="Trial",
                y_label=label,
                show=False,
            )
            figures["block"][value_col] = plot_performance_group_trends(
                df,
                value_col,
                x_col="block",
                participant_id_col=participant_id_col,
                group_col=group_col,
                within_series_reduce="mean",
                line_center_reduce="mean",
                overall_center_reduce="mean",
                title=f"{title_prefix} by block and performance group",
                x_label="Block",
                y_label=label,
                show=False,
            )

    if show:
        plt.show()
    return figures
