#!/usr/bin/env python3
"""Summary analyses for bandit pilot data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_pilot_participant_plots_module():
    module_path = Path(__file__).with_name("pilot_participant_plots.py")
    spec = importlib.util.spec_from_file_location("pilot_participant_plots", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pilot_participant_plots from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_viz = _load_pilot_participant_plots_module()


def summarize_choice_reselection(
    df: pd.DataFrame, *, participant_id_col: Optional[str] = None,
    group_col: str = "performance_group",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return annotated trials, participant/env fractions, and group/env quartiles.

    Exact coordinate repetitions (including consecutive repeats) count only
    within a participant/environment/block. Every observed turn contributes to
    the denominator, including each block's first turn. Counts are pooled over
    blocks before calculating fractions; group quantiles weight participants
    equally and use linear interpolation. Input data and assignments are kept.
    """
    pid = _viz.choose_participant_id_col(df, participant_id_col)
    numeric = ["env", "block", "trial", "choice_x", "choice_y"]
    required = [pid, group_col, *numeric]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    work = df.copy()
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if not np.isfinite(work[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Environment, block, trial and coordinates must be finite numbers")
    for col in [pid, group_col]:
        if work[col].isna().any() or work[col].map(
            lambda value: isinstance(value, (int, float, np.number)) and not np.isfinite(value)
        ).any():
            raise ValueError("Participant and performance-group identifiers must be present and finite")
    membership = work[[pid, group_col]].drop_duplicates()
    if membership[pid].duplicated().any():
        raise ValueError("Each participant must have a single performance group")
    trajectory = [pid, "env", "block"]
    if work.duplicated([*trajectory, "trial"]).any():
        raise ValueError("Duplicate trials within a participant/environment/block")
    work = work.sort_values(["block", "trial"], kind="mergesort").copy()
    work["is_reselection"] = work.duplicated([*trajectory, "choice_x", "choice_y"])
    participants = (
        work.groupby([pid, "env"], observed=True, sort=False)["is_reselection"]
        .agg(n_turns="size", n_reselections="sum").reset_index()
        .merge(membership, on=pid, how="left", validate="many_to_one")
    )
    participants["reselection_fraction"] = participants.n_reselections / participants.n_turns
    groups = (
        participants.groupby([group_col, "env"], observed=True)["reselection_fraction"]
        .agg(n_participants="count", median="median",
             q25=lambda values: values.quantile(.25, interpolation="linear"),
             q75=lambda values: values.quantile(.75, interpolation="linear"))
        .reset_index()
    )
    groups["iqr"] = groups.q75 - groups.q25
    return work, participants, groups


def plot_choice_reselection_by_performance(
    participant_summary: pd.DataFrame, group_summary: pd.DataFrame, *,
    group_col: str = "performance_group", show: bool = False,
):
    """Participant fractions with group medians and Q25-Q75 intervals by env."""
    order = _performance_group_order_from_data(participant_summary, group_col)
    colors = _performance_group_colors(order)
    envs = sorted(participant_summary.env.unique())
    positions = {env: index for index, env in enumerate(envs)}
    fig, axes = plt.subplots(1, max(1, len(order)), figsize=(6 * max(1, len(order)), 4),
                             sharex=True, sharey=True, squeeze=False)
    for ax, group in zip(axes.flat, order):
        points = participant_summary.loc[participant_summary[group_col].eq(group)]
        stats = group_summary.loc[group_summary[group_col].eq(group)].sort_values("env")
        ax.scatter(points.env.map(positions), points.reselection_fraction,
                   color=colors[group], alpha=.25, s=22, label="Participants")
        if stats.empty:
            ax.text(.5, .5, "No observations", ha="center", transform=ax.transAxes)
        else:
            ax.errorbar(stats.env.map(positions), stats["median"],
                        yerr=[stats["median"] - stats.q25, stats.q75 - stats["median"]],
                        fmt="o", capsize=4, color=colors[group], label="Median and Q25-Q75")
        ax.set(title=str(group), xticks=list(positions.values()), xticklabels=[str(e) for e in envs],
               ylim=(0, 1), xlabel="Environment")
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    if not order:
        axes.flat[0].text(.5, .5, "No observations", ha="center", transform=axes.flat[0].transAxes)
        axes.flat[0].set_ylim(0, 1)
    axes.flat[0].set_ylabel("Fraction of turns reselecting a past choice")
    fig.suptitle("Reselection by performance group")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


FIRST_HIT_METRICS = (
    "search_distance_median", "search_distance_iqr", "reward_median", "reward_iqr",
)


def _iqr(values: pd.Series) -> float:
    """Q75 minus Q25 using linear interpolation; empty=NaN, singleton=0."""
    return float(values.quantile(0.75, interpolation="linear") - values.quantile(0.25, interpolation="linear"))


def extract_first_global_max_windows(
    df: pd.DataFrame, *, window_size: int = 5,
    participant_id_col: Optional[str] = None, reward_col: str = "normalized_score",
) -> pd.DataFrame:
    """One first radius-hit per participant/environment, selected across blocks.

    Windows exclude the hit trial (including its incoming movement). A complete
    window needs consecutive trial numbers in the same block and finite values.
    IQR is Q75 minus Q25 with linear interpolation (zero for a singleton). Ineligible first hits are retained;
    later hits never replace them. Input data are not modified.
    """
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    pid = _viz.choose_participant_id_col(df, participant_id_col)
    required = [pid, "env", "block", "trial", "hit_global_max", "search_distance", reward_col]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    work = df[required].copy()
    for col in required[1:]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if work[[pid, "env", "block", "trial"]].isna().any().any():
        raise ValueError("Participant, environment, block and trial identifiers must be present")
    if work.duplicated([pid, "env", "block", "trial"]).any():
        raise ValueError("Duplicate trials within a participant/environment/block")
    work = work.sort_values(["block", "trial"], kind="mergesort")
    rows = []
    for (participant, env), trajectory in work.groupby([pid, "env"], sort=False, observed=True):
        hits = trajectory.loc[trajectory.hit_global_max.eq(1)]
        if hits.empty:
            continue
        hit = hits.iloc[0]
        prior = trajectory.loc[
            trajectory.block.eq(hit.block) & trajectory.trial.lt(hit.trial)
        ].tail(window_size)
        reason = ""
        if len(prior) < window_size:
            reason = "insufficient_history"
        elif not np.array_equal(prior.trial.to_numpy(), np.arange(hit.trial - window_size, hit.trial)):
            reason = "nonconsecutive_trials"
        elif not np.isfinite(prior[["search_distance", reward_col]].to_numpy(dtype=float)).all():
            reason = "nonfinite_window_values"
        row = {pid: participant, "env": env, "hit_block": hit.block,
               "hit_trial": hit.trial, "window_size": window_size,
               "eligible": not reason, "exclusion_reason": reason}
        for source, prefix in [("search_distance", "search_distance"), (reward_col, "reward")]:
            row[f"{prefix}_median"] = prior[source].median() if not reason else np.nan
            row[f"{prefix}_iqr"] = _iqr(prior[source]) if not reason else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=[pid, "env", "hit_block", "hit_trial", "window_size",
                                      "eligible", "exclusion_reason", *FIRST_HIT_METRICS])


def summarize_first_global_max_windows(
    events: pd.DataFrame, population: pd.DataFrame, *,
    participant_id_col: Optional[str] = None, group_col: str = "performance_group",
):
    """Return participant and group tables using the full assigned population.

    Participant metrics describe distributions across eligible environments.
    Group metric medians/IQRs describe participant medians, with equal weight per
    participant. Coverage includes participants without any eligible first hits.
    """
    pid = _viz.choose_participant_id_col(population, participant_id_col)
    membership = population[[pid, group_col]].drop_duplicates()
    if membership[pid].duplicated().any():
        raise ValueError("Each participant must have a single performance group")
    participants = membership.merge(
        population.groupby(pid, observed=True).env.nunique().rename("n_environments"), on=pid,
    )
    eligible = events.loc[events.eligible.eq(True)]
    for name, frame in [("n_first_hits", events), ("n_eligible_environments", eligible)]:
        counts = frame.groupby(pid, observed=True).size().rename(name)
        participants = participants.merge(counts, on=pid, how="left")
        participants[name] = participants[name].fillna(0).astype(int)
    participants["n_never_hit"] = participants.n_environments - participants.n_first_hits
    participants["n_ineligible_first_hits"] = participants.n_first_hits - participants.n_eligible_environments
    for metric in FIRST_HIT_METRICS:
        stats = eligible.groupby(pid, observed=True)[metric].agg(count="count", median="median", iqr=_iqr, min="min", max="max")
        stats = stats.add_prefix(f"{metric}_")
        participants = participants.merge(stats, on=pid, how="left")
        participants[f"{metric}_count"] = participants[f"{metric}_count"].fillna(0).astype(int)
    rows = []
    order = _performance_group_order_from_data(population, group_col)
    for group in order:
        members = participants.loc[participants[group_col].eq(group)]
        row = {group_col: group, "n_participants": len(members),
               "n_contributing_participants": int(members.n_eligible_environments.gt(0).sum())}
        for col in ["n_environments", "n_first_hits", "n_eligible_environments", "n_never_hit", "n_ineligible_first_hits"]:
            row[col] = int(members[col].sum())
        for metric in FIRST_HIT_METRICS:
            values = members[f"{metric}_median"].dropna()
            row.update({f"{metric}_median": values.median(), f"{metric}_iqr": _iqr(values),
                        f"{metric}_n_participants": len(values)})
        rows.append(row)
    return participants, pd.DataFrame(rows)


def plot_first_global_max_by_performance(
    events: pd.DataFrame, population: pd.DataFrame, *,
    participant_id_col: Optional[str] = None, group_col: str = "performance_group",
    bins: int = 15, show: bool = False,
):
    """Plot equal-participant histogram mixtures; dashed lines mark median of medians.

    Histogram densities are averaged to preserve equal participant weighting
    and unit area; no Gaussian fit is used.
    """
    pid = _viz.choose_participant_id_col(population, participant_id_col)
    membership = population[[pid, group_col]].drop_duplicates()
    work = events.loc[events.eligible.eq(True)].merge(membership, on=pid, validate="many_to_one")
    order = _performance_group_order_from_data(population, group_col)
    colors = _performance_group_colors(order)
    figures = {}
    for metric in FIRST_HIT_METRICS:
        edges = _viz.first_hit_histogram_edges(work[metric], bins)
        fig, ax = plt.subplots(figsize=(7, 4))
        for group in order:
            subset = work.loc[work[group_col].eq(group)]
            histograms, medians = [], []
            for _, participant in subset.groupby(pid, observed=True):
                values = pd.to_numeric(participant[metric], errors="coerce").dropna()
                if not values.empty:
                    counts, _ = np.histogram(values, bins=edges)
                    histograms.append(counts / len(values) / np.diff(edges))
                    medians.append(values.median())
            label = f"{group} (participants={len(medians)})"
            if histograms:
                ax.stairs(np.mean(histograms, axis=0), edges, color=colors[group], label=label)
                ax.axvline(np.median(medians), color=colors[group], linestyle="--", alpha=0.7)
            else:
                ax.plot([], [], color=colors[group], label=f"{label}: no eligible values")
        ax.set(xlabel=metric.replace("_", " "), ylabel="Density (equal participant weights)",
               title="Before Mexican Hat discovery")
        ax.legend()
        fig.tight_layout()
        figures[metric] = (fig, ax)
    if show:
        plt.show()
    return figures


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


def _performance_group_order(n_groups: int = 3) -> Sequence[str]:
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")
    if n_groups == 2:
        return ("low", "high")
    if n_groups == 3:
        return ("low", "middle", "high")
    return tuple(f"group_{index + 1}" for index in range(n_groups))


def _performance_group_order_from_data(df: pd.DataFrame, group_col: str) -> Sequence[str]:
    group_data = df[group_col]
    if isinstance(group_data.dtype, pd.CategoricalDtype):
        return tuple(str(category) for category in group_data.cat.categories)
    return tuple(str(group) for group in group_data.dropna().drop_duplicates())


def _performance_group_colors(group_order: Optional[Sequence[str]] = None) -> Dict[str, str]:
    base_colors = {
        "low": "tab:blue",
        "middle": "tab:orange",
        "high": "tab:green",
    }
    if group_order is None:
        return base_colors
    palette = plt.get_cmap("tab10").colors
    return {
        group_name: base_colors.get(group_name, palette[index % len(palette)])
        for index, group_name in enumerate(group_order)
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
    group_order = _performance_group_order_from_data(df, group_col)
    colors = _performance_group_colors(group_order)
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=False)

    for group_name in group_order:
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
    n_groups: int = 3,
) -> pd.DataFrame:
    """Divide participants into quantile groups based on overall reward."""
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

    group_order = _performance_group_order(n_groups)
    ranked = participant_reward[performance_col].rank(method="first")
    participant_reward["performance_group"] = pd.qcut(
        ranked,
        q=n_groups,
        labels=list(group_order),
    )

    merged = df.merge(participant_reward[[pid_col, "performance_group"]], on=pid_col, how="left")
    merged["performance_group"] = pd.Categorical(
        merged["performance_group"],
        categories=list(group_order),
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
    """Plot participant-level lines and grouped grand averages by performance group."""
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
    group_order = _performance_group_order_from_data(df, group_col)
    colors = _performance_group_colors(group_order)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharex=True, sharey=False)

    for group_name in group_order:
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
            title="Reward by round and performance group",
            x_label="Round",
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
            title="RT by round and performance group",
            x_label="Round",
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
            title="Search distance by round and performance group",
            x_label="Round",
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
                title=f"{title_prefix} by round and performance group",
                x_label="Round",
                y_label=label,
                show=False,
            )

    if show:
        plt.show()
    return figures
