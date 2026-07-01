#!/usr/bin/env python3
"""
Visualize bandit task behavior against reward environments.

Creates separate participant-level outputs. For each participant ID found in the
CSV, it writes one combined, multi-panel figure per visualization type:
  1. <participant-tag>_heatmaps_all_envs.png
     One heatmap subplot per environment/grid, with that participant's choice
     paths overlaid.
  2. <participant-tag>_normalized_rewards_all_envs.png
     One reward-over-trials subplot per environment/grid. Rewards are normalized
     to [0, 1] using the min and max payoff available on that grid.
  3. <participant-tag>_reaction_times_all_envs.png
     One reaction-time-over-trials subplot per environment/grid, if an RT column
     is present.
  4. <participant-tag>_maxima_indicators_all_envs.png
     One binary-indicator subplot per environment/grid. Indicators equal 1 when
     the choice is within --max-radius of the global maximum or within
     --max-radius of one of the top-N local maxima excluding the global maximum.

Also writes <participant-tag>_behavior_with_peak_flags.csv for each participant,
which is that participant's behavioral CSV plus these derived columns:
  - global_max_x, global_max_y
  - top_local_max_coords
  - hit_global_max
  - hit_top_local_max
  - payoff_at_choice
  - normalized_score

Typical use:
    python visualize_bandit_data.py \
        --data import_test/socContPilot.csv \
        --grid-loader gridLoader.js \
        --grid-dir . \
        --out-dir figures

Single-environment/demo use:
    python visualize_bandit_data.py \
        --data import_test/socContPilot.csv \
        --env-file MH_GP_g20_l3_s1.json \
        --out-dir figures_demo
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Coord = Tuple[int, int]

NOTEBOOK_VISUALIZATIONS = [
    "choice_heatmaps",
    "normalized_rewards",
    "reaction_times",
    "search_distance",
]


def parse_json_grid_files(grid_loader_path: Path) -> List[str]:
    """Extract active jsonGridFiles from gridLoader.js, ignoring comments."""
    text = grid_loader_path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    matches = re.findall(r"jsonGridFiles\s*:\s*\[(.*?)\]", text, flags=re.S)
    if not matches:
        return []
    return re.findall(r"['\"]([^'\"]+\.json)['\"]", matches[-1])


def load_json_grid(path: Path) -> pd.DataFrame:
    """Load one grid JSON into columns x, y, payoff."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for row in raw:
        if isinstance(row, list):
            cells.extend(row)
        else:
            cells.append(row)

    grid = pd.DataFrame(cells).rename(columns={"x1": "x", "x2": "y"})
    missing = {"x", "y", "payoff"}.difference(grid.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    grid["x"] = grid["x"].astype(int)
    grid["y"] = grid["y"].astype(int)
    grid["payoff"] = pd.to_numeric(grid["payoff"], errors="coerce")
    return grid[["x", "y", "payoff"]]


def load_envs(
    grid_loader: Optional[Path],
    grid_dir: Optional[Path],
    env_file: Optional[Path],
) -> Dict[int, pd.DataFrame]:
    """Load environment grids keyed by gridLoader index or -1 for --env-file."""
    envs: Dict[int, pd.DataFrame] = {}

    if grid_loader is not None:
        base = grid_dir if grid_dir is not None else grid_loader.parent
        for idx, rel in enumerate(parse_json_grid_files(grid_loader)):
            candidates = [base / rel, base / Path(rel).name]
            path = next((p for p in candidates if p.exists()), None)
            if path is not None:
                envs[idx] = load_json_grid(path)
            else:
                print(f"Warning: gridLoader env {idx} not found: {rel}")

    if env_file is not None:
        envs[-1] = load_json_grid(env_file)

    if not envs:
        raise FileNotFoundError(
            "No environments loaded. Provide --grid-loader with available JSON files "
            "or provide --env-file."
        )

    return envs


def grid_to_matrix(grid: pd.DataFrame) -> np.ndarray:
    """Convert grid DataFrame to a y-by-x matrix indexed from min coordinate."""
    min_x, max_x = int(grid.x.min()), int(grid.x.max())
    min_y, max_y = int(grid.y.min()), int(grid.y.max())
    mat = np.full((max_y - min_y + 1, max_x - min_x + 1), np.nan)

    for row in grid.itertuples(index=False):
        mat[int(row.y) - min_y, int(row.x) - min_x] = float(row.payoff)

    return mat


def peak_coordinates(grid: pd.DataFrame, top_n_local: int = 3) -> Tuple[Coord, List[Coord]]:
    """Return global maximum coordinate and top local-maximum coordinates.

    Local maxima are cells whose payoff is >= all 8 immediate neighbors. The
    global maximum is excluded before taking the top N local peaks. If fewer
    than N local peaks exist, the function falls back to the highest remaining
    coordinates so the indicator still has N comparison points.
    """
    mat = grid_to_matrix(grid)
    min_x, min_y = int(grid.x.min()), int(grid.y.min())

    global_idx = np.unravel_index(np.nanargmax(mat), mat.shape)
    global_coord = (global_idx[1] + min_x, global_idx[0] + min_y)

    local_peaks: List[Tuple[Coord, float]] = []
    n_y, n_x = mat.shape

    for iy in range(n_y):
        for ix in range(n_x):
            val = mat[iy, ix]
            if np.isnan(val):
                continue

            coord = (ix + min_x, iy + min_y)
            if coord == global_coord:
                continue

            y0, y1 = max(0, iy - 1), min(n_y, iy + 2)
            x0, x1 = max(0, ix - 1), min(n_x, ix + 2)
            neigh = mat[y0:y1, x0:x1]

            if np.all(val >= neigh[~np.isnan(neigh)]):
                local_peaks.append((coord, float(val)))

    local_peaks.sort(key=lambda item: item[1], reverse=True)
    coords = [coord for coord, _ in local_peaks[:top_n_local]]

    if len(coords) < top_n_local:
        ranked = (
            grid.assign(coord=list(zip(grid.x.astype(int), grid.y.astype(int))))
            .query("coord != @global_coord")
            .sort_values("payoff", ascending=False)
        )
        for coord in ranked["coord"]:
            if coord not in coords:
                coords.append(coord)
            if len(coords) == top_n_local:
                break

    return global_coord, coords


def choose_env(row_env: int, envs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Resolve the experiment env value to a loaded grid."""
    if row_env in envs:
        return envs[row_env]
    if row_env - 1 in envs:
        return envs[row_env - 1]
    if -1 in envs:
        return envs[-1]
    if len(envs) == 1:
        return next(iter(envs.values()))
    raise KeyError(f"No loaded grid for env={row_env}; available keys={sorted(envs)}")


def euclidean_distance(a: Coord, b: Coord) -> float:
    """Euclidean distance between two grid coordinates."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def add_peak_flags(
    df: pd.DataFrame,
    envs: Dict[int, pd.DataFrame],
    top_n_local: int,
    max_radius: float = 1.0,
) -> pd.DataFrame:
    """Add maxima coordinates, radius-based indicators, and normalized rewards."""
    out = df.copy()
    peak_cache: Dict[int, Tuple[Coord, List[Coord]]] = {}

    def get_peaks(env_value: int) -> Tuple[Coord, List[Coord]]:
        if env_value not in peak_cache:
            peak_cache[env_value] = peak_coordinates(choose_env(env_value, envs), top_n_local)
        return peak_cache[env_value]

    global_x, global_y, local_coords = [], [], []
    hit_global, hit_local = [], []
    payoff_at_choice, norm_scores = [], []

    for row in out.itertuples(index=False):
        env_value = int(getattr(row, "env"))
        env_grid = choose_env(env_value, envs)
        global_coord, locals_ = get_peaks(env_value)
        choice = (int(getattr(row, "choice_x")), int(getattr(row, "choice_y")))

        grid_min = float(env_grid["payoff"].min())
        grid_max = float(env_grid["payoff"].max())
        match = env_grid[(env_grid["x"] == choice[0]) & (env_grid["y"] == choice[1])]
        chosen_payoff = float(match["payoff"].iloc[0]) if not match.empty else np.nan

        payoff_at_choice.append(chosen_payoff)
        if grid_max > grid_min and not np.isnan(chosen_payoff):
            norm_scores.append((chosen_payoff - grid_min) / (grid_max - grid_min))
        else:
            norm_scores.append(np.nan)

        global_x.append(global_coord[0])
        global_y.append(global_coord[1])
        local_coords.append(";".join(f"{x},{y}" for x, y in locals_))
        hit_global.append(int(euclidean_distance(choice, global_coord) <= max_radius))
        hit_local.append(int(any(euclidean_distance(choice, loc) <= max_radius for loc in locals_)))

    out["global_max_x"] = global_x
    out["global_max_y"] = global_y
    out["top_local_max_coords"] = local_coords
    out["hit_global_max"] = hit_global
    out["hit_top_local_max"] = hit_local
    out["payoff_at_choice"] = payoff_at_choice
    out["normalized_score"] = norm_scores
    return out


def find_rt_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first recognized reaction-time column name, if present."""
    candidates = ["RT", "rt", "reaction_time", "reactionTime", "response_time", "responseTime"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def unique_envs_in_order(df: pd.DataFrame) -> List[int]:
    return sorted(int(e) for e in pd.unique(df["env"]) if pd.notna(e))


def subplot_grid(n_panels: int, max_cols: int = 5) -> Tuple[int, int]:
    """Choose a compact rows-by-cols layout."""
    if n_panels <= 0:
        return 1, 1
    ncols = min(max_cols, math.ceil(math.sqrt(n_panels)))
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def safe_filename_token(value: object) -> str:
    """Convert an ID value into a filesystem-friendly string."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text or "unknown"


def participant_filename_tag(df: pd.DataFrame, id_col: str = "participant_index") -> str:
    """Create a short participant tag for output filenames."""
    if id_col not in df.columns:
        raise ValueError(f"Filename ID column not found: {id_col}")
    participants = sorted(safe_filename_token(p) for p in pd.unique(df[id_col]) if pd.notna(p))
    if len(participants) == 1:
        return f"participant-{participants[0]}"
    return "all-participants"




def choose_participant_id_col(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    """Choose the participant identifier column for per-participant outputs.

    Default priority is workerID, PROLIFIC_PID, id, then participant_index. The
    import script preserves id and workerID from raw Prolific-style CSVs, so
    workerID will usually be used automatically.
    """
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Participant ID column not found: {requested}")
        return requested

    for col in ["workerID", "PROLIFIC_PID", "id", "participant_index"]:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a participant ID column. Expected one of: "
        "workerID, PROLIFIC_PID, id, participant_index."
    )


def iter_participant_frames(
    df: pd.DataFrame,
    id_col: str,
    only_participant_id: Optional[str] = None,
) -> Iterable[Tuple[object, pd.DataFrame]]:
    """Yield one dataframe per participant ID."""
    work = df.copy()
    if only_participant_id is not None:
        work = work[work[id_col].astype(str) == str(only_participant_id)]
        if work.empty:
            raise ValueError(f"No rows found for {id_col}={only_participant_id}")

    for pid, participant_df in work.groupby(id_col, sort=True, dropna=False):
        yield pid, participant_df.copy()


def flatten_axes(axes) -> List[plt.Axes]:
    return list(np.asarray(axes).reshape(-1))


def iter_trajectories(env_df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """Yield one sorted trajectory per participant/block within an environment."""
    for _, group in env_df.groupby(["participant_index", "block"], sort=True):
        yield group.sort_values("trial")


def save_combined_heatmaps(
    df: pd.DataFrame,
    envs: Dict[int, pd.DataFrame],
    out_path: Path,
    max_cols: int = 5,
) -> None:
    env_values = unique_envs_in_order(df)
    nrows, ncols = subplot_grid(len(env_values), max_cols=max_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols + 0.8, 3.0 * nrows), squeeze=False)
    axes_flat = flatten_axes(axes)

    images = []
    for ax, env_value in zip(axes_flat, env_values):
        env_grid = choose_env(env_value, envs)
        mat = grid_to_matrix(env_grid)
        min_x, max_x = int(env_grid.x.min()), int(env_grid.x.max())
        min_y, max_y = int(env_grid.y.min()), int(env_grid.y.max())

        im = ax.imshow(
            mat,
            origin="lower",
            extent=[min_x - 0.5, max_x + 0.5, min_y - 0.5, max_y + 0.5],
            aspect="equal",
        )
        images.append(im)

        env_df = df[df["env"].astype(int) == env_value]
        for traj in iter_trajectories(env_df):
            ax.plot(
                traj["choice_x"],
                traj["choice_y"],
                marker="o",
                linewidth=0.8,
                markersize=2.2,
                color="black",
                alpha=0.75,
            )
            last_click = traj.tail(1)
            if not last_click.empty:
                ax.scatter(
                    last_click["choice_x"],
                    last_click["choice_y"],
                    marker="P",
                    s=55,
                    color="black",
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=5,
                )

        global_coord, local_coords = peak_coordinates(env_grid)
        ax.scatter(
            [global_coord[0]],
            [global_coord[1]],
            marker="*",
            s=80,
            color="tab:blue",
            edgecolors="white",
            linewidths=0.4,
            zorder=6,
        )
        if local_coords:
            ax.scatter(
                [c[0] for c in local_coords],
                [c[1] for c in local_coords],
                marker="X",
                s=45,
                color="tab:orange",
                edgecolors="white",
                linewidths=0.4,
                zorder=6,
            )

        ax.set_title(f"env {env_value}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)

    for ax in axes_flat[len(env_values):]:
        ax.axis("off")

    if images:
        # Reserve a dedicated strip on the right so the colorbar cannot cover subplots.
        cbar_ax = fig.add_axes([0.92, 0.16, 0.018, 0.72])
        fig.colorbar(images[0], cax=cbar_ax, label="Payoff")

    legend_handles = [
        plt.Line2D([0], [0], color="black", marker="o", linewidth=0.8, markersize=3, label="choice path"),
        plt.Line2D([0], [0], color="black", marker="P", linewidth=0, markersize=7, label="last click"),
        plt.Line2D([0], [0], color="tab:blue", marker="*", linewidth=0, markersize=9, label="global max"),
        plt.Line2D([0], [0], color="tab:orange", marker="X", linewidth=0, markersize=7, label="top local max"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.46, 0.005),
    )
    fig.suptitle("Choices over environment heatmaps", fontsize=11)
    fig.tight_layout(rect=[0, 0.045, 0.90, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_combined_line_panels(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    y_label: str,
    ylim: Optional[Tuple[float, float]] = None,
    step: bool = False,
    max_cols: int = 5,
) -> None:
    env_values = unique_envs_in_order(df)
    nrows, ncols = subplot_grid(len(env_values), max_cols=max_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False, sharex=True)
    axes_flat = flatten_axes(axes)

    for ax, env_value in zip(axes_flat, env_values):
        env_df = df[df["env"].astype(int) == env_value]
        for traj in iter_trajectories(env_df):
            if step:
                ax.step(traj["trial"], traj[value_col], where="mid", linewidth=0.9, alpha=0.65)
            else:
                ax.plot(traj["trial"], traj[value_col], linewidth=0.9, alpha=0.65)

        ax.set_title(f"env {env_value}", fontsize=9)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7, length=2)
        ax.grid(True, linewidth=0.25, alpha=0.35)

    for ax in axes_flat[len(env_values):]:
        ax.axis("off")

    fig.supxlabel("Trial", fontsize=9)
    fig.supylabel(y_label, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_combined_maxima_indicators(
    df: pd.DataFrame,
    out_path: Path,
    max_cols: int = 5,
) -> None:
    env_values = unique_envs_in_order(df)
    nrows, ncols = subplot_grid(len(env_values), max_cols=max_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False, sharex=True, sharey=True)
    axes_flat = flatten_axes(axes)

    for ax, env_value in zip(axes_flat, env_values):
        env_df = df[df["env"].astype(int) == env_value]
        for traj in iter_trajectories(env_df):
            # No vertical offset: both series are true binary 0/1 values.
            ax.step(
                traj["trial"],
                traj["hit_global_max"],
                where="mid",
                linewidth=0.9,
                alpha=0.65,
                color="tab:blue",
            )
            ax.step(
                traj["trial"],
                traj["hit_top_local_max"],
                where="mid",
                linewidth=0.9,
                alpha=0.65,
                linestyle="--",
                color="tab:orange",
            )

        ax.set_title(f"env {env_value}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=7, length=2)
        ax.grid(True, linewidth=0.25, alpha=0.35)

    for ax in axes_flat[len(env_values):]:
        ax.axis("off")

    # A single compact legend avoids repeating text in every panel.
    legend_lines = [
        plt.Line2D([0], [0], color="tab:blue", linewidth=1.0, label="global radius"),
        plt.Line2D([0], [0], color="tab:orange", linewidth=1.0, linestyle="--", label="local radius"),
    ]
    fig.legend(handles=legend_lines, loc="upper right", frameon=False, fontsize=8)
    fig.supxlabel("Trial", fontsize=9)
    fig.supylabel("Indicator", fontsize=9)
    fig.suptitle("Maxima-radius indicators", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.02, 0.96, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_search_distance(df: pd.DataFrame) -> pd.Series:
    """Compute per-trial search distance from the previous choice in each trajectory."""
    work = df.copy()
    for col in ["trial", "choice_x", "choice_y"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.sort_values(["participant_index", "block", "env", "trial"], kind="mergesort")
    distances = pd.Series(np.nan, index=work.index, dtype=float)

    for _, traj in work.groupby(["participant_index", "block", "env"], sort=False, dropna=False):
        ordered = traj.sort_values("trial")
        if len(ordered) <= 1:
            continue

        prev_xy = ordered[["choice_x", "choice_y"]].shift(1)
        curr_xy = ordered[["choice_x", "choice_y"]]
        delta = (curr_xy - prev_xy).to_numpy(dtype=float)
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        dist[0] = np.nan
        distances.loc[ordered.index] = dist

    return distances


def prepare_bandit_data(
    df: pd.DataFrame,
    envs: Optional[Dict[int, pd.DataFrame]],
    *,
    top_n_local: int = 3,
    max_radius: float = 3.0,
    participant_id_col: Optional[str] = None,
    participant_id: Optional[str] = None,
    block: Optional[int] = None,
    rt_col: Optional[str] = None,
    ensure_numeric: bool = True,
) -> pd.DataFrame:
    """Prepare a bandit dataframe for notebook-style plotting and analysis."""
    work = df.copy()

    if participant_id is not None:
        if participant_id_col is None:
            participant_id_col = choose_participant_id_col(work)
        if participant_id_col not in work.columns:
            raise ValueError(f"Participant ID column not found: {participant_id_col}")
        work = work[work[participant_id_col].astype(str) == str(participant_id)]
        if work.empty:
            raise ValueError(f"No rows found for {participant_id_col}={participant_id}")

    if block is not None:
        if "block" not in work.columns:
            raise ValueError("block column not found")
        work = work[work["block"] == block]
        if work.empty:
            raise ValueError(f"No rows remain after filtering for block={block}")

    if ensure_numeric:
        for col in ["env", "trial", "choice_x", "choice_y", "score"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["env", "trial", "choice_x", "choice_y"])
        work["env"] = work["env"].astype(int)
        work["trial"] = work["trial"].astype(int)
        work["choice_x"] = work["choice_x"].astype(int)
        work["choice_y"] = work["choice_y"].astype(int)

    if rt_col is None:
        rt_col = find_rt_column(work)
    if rt_col is not None and rt_col in work.columns:
        work[rt_col] = pd.to_numeric(work[rt_col], errors="coerce")

    work["search_distance"] = compute_search_distance(work)
    if envs:
        work = add_peak_flags(work, envs, top_n_local, max_radius)
    elif "normalized_score" not in work.columns and "score" in work.columns:
        work["normalized_score"] = pd.to_numeric(work["score"], errors="coerce")
    return work.sort_values(["participant_index", "env", "block", "trial"], kind="mergesort").reset_index(drop=True)


def plot_bandit_heatmaps(
    df: pd.DataFrame,
    envs: Dict[int, pd.DataFrame],
    *,
    max_cols: int = 5,
    title: str = "Choices over environment heatmaps",
    show: bool = False,
):
    """Plot a heatmap for each environment with choice trajectories overlaid."""
    env_values = unique_envs_in_order(df)
    nrows, ncols = subplot_grid(len(env_values), max_cols=max_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols + 0.8, 3.0 * nrows), squeeze=False)
    axes_flat = flatten_axes(axes)

    for ax, env_value in zip(axes_flat, env_values):
        env_grid = choose_env(env_value, envs)
        mat = grid_to_matrix(env_grid)
        min_x, max_x = int(env_grid.x.min()), int(env_grid.x.max())
        min_y, max_y = int(env_grid.y.min()), int(env_grid.y.max())

        ax.imshow(
            mat,
            origin="lower",
            extent=[min_x - 0.5, max_x + 0.5, min_y - 0.5, max_y + 0.5],
            aspect="equal",
        )

        env_df = df[df["env"].astype(int) == env_value]
        for traj in iter_trajectories(env_df):
            ax.plot(
                traj["choice_x"],
                traj["choice_y"],
                marker="o",
                linewidth=0.8,
                markersize=2.2,
                color="black",
                alpha=0.75,
            )

        ax.set_title(f"env {env_value}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[len(env_values):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes_flat


def plot_trial_series(
    df: pd.DataFrame,
    *,
    value_col: str,
    max_cols: int = 5,
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot a per-trial series for each environment."""
    env_values = unique_envs_in_order(df)
    nrows, ncols = subplot_grid(len(env_values), max_cols=max_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False, sharex=True)
    axes_flat = flatten_axes(axes)

    for ax, env_value in zip(axes_flat, env_values):
        env_df = df[df["env"].astype(int) == env_value]
        for traj in iter_trajectories(env_df):
            ax.plot(traj["trial"], traj[value_col], linewidth=0.9, alpha=0.65)

        ax.set_title(f"env {env_value}", fontsize=9)
        ax.grid(True, linewidth=0.25, alpha=0.35)

    for ax in axes_flat[len(env_values):]:
        ax.axis("off")

    fig.supxlabel("Trial", fontsize=9)
    fig.supylabel(value_col.replace("_", " "), fontsize=9)
    fig.suptitle(title or f"{value_col} over trials", fontsize=11)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes_flat


def plot_reaction_times(
    df: pd.DataFrame,
    *,
    rt_col: Optional[str] = None,
    max_cols: int = 5,
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot reaction-time traces for each environment."""
    rt_col = rt_col or find_rt_column(df)
    if rt_col is None:
        raise ValueError("No reaction-time column was found in the dataframe")
    return plot_trial_series(df, value_col=rt_col, max_cols=max_cols, title=title or f"{rt_col} over trials", show=show)


def plot_search_distance(
    df: pd.DataFrame,
    *,
    max_cols: int = 5,
    title: str = "Search-distance traces",
    show: bool = False,
):
    """Plot the distance between consecutive choices over trials."""
    if "search_distance" not in df.columns:
        df = df.copy()
        df["search_distance"] = compute_search_distance(df)
    return plot_trial_series(df, value_col="search_distance", max_cols=max_cols, title=title, show=show)


def make_preliminary_visualizations(
    df: pd.DataFrame,
    envs: Optional[Dict[int, pd.DataFrame]],
    *,
    participant_id_col: Optional[str] = None,
    participant_id: Optional[str] = None,
    block: Optional[int] = None,
    top_n_local: int = 3,
    max_radius: float = 1.0,
    max_cols: int = 5,
    show: bool = False,
):
    """Create the core notebook visualizations for a prepared bandit dataframe."""
    prepared = prepare_bandit_data(
        df,
        envs,
        top_n_local=top_n_local,
        max_radius=max_radius,
        participant_id_col=participant_id_col,
        participant_id=participant_id,
        block=block,
    )

    figures = {}
    if envs:
        figures["choice_heatmaps"] = plot_bandit_heatmaps(prepared, envs, max_cols=max_cols, show=False)

    if "normalized_score" in prepared.columns:
        figures["normalized_rewards"] = plot_trial_series(
            prepared,
            value_col="normalized_score",
            max_cols=max_cols,
            title="Normalized rewards over trials",
            show=False,
        )
    elif "score" in prepared.columns:
        figures["normalized_rewards"] = plot_trial_series(
            prepared,
            value_col="score",
            max_cols=max_cols,
            title="Score over trials",
            show=False,
        )

    rt_col = find_rt_column(prepared)
    if rt_col is not None:
        figures["reaction_times"] = plot_reaction_times(prepared, rt_col=rt_col, max_cols=max_cols, show=False)

    figures["search_distance"] = plot_search_distance(prepared, max_cols=max_cols, show=False)

    if show:
        plt.show()
    return prepared, figures


def summarize_group_average_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    trial_col: str = "trial",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return env-level and overall summaries for a trial-by-trial measure."""
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")

    pid_col = choose_participant_id_col(df, participant_id_col)
    work = df[[pid_col, env_col, trial_col, value_col]].copy()
    work[env_col] = pd.to_numeric(work[env_col], errors="coerce")
    work[trial_col] = pd.to_numeric(work[trial_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[env_col, trial_col, value_col])

    participant_means = (
        work.groupby([env_col, trial_col, pid_col], dropna=False)[value_col]
        .mean()
        .reset_index(name="value")
    )

    def sem(values: pd.Series) -> float:
        values = values.dropna()
        if len(values) <= 1:
            return 0.0
        return float(values.std(ddof=1) / (len(values) ** 0.5))

    env_summary = (
        participant_means.groupby([env_col, trial_col], dropna=False)["value"]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    overall_summary = (
        env_summary.groupby(trial_col, dropna=False)[["mean", "sem", "n"]]
        .agg(mean=("mean", "mean"), sem=("sem", "mean"), n=("n", "sum"))
        .reset_index()
    )
    return env_summary, overall_summary


def plot_group_average_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    trial_col: str = "trial",
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot env-level means with shaded error bands and a pooled grand-average trend."""
    env_summary, overall_summary = summarize_group_average_trends(
        df,
        value_col,
        participant_id_col=participant_id_col,
        env_col=env_col,
        trial_col=trial_col,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharex=True, sharey=False)
    axes = np.atleast_1d(axes)

    for env_value, sub in env_summary.groupby(env_col, sort=True):
        axes[0].plot(
            sub[trial_col],
            sub["mean"],
            linewidth=1.2,
            alpha=0.9,
            label=f"env {int(env_value)}",
        )
        axes[0].fill_between(
            sub[trial_col],
            sub["mean"] - sub["sem"],
            sub["mean"] + sub["sem"],
            alpha=0.18,
        )

    axes[0].set_title("Mean across participants within each env")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel(value_col.replace("_", " "))
    axes[0].grid(True, alpha=0.3)
    if len(env_summary[env_col].dropna().unique()) <= 6:
        axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(
        overall_summary[trial_col],
        overall_summary["mean"],
        linewidth=1.2,
        alpha=0.9,
        color="tab:purple",
    )
    axes[1].fill_between(
        overall_summary[trial_col],
        overall_summary["mean"] - overall_summary["sem"],
        overall_summary["mean"] + overall_summary["sem"],
        alpha=0.18,
        color="tab:purple",
    )
    axes[1].set_title("Grand mean across envs")
    axes[1].set_xlabel("Trial")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title or f"{value_col} by trial (group means)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def summarize_participant_average_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    trial_col: str = "trial",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return participant-level and overall summaries for a trial-by-trial measure."""
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")

    pid_col = choose_participant_id_col(df, participant_id_col)
    work = df[[pid_col, env_col, trial_col, value_col]].copy()
    work[env_col] = pd.to_numeric(work[env_col], errors="coerce")
    work[trial_col] = pd.to_numeric(work[trial_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[pid_col, env_col, trial_col, value_col])

    participant_means = (
        work.groupby([pid_col, trial_col], dropna=False)[value_col]
        .mean()
        .reset_index(name="mean")
    )

    def sem(values: pd.Series) -> float:
        values = values.dropna()
        if len(values) <= 1:
            return 0.0
        return float(values.std(ddof=1) / (len(values) ** 0.5))

    participant_summary = (
        work.groupby([pid_col, trial_col], dropna=False)[value_col]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    overall_summary = (
        participant_summary.groupby(trial_col, dropna=False)[["mean", "sem", "n"]]
        .agg(mean=("mean", "mean"), sem=("sem", "mean"), n=("n", "sum"))
        .reset_index()
    )
    return participant_summary, overall_summary


def plot_participant_average_trends(
    df: pd.DataFrame,
    value_col: str,
    *,
    participant_id_col: Optional[str] = None,
    env_col: str = "env",
    trial_col: str = "trial",
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot participant-level means across environments with shaded error bands and a grand mean."""
    participant_summary, overall_summary = summarize_participant_average_trends(
        df,
        value_col,
        participant_id_col=participant_id_col,
        env_col=env_col,
        trial_col=trial_col,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharex=True, sharey=False)
    axes = np.atleast_1d(axes)

    for pid_value, sub in participant_summary.groupby(participant_id_col or choose_participant_id_col(df), sort=True):
        axes[0].plot(
            sub[trial_col],
            sub["mean"],
            linewidth=1.1,
            alpha=0.85,
            label=str(pid_value),
        )
        axes[0].fill_between(
            sub[trial_col],
            sub["mean"] - sub["sem"],
            sub["mean"] + sub["sem"],
            alpha=0.12,
        )

    axes[0].set_title("Mean across envs within each participant")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel(value_col.replace("_", " "))
    axes[0].grid(True, alpha=0.3)
    if participant_summary.shape[0] <= 20:
        axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(
        overall_summary[trial_col],
        overall_summary["mean"],
        linewidth=1.2,
        alpha=0.9,
        color="tab:green",
    )
    axes[1].fill_between(
        overall_summary[trial_col],
        overall_summary["mean"] - overall_summary["sem"],
        overall_summary["mean"] + overall_summary["sem"],
        alpha=0.18,
        color="tab:green",
    )
    axes[1].set_title("Grand mean across participants")
    axes[1].set_xlabel("Trial")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title or f"{value_col} by trial (participant means)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def make_group_average_visualizations(
    df: pd.DataFrame,
    *,
    participant_id_col: Optional[str] = None,
    title_prefix: str = "Group-average",
    show: bool = False,
):
    """Create a small suite of group-average plots for common bandit metrics."""
    figures = {}
    value_cols = []
    if "normalized_score" in df.columns:
        value_cols.append("normalized_score")
    elif "score" in df.columns:
        value_cols.append("score")

    rt_col = find_rt_column(df)
    if rt_col is not None:
        value_cols.append(rt_col)

    for col in ["hit_global_max", "hit_top_local_max", "search_distance"]:
        if col in df.columns:
            value_cols.append(col)

    for value_col in value_cols:
        figures[value_col] = plot_group_average_trends(
            df,
            value_col,
            participant_id_col=participant_id_col,
            title=f"{title_prefix}: {value_col}",
            show=False,
        )

    if show:
        plt.show()
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bandit choices and rewards against reward environments.")
    parser.add_argument("--data", required=True, type=Path, help="Processed behavioral CSV from ProlificImport_updated.py")
    parser.add_argument("--grid-loader", type=Path, help="gridLoader.js containing jsonGridFiles")
    parser.add_argument("--grid-dir", type=Path, help="Directory containing the JSON grid files from gridLoader.js")
    parser.add_argument("--env-file", type=Path, help="Single JSON environment file, used as fallback/demo environment")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Directory for PNG outputs and annotated CSV")
    parser.add_argument("--participant", type=int, help="Optional participant_index to plot; kept for backward compatibility")
    parser.add_argument("--participant-id", help="Optional participant identifier value to plot, using --participant-id-col or the auto-selected ID column")
    parser.add_argument("--participant-id-col", help="Column used to split outputs by participant; default priority: workerID, PROLIFIC_PID, id, participant_index")
    parser.add_argument("--block", type=int, help="Optional block to plot")
    parser.add_argument("--top-local", type=int, default=3, help="Number of local maxima to flag, excluding the global max")
    parser.add_argument("--max-radius", type=float, default=1.0, help="Euclidean radius around global/local maxima counted as a hit")
    parser.add_argument("--max-cols", type=int, default=5, help="Maximum columns in combined subplot figures")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    required = {"participant_index", "block", "env", "trial", "choice_x", "choice_y", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    if args.participant is not None:
        df = df[df["participant_index"] == args.participant]
    if args.block is not None:
        df = df[df["block"] == args.block]
    if df.empty:
        raise ValueError("No rows remain after participant/block filtering.")

    participant_id_col = choose_participant_id_col(df, args.participant_id_col)

    envs = load_envs(args.grid_loader, args.grid_dir, args.env_file)
    df = prepare_bandit_data(
        df,
        envs,
        top_n_local=args.top_local,
        max_radius=args.max_radius,
        participant_id_col=participant_id_col,
        participant_id=args.participant_id,
        block=args.block,
    )

    rt_col = find_rt_column(df)
    if rt_col is not None:
        df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")
    else:
        print("Warning: no reaction-time column found; skipping reaction-time plot.")

    n_participants = 0
    for participant_id, participant_df in iter_participant_frames(df, participant_id_col, args.participant_id):
        n_participants += 1
        filename_tag = f"participant-{safe_filename_token(participant_id)}"

        annotated_path = args.out_dir / f"{filename_tag}_behavior_with_peak_flags.csv"
        heatmap_path = args.out_dir / f"{filename_tag}_heatmaps_all_envs.png"
        rewards_path = args.out_dir / f"{filename_tag}_normalized_rewards_all_envs.png"
        indicators_path = args.out_dir / f"{filename_tag}_maxima_indicators_all_envs.png"

        participant_df.to_csv(annotated_path, index=False)

        save_combined_heatmaps(participant_df, envs, heatmap_path, max_cols=args.max_cols)
        save_combined_line_panels(
            participant_df,
            value_col="normalized_score",
            out_path=rewards_path,
            title=f"Normalized rewards: {participant_id_col}={participant_id}",
            y_label="Normalized reward",
            ylim=(-0.05, 1.05),
            step=False,
            max_cols=args.max_cols,
        )
        save_combined_maxima_indicators(participant_df, indicators_path, max_cols=args.max_cols)

        if rt_col is not None:
            rt_path = args.out_dir / f"{filename_tag}_reaction_times_all_envs.png"
            save_combined_line_panels(
                participant_df,
                value_col=rt_col,
                out_path=rt_path,
                title=f"Reaction times: {participant_id_col}={participant_id}",
                y_label=rt_col,
                ylim=None,
                step=False,
                max_cols=args.max_cols,
            )
            print(f"Wrote reaction-time plot: {rt_path}")

        print(f"Wrote annotated data: {annotated_path}")
        print(f"Wrote heatmap plot: {heatmap_path}")
        print(f"Wrote normalized reward plot: {rewards_path}")
        print(f"Wrote maxima indicator plot: {indicators_path}")

    print(f"Finished {n_participants} participant-level output set(s) using ID column: {participant_id_col}")


if __name__ == "__main__":
    main()
