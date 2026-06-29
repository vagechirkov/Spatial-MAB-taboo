#!/usr/bin/env python3
"""
Visualize bandit task behavior against reward environments.

Creates one combined, multi-panel figure per visualization type:
  1. <participant-tag>_heatmaps_all_envs.png
     One heatmap subplot per environment/grid, with choice paths overlaid.
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

Also writes <participant-tag>_behavior_with_peak_flags.csv, which is the input behavioral CSV plus
these derived columns:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bandit choices and rewards against reward environments.")
    parser.add_argument("--data", required=True, type=Path, help="Processed behavioral CSV from ProlificImport_updated.py")
    parser.add_argument("--grid-loader", type=Path, help="gridLoader.js containing jsonGridFiles")
    parser.add_argument("--grid-dir", type=Path, help="Directory containing the JSON grid files from gridLoader.js")
    parser.add_argument("--env-file", type=Path, help="Single JSON environment file, used as fallback/demo environment")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Directory for PNG outputs and annotated CSV")
    parser.add_argument("--participant", type=int, help="Optional participant_index to plot")
    parser.add_argument("--block", type=int, help="Optional block to plot")
    parser.add_argument("--top-local", type=int, default=3, help="Number of local maxima to flag, excluding the global max")
    parser.add_argument("--max-radius", type=float, default=1.0, help="Euclidean radius around global/local maxima counted as a hit")
    parser.add_argument("--max-cols", type=int, default=5, help="Maximum columns in combined subplot figures")
    parser.add_argument(
        "--filename-id-col",
        default="participant_index",
        help="Column used to add participant ID to saved filenames",
    )
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

    # Ensure numeric plotting columns.
    for col in ["env", "trial", "choice_x", "choice_y", "score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["env", "trial", "choice_x", "choice_y"])
    df["env"] = df["env"].astype(int)
    df["trial"] = df["trial"].astype(int)
    df["choice_x"] = df["choice_x"].astype(int)
    df["choice_y"] = df["choice_y"].astype(int)

    envs = load_envs(args.grid_loader, args.grid_dir, args.env_file)
    df = add_peak_flags(df, envs, args.top_local, args.max_radius)

    rt_col = find_rt_column(df)
    if rt_col is not None:
        df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")
    else:
        print("Warning: no reaction-time column found; skipping reaction-time plot.")

    filename_tag = participant_filename_tag(df, args.filename_id_col)
    annotated_path = args.out_dir / f"{filename_tag}_behavior_with_peak_flags.csv"
    df.to_csv(annotated_path, index=False)
    heatmap_path = args.out_dir / f"{filename_tag}_heatmaps_all_envs.png"
    rewards_path = args.out_dir / f"{filename_tag}_normalized_rewards_all_envs.png"
    indicators_path = args.out_dir / f"{filename_tag}_maxima_indicators_all_envs.png"

    save_combined_heatmaps(df, envs, heatmap_path, max_cols=args.max_cols)
    save_combined_line_panels(
        df,
        value_col="normalized_score",
        out_path=rewards_path,
        title="Normalized rewards",
        y_label="Normalized reward",
        ylim=(-0.05, 1.05),
        step=False,
        max_cols=args.max_cols,
    )
    save_combined_maxima_indicators(df, indicators_path, max_cols=args.max_cols)

    if rt_col is not None:
        rt_path = args.out_dir / f"{filename_tag}_reaction_times_all_envs.png"
        save_combined_line_panels(
            df,
            value_col=rt_col,
            out_path=rt_path,
            title="Reaction times",
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


if __name__ == "__main__":
    main()
