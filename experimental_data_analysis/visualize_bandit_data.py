#!/usr/bin/env python3
"""
Visualize bandit task behavior against reward environments.

Creates:
  1. Heatmaps of each environment with participant choices overlaid.
  2. Reward trajectories over trials, normalized within each environment grid.
  3. Reaction-time trajectories over trials.
  4. Binary trajectories indicating whether each choice fell within a radius
     of the global maximum or one of the top-N local maxima excluding the
     global maximum.

This script is designed for CSVs produced by ProlificImport_updated.py and for
reward-grid JSON files loaded by gridLoader.js.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Coord = Tuple[int, int]


def parse_json_grid_files(grid_loader_path: Path) -> List[str]:
    text = grid_loader_path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    matches = re.findall(r"jsonGridFiles\s*:\s*\[(.*?)\]", text, flags=re.S)
    if not matches:
        return []
    return re.findall(r"['\"]([^'\"]+\.json)['\"]", matches[-1])


def load_json_grid(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for row in raw:
        cells.extend(row if isinstance(row, list) else [row])

    grid = pd.DataFrame(cells).rename(columns={"x1": "x", "x2": "y"})
    missing = {"x", "y", "payoff"}.difference(grid.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    grid["x"] = grid["x"].astype(int)
    grid["y"] = grid["y"].astype(int)
    grid["payoff"] = pd.to_numeric(grid["payoff"])
    return grid[["x", "y", "payoff"]]


def load_envs(
    grid_loader: Optional[Path],
    grid_dir: Optional[Path],
    env_file: Optional[Path],
) -> Dict[int, pd.DataFrame]:
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
        raise FileNotFoundError("No environments loaded.")

    return envs


def grid_to_matrix(grid: pd.DataFrame) -> np.ndarray:
    min_x, max_x = int(grid.x.min()), int(grid.x.max())
    min_y, max_y = int(grid.y.min()), int(grid.y.max())
    mat = np.full((max_y - min_y + 1, max_x - min_x + 1), np.nan)

    for row in grid.itertuples(index=False):
        mat[int(row.y) - min_y, int(row.x) - min_x] = float(row.payoff)

    return mat


def peak_coordinates(grid: pd.DataFrame, top_n_local: int = 3) -> Tuple[Coord, List[Coord]]:
    mat = grid_to_matrix(grid)
    min_x, min_y = int(grid.x.min()), int(grid.y.min())

    global_idx = np.unravel_index(np.nanargmax(mat), mat.shape)
    global_coord = (global_idx[1] + min_x, global_idx[0] + min_y)

    locals_: List[Tuple[Coord, float]] = []
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
                locals_.append((coord, float(val)))

    locals_.sort(key=lambda item: item[1], reverse=True)
    coords = [coord for coord, _ in locals_[:top_n_local]]

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
    if row_env in envs:
        return envs[row_env]
    if row_env - 1 in envs:
        return envs[row_env - 1]
    if -1 in envs:
        return envs[-1]
    if len(envs) == 1:
        return next(iter(envs.values()))

    raise KeyError(f"No loaded grid for env={row_env}; available keys={sorted(envs)}")


def distance(a: Coord, b: Coord) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def add_peak_flags(
    df: pd.DataFrame,
    envs: Dict[int, pd.DataFrame],
    top_n_local: int,
    max_radius: float = 1.0,
) -> pd.DataFrame:
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

        hit_global.append(int(distance(choice, global_coord) <= max_radius))
        hit_local.append(int(any(distance(choice, loc) <= max_radius for loc in locals_)))

    out["global_max_x"] = global_x
    out["global_max_y"] = global_y
    out["top_local_max_coords"] = local_coords
    out["hit_global_max"] = hit_global
    out["hit_top_local_max"] = hit_local
    out["payoff_at_choice"] = payoff_at_choice
    out["normalized_score"] = norm_scores

    return out


def plot_choice_heatmap(group: pd.DataFrame, env_grid: pd.DataFrame, out_path: Path, title: str) -> None:
    mat = grid_to_matrix(env_grid)
    min_x, max_x = int(env_grid.x.min()), int(env_grid.x.max())
    min_y, max_y = int(env_grid.y.min()), int(env_grid.y.max())

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        mat,
        origin="lower",
        extent=[min_x - 0.5, max_x + 0.5, min_y - 0.5, max_y + 0.5],
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Environment payoff")

    ordered = group.sort_values("trial")

    ax.plot(
        ordered["choice_x"],
        ordered["choice_y"],
        marker="o",
        linewidth=1,
        markersize=3,
        color="black",
        label="Choice path",
    )

    last_click = ordered.tail(1)
    if not last_click.empty:
        ax.scatter(
            last_click["choice_x"],
            last_click["choice_y"],
            marker="P",
            s=140,
            color="black",
            edgecolors="white",
            linewidths=0.8,
            label="Last click",
            zorder=5,
        )

    trial0 = ordered[ordered["trial"] == 0]
    if not trial0.empty:
        ax.scatter(trial0["choice_x"], trial0["choice_y"], marker="s", s=90, label="Initial reveal")

    global_coord, local_coords = peak_coordinates(env_grid)

    ax.scatter([global_coord[0]], [global_coord[1]], marker="*", s=180, label="Global max")

    if local_coords:
        ax.scatter(
            [c[0] for c in local_coords],
            [c[1] for c in local_coords],
            marker="X",
            s=90,
            label="Top local peaks",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(min_x, max_x + 1))
    ax.set_yticks(range(min_y, max_y + 1))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trials(group: pd.DataFrame, out_path: Path, title: str) -> None:
    ordered = group.sort_values("trial")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ordered["trial"], ordered["normalized_score"], marker="o", markersize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Trial (0 = automatic initial reveal)")
    ax.set_ylabel("Reward normalized within grid")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def find_rt_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["RT", "rt", "reaction_time", "reactionTime", "response_time", "responseTime"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def plot_reaction_time(group: pd.DataFrame, rt_col: str, out_path: Path, title: str) -> None:
    ordered = group.sort_values("trial").copy()
    ordered[rt_col] = pd.to_numeric(ordered[rt_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ordered["trial"], ordered[rt_col], marker="o", markersize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Trial (0 = automatic initial reveal)")
    ax.set_ylabel(f"Reaction time ({rt_col})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_peak_flags(group: pd.DataFrame, out_path: Path, title: str) -> None:
    ordered = group.sort_values("trial")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(ordered["trial"], ordered["hit_global_max"], where="mid", label="Within radius of global max")
    ax.step(
        ordered["trial"],
        ordered["hit_top_local_max"] + 0.05,
        where="mid",
        label="Within radius of top local max",
    )
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0, 1])
    ax.set_title(title)
    ax.set_xlabel("Trial (0 = automatic initial reveal)")
    ax.set_ylabel("Indicator")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bandit choices and rewards against reward environments.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--grid-loader", type=Path)
    parser.add_argument("--grid-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--participant", type=int)
    parser.add_argument("--block", type=int)
    parser.add_argument("--top-local", type=int, default=3)
    parser.add_argument("--max-radius", type=float, default=1.0)
    parser.add_argument("--max-heatmaps", type=int, default=50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_dir = args.out_dir / "heatmaps"
    reward_dir = args.out_dir / "normalized_rewards"
    rt_dir = args.out_dir / "reaction_times"
    binary_dir = args.out_dir / "maxima_indicators"

    for d in [heatmap_dir, reward_dir, rt_dir, binary_dir]:
        d.mkdir(parents=True, exist_ok=True)

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

    envs = load_envs(args.grid_loader, args.grid_dir, args.env_file)
    df = add_peak_flags(df, envs, args.top_local, args.max_radius)

    rt_col = find_rt_column(df)
    if rt_col is None:
        print("Warning: no reaction-time column found; skipping reaction-time plots.")

    annotated_path = args.out_dir / "behavior_with_peak_flags.csv"
    df.to_csv(annotated_path, index=False)

    heatmaps_saved = 0

    for (participant, block), group in df.groupby(["participant_index", "block"], sort=True):
        env_value = int(group["env"].iloc[0])
        env_grid = choose_env(env_value, envs)

        stem = f"participant-{participant}_block-{block}_env-{env_value}"

        if heatmaps_saved < args.max_heatmaps:
            plot_choice_heatmap(
                group,
                env_grid,
                heatmap_dir / f"{stem}_heatmap.png",
                f"Choices: participant {participant}, block {block}, env {env_value}",
            )
            heatmaps_saved += 1

        plot_trials(
            group,
            reward_dir / f"{stem}_normalized_rewards.png",
            f"Normalized rewards: participant {participant}, block {block}, env {env_value}",
        )

        if rt_col is not None:
            plot_reaction_time(
                group,
                rt_col,
                rt_dir / f"{stem}_reaction_time.png",
                f"Reaction time: participant {participant}, block {block}, env {env_value}",
            )

        plot_peak_flags(
            group,
            binary_dir / f"{stem}_maxima_indicators.png",
            f"Maxima-radius hits: participant {participant}, block {block}, env {env_value}",
        )

    print(f"Wrote annotated data: {annotated_path}")
    print(f"Wrote heatmaps to: {heatmap_dir} ({heatmaps_saved} saved)")
    print(f"Wrote normalized reward plots to: {reward_dir}")

    if rt_col is not None:
        print(f"Wrote reaction-time plots to: {rt_dir}")

    print(f"Wrote binary indicator plots to: {binary_dir}")


if __name__ == "__main__":
    main()