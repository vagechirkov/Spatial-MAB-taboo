from collections.abc import Mapping

import matplotlib.pyplot as plt
import mesa
import numpy as np
import pandas as pd
import seaborn as sns


def run_condition_batches(
    *,
    model_cls,
    parameter_sets: Mapping[str, dict],
    n_runs: int,
    max_steps: int,
    data_collection_period: int = 1,
    number_processes: int | None = None,
    display_progress: bool = True,
    rng_values: list | None = None,
) -> pd.DataFrame:
    """Run a batch for each named parameter set and return one combined dataframe."""
    frames = []

    for condition_name, condition_parameters in parameter_sets.items():
        run_parameters = dict(condition_parameters)

        for reporter_key in ("model_reporters_to_collect", "agent_reporters_to_collect"):
            reporter_value = run_parameters.get(reporter_key)
            if (
                isinstance(reporter_value, list)
                and (len(reporter_value) == 0 or all(isinstance(item, str) for item in reporter_value))
            ):
                run_parameters[reporter_key] = [reporter_value]

        run_rng = None
        if rng_values is not None:
            run_rng = rng_values
        elif "rng" not in run_parameters:
            run_rng = [None] * n_runs

        results = mesa.batch_run(
            model_cls,
            parameters=run_parameters,
            max_steps=max_steps,
            data_collection_period=data_collection_period,
            display_progress=display_progress,
            number_processes=number_processes,
            rng=run_rng,
        )

        condition_df = pd.DataFrame(results)
        condition_df["condition"] = condition_name
        frames.append(condition_df)

    if len(frames) == 0:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def summarize_condition_timeseries(
    batch_df: pd.DataFrame,
    condition_col: str = "condition",
    step_col: str = "Step",
) -> pd.DataFrame:
    """Aggregate per-step metrics by condition."""
    required_columns = [
        condition_col,
        step_col,
        "cumulative_reward",
        "reward",
        "global_max",
        "local_max",
        "no_max",
    ]
    missing_columns = [column for column in required_columns if column not in batch_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for summary: {missing_columns}")

    summary_df = (
        batch_df.groupby([condition_col, step_col], as_index=False)
        .agg(
            {
                "cumulative_reward": "mean",
                "reward": "mean",
                "global_max": "mean",
                "local_max": "mean",
                "no_max": "mean",
            }
        )
        .sort_values([condition_col, step_col])
    )

    return summary_df


def mean_reward_last_steps_by_trajectory(
    batch_df: pd.DataFrame,
    window_size: int,
    *,
    reward_col: str = "reward",
    step_col: str = "Step",
    condition_col: str = "condition",
) -> pd.DataFrame:
    """
    Compute mean reward over the last `window_size` steps for each trajectory.

    A trajectory is identified by available columns among:
    [`condition_col`, `RunId`, `iteration`, `AgentID`].
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    required_columns = [step_col, reward_col]
    missing_columns = [column for column in required_columns if column not in batch_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for reward window summary: {missing_columns}")

    trajectory_columns = [
        column
        for column in [condition_col, "RunId", "iteration", "AgentID"]
        if column in batch_df.columns
    ]
    if len(trajectory_columns) == 0:
        raise KeyError(
            "Could not infer trajectory columns. Expected one or more of: "
            f"{[condition_col, 'RunId', 'iteration', 'AgentID']}"
        )

    window_column = f"mean_{reward_col}_last_{window_size}_steps"

    tail_df = (
        batch_df.sort_values(trajectory_columns + [step_col])
        .groupby(trajectory_columns, as_index=False, group_keys=False)
        .tail(window_size)
    )

    return (
        tail_df.groupby(trajectory_columns, as_index=False)[reward_col]
        .mean()
        .rename(columns={reward_col: window_column})
    )


def summarize_mean_reward_last_steps(
    batch_df: pd.DataFrame,
    window_size: int,
    *,
    reward_col: str = "reward",
    step_col: str = "Step",
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Aggregate last-window mean rewards by condition."""
    per_trajectory = mean_reward_last_steps_by_trajectory(
        batch_df=batch_df,
        window_size=window_size,
        reward_col=reward_col,
        step_col=step_col,
        condition_col=condition_col,
    )

    if condition_col not in per_trajectory.columns:
        raise KeyError(
            f"Column '{condition_col}' is required to summarize by condition"
        )

    window_column = f"mean_{reward_col}_last_{window_size}_steps"

    return (
        per_trajectory.groupby(condition_col, as_index=False)
        .agg(
            mean_reward_last_window=(window_column, "mean"),
            std_reward_last_window=(window_column, "std"),
            n_trajectories=(window_column, "count"),
        )
        .sort_values(condition_col)
    )


def plot_condition_comparison(
    summary_df: pd.DataFrame,
    condition_col: str = "condition",
    step_col: str = "Step",
    figsize: tuple[float, float] = (12, 12),
):
    """Plot 3 subplots across conditions: cumulative reward, mean reward, and max-state probabilities."""
    conditions = sorted(summary_df[condition_col].unique())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    probability_metrics = [
        ("global_max", "global", "-"),
        ("local_max", "local", "--"),
        ("no_max", "no max", ":"),
    ]

    for index, condition_name in enumerate(conditions):
        condition_data = summary_df[summary_df[condition_col] == condition_name]
        color = cmap(index % 10)

        axes[0].plot(
            condition_data[step_col],
            condition_data["cumulative_reward"],
            label=condition_name,
            color=color,
        )
        axes[1].plot(
            condition_data[step_col],
            condition_data["reward"],
            label=condition_name,
            color=color,
        )

        for metric_name, metric_label, line_style in probability_metrics:
            axes[2].plot(
                condition_data[step_col],
                condition_data[metric_name],
                label=f"{condition_name} ({metric_label})",
                color=color,
                linestyle=line_style,
            )

    axes[0].set_title("Cumulative reward over steps")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cumulative reward")
    # axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_title("Mean reward over steps")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean reward")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].set_title("Probability of global/local/no max")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Probability")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.show()
    return fig, axes


def summarize_batch_metrics(
    batch_df: pd.DataFrame,
    metric_columns: list[str],
    group_columns: tuple[str, ...] = ("beta", "length_scale", "tau"),
) -> pd.DataFrame:
    """Aggregate batch results into one value per parameter tuple."""
    aggregation = {metric_name: "mean" for metric_name in metric_columns}
    return batch_df.groupby(list(group_columns), as_index=False).agg(aggregation)


def plot_metric_heatmaps_by_lambda(
    summary_df: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    annot: bool = False,
    fmt: str = ".2f",
    cmap: str = "viridis",
    ncols: int = 2,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    mark_max: bool = False,
):
    """Plot one heatmap per lambda panel for a given metric."""
    lambda_values = np.sort(summary_df["length_scale"].unique())
    n_panels = len(lambda_values)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for panel_idx, lambda_value in enumerate(lambda_values):
        ax = axes_flat[panel_idx]
        row_idx = panel_idx // ncols
        col_idx = panel_idx % ncols

        lambda_data = summary_df[summary_df["length_scale"] == lambda_value]
        pivot_table = lambda_data.pivot(index="tau", columns="beta", values=metric_column)

        sns.heatmap(
            pivot_table,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            ax=ax,
            cbar_kws={"label": cbar_label if cbar_label is not None else metric_label},
        )

        if mark_max:
            pivot_values = pivot_table.to_numpy(dtype=float)
            has_finite_values = np.isfinite(pivot_values).any()
            if has_finite_values:
                max_flat_index = np.nanargmax(pivot_values)
                max_row, max_col = np.unravel_index(max_flat_index, pivot_values.shape)
                ax.scatter(
                    max_col + 0.5,
                    max_row + 0.5,
                    marker="*",
                    s=260,
                    c="crimson",
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=10,
                )

        ax.set_title(rf"{metric_label} ($\lambda$ = {float(lambda_value):.3f})")
        # ax.set_xticklabels(
        #     [f"{float(beta):.3f}" for beta in pivot_table.columns],
        #     rotation=45,
        #     ha="right",
        # )
        # ax.set_yticklabels(
        #     [f"{float(tau):.3f}" for tau in pivot_table.index],
        #     rotation=0,
        # )

        show_x = row_idx == (nrows - 1)
        show_y = col_idx == 0
        ax.set_xlabel(r"$\beta$") # if show_x else "")
        ax.set_ylabel(r"$\tau$") # if show_y else "")

    for panel_idx in range(n_panels, len(axes_flat)):
        axes_flat[panel_idx].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_metric_suite_by_lambda(
    summary_df: pd.DataFrame,
    metric_specs: list[dict],
    cmap: str = "viridis",
    ncols: int = 2,
    mark_max: bool = True,
):
    """Plot a full heatmap suite (one figure per metric spec)."""
    figures = []
    for metric_spec in metric_specs:
        fig, axes = plot_metric_heatmaps_by_lambda(
            summary_df=summary_df,
            metric_column=metric_spec["column"],
            metric_label=metric_spec["label"],
            fmt=metric_spec.get("fmt", ".2f"),
            cmap=cmap,
            ncols=ncols,
            vmin=metric_spec.get("vmin"),
            vmax=metric_spec.get("vmax"),
            cbar_label=metric_spec.get("cbar_label"),
            mark_max=mark_max,
        )
        figures.append((fig, axes))
    return figures
