from collections.abc import Mapping

import matplotlib.pyplot as plt
import mesa
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import sample_parameters_from_csv


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


def run_sampled_batches(
    *,
    model_cls,
    base_parameters: dict,
    csv_path: str,
    param_columns: dict[str, str],
    n_runs: int,
    max_steps: int,
    data_collection_period: int = 1,
    number_processes: int | None = None,
    display_progress: bool = True,
    rng: np.random.Generator | None = None,
    condition_name: str = "sampled",
    param_scaling: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Run batch simulations with parameters sampled from a CSV file.

    For each run, sample a new parameter set from the CSV. Parameters not specified
    in param_columns are used as fixed values from base_parameters.

    Parameters
    ----------
    model_cls : class
        The model class to instantiate (e.g., SocialGPModel).
    base_parameters : dict
        Base parameters for the model. Parameters to be sampled should be omitted
        or set to placeholder values (they will be overridden).
    csv_path : str
        Path to the CSV file containing parameter distributions.
    param_columns : dict[str, str]
        Mapping from model parameter names to CSV column names.
        Example: {'length_scale': 'lambda_0', 'tau': 'tau_0', 'beta': 'beta_0'}
    n_runs : int
        Number of simulation runs.
    max_steps : int
        Maximum number of steps per run.
    data_collection_period : int, optional
        How often to collect data. Default is 1 (every step).
    number_processes : int | None, optional
        Number of processes for parallel execution. Default is None (sequential).
    display_progress : bool, optional
        Whether to display progress. Default is True.
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().
    condition_name : str, optional
        Name for the condition in the output dataframe. Default is "sampled".
    param_scaling : dict[str, tuple[float, float]] | None, optional
        Scaling to apply to specific parameters after sampling.
        Keys are parameter names (must be in param_columns).
        Values are tuples of (source_lambda, target_lambda) - the sampled value
        will be scaled as: scaled_value = sample * target_lambda / source_lambda.
        Example: {'length_scale': (1.5, 4.5)} scales length_scale from an
        environment with lambda=1.5 to lambda=4.5.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with all simulation results, including a 'condition' column.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample parameter sets for each run
    sampled_params = sample_parameters_from_csv(
        csv_path=csv_path,
        n_samples=n_runs,
        param_columns=param_columns,
        rng=rng,
        param_scaling=param_scaling,
    )

    frames = []

    for run_idx, sampled in enumerate(sampled_params):
        # Create parameter dict for this run: start with base, override with sampled
        run_parameters = dict(base_parameters)
        run_parameters.update(sampled)

        # Handle reporter formatting for mesa.batch_run
        for reporter_key in ("model_reporters_to_collect", "agent_reporters_to_collect"):
            reporter_value = run_parameters.get(reporter_key)
            if (
                isinstance(reporter_value, list)
                and (len(reporter_value) == 0 or all(isinstance(item, str) for item in reporter_value))
            ):
                run_parameters[reporter_key] = [reporter_value]

        # Use a unique seed for each run to ensure reproducibility
        run_seed = int(rng.integers(0, 2**31))

        results = mesa.batch_run(
            model_cls,
            parameters=run_parameters,
            max_steps=max_steps,
            data_collection_period=data_collection_period,
            display_progress=False,  # Disable per-run progress to reduce output
            number_processes=number_processes,
            rng=[run_seed],
        )

        run_df = pd.DataFrame(results)
        run_df["condition"] = condition_name
        run_df["sample_idx"] = run_idx
        frames.append(run_df)

        if display_progress:
            print(f"Completed run {run_idx + 1}/{n_runs}")

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


def _default_parameter_label(parameter_name: str) -> str:
    """Return a readable axis/panel label for known sweep parameter names."""
    if parameter_name == "beta":
        return r"$\beta$"
    if parameter_name == "tau":
        return r"$\tau$"
    if parameter_name == "length_scale":
        return r"$\lambda$"
    return parameter_name


def _format_tick_value(value, n=2) -> str:
    """Format numeric tick labels with at most two decimal places."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    return f"{numeric_value:.{n}f}".rstrip("0").rstrip(".")


def _adaptive_tick_indices(n_values: int, max_ticks: int | None) -> np.ndarray:
    """Return evenly spaced tick indices with optional upper bound on label count."""
    if n_values <= 0:
        return np.array([], dtype=int)
    if max_ticks is None or max_ticks <= 0 or max_ticks >= n_values:
        return np.arange(n_values, dtype=int)

    step = int(np.ceil(n_values / max_ticks))
    indices = np.arange(0, n_values, step, dtype=int)
    if indices[-1] != n_values - 1:
        indices = np.append(indices, n_values - 1)
    return indices


def plot_metric_heatmaps(
    summary_df: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    *,
    panel_column: str = "length_scale",
    x_column: str = "beta",
    y_column: str = "tau",
    panel_label: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    cmap: str = "viridis",
    ncols: int = 2,
    subplot_size: tuple[float, float] = (9.5, 4.5),
    max_x_ticks: int | None = 12,
    max_y_ticks: int | None = 12,
    tick_decimals: int = 1,
    y_descending: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    mark_max: bool = False,
):
    """Plot one heatmap per panel value for a given metric and parameter mapping."""
    required_columns = [panel_column, x_column, y_column, metric_column]
    missing_columns = [column for column in required_columns if column not in summary_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for heatmap plotting: {missing_columns}")

    panel_values = np.sort(summary_df[panel_column].unique())
    n_panels = len(panel_values)
    nrows = int(np.ceil(n_panels / ncols))

    resolved_panel_label = panel_label if panel_label is not None else _default_parameter_label(panel_column)
    resolved_x_label = x_label if x_label is not None else _default_parameter_label(x_column)
    resolved_y_label = y_label if y_label is not None else _default_parameter_label(y_column)

    fig_width, fig_height = subplot_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width * ncols, fig_height * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for panel_idx, panel_value in enumerate(panel_values):
        ax = axes_flat[panel_idx]

        panel_data = summary_df[summary_df[panel_column] == panel_value]
        pivot_table = panel_data.pivot(index=y_column, columns=x_column, values=metric_column)
        pivot_table = pivot_table.sort_index(ascending=not y_descending)

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

        try:
            panel_text = f"{float(panel_value):.3f}"
        except (TypeError, ValueError):
            panel_text = str(panel_value)

        ax.set_title(f"{metric_label} ({resolved_panel_label} = {panel_text})")
        ax.set_xlabel(resolved_x_label)
        ax.set_ylabel(resolved_y_label)

        x_tick_indices = _adaptive_tick_indices(len(pivot_table.columns), max_x_ticks)
        y_tick_indices = _adaptive_tick_indices(len(pivot_table.index), max_y_ticks)

        ax.set_xticks(x_tick_indices + 0.5)
        ax.set_yticks(y_tick_indices + 0.5)

        ax.set_xticklabels(
            [_format_tick_value(pivot_table.columns[idx], n=tick_decimals) for idx in x_tick_indices],
            rotation=0,
        )
        ax.set_yticklabels(
            [_format_tick_value(pivot_table.index[idx], n=tick_decimals) for idx in y_tick_indices],
            rotation=0,
        )

    for panel_idx in range(n_panels, len(axes_flat)):
        axes_flat[panel_idx].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_metric_heatmaps_by_lambda(
    summary_df: pd.DataFrame,
    metric_column: str,
    metric_label: str,
    annot: bool = False,
    fmt: str = ".2f",
    cmap: str = "viridis",
    ncols: int = 2,
    subplot_size: tuple[float, float] = (9.5, 4.5),
    max_x_ticks: int | None = 12,
    max_y_ticks: int | None = 12,
    tick_decimals: int = 1,
    y_descending: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    mark_max: bool = False,
):
    """Backward-compatible wrapper: beta x tau panels by length scale."""
    return plot_metric_heatmaps(
        summary_df=summary_df,
        metric_column=metric_column,
        metric_label=metric_label,
        panel_column="length_scale",
        x_column="beta",
        y_column="tau",
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ncols=ncols,
        subplot_size=subplot_size,
        max_x_ticks=max_x_ticks,
        max_y_ticks=max_y_ticks,
        tick_decimals=tick_decimals,
        y_descending=y_descending,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
        mark_max=mark_max,
    )


def plot_metric_suite_by_lambda(
    summary_df: pd.DataFrame,
    metric_specs: list[dict],
    cmap: str = "viridis",
    ncols: int = 2,
    subplot_size: tuple[float, float] = (9.5, 4.5),
    max_x_ticks: int | None = 12,
    max_y_ticks: int | None = 12,
    tick_decimals: int = 1,
    y_descending: bool = False,
    mark_max: bool = True,
    *,
    panel_column: str = "length_scale",
    x_column: str = "beta",
    y_column: str = "tau",
    panel_label: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
):
    """Plot a full heatmap suite (one figure per metric spec) with configurable axes/panels."""
    figures = []
    for metric_spec in metric_specs:
        fig, axes = plot_metric_heatmaps(
            summary_df=summary_df,
            metric_column=metric_spec["column"],
            metric_label=metric_spec["label"],
            panel_column=panel_column,
            x_column=x_column,
            y_column=y_column,
            panel_label=panel_label,
            x_label=x_label,
            y_label=y_label,
            fmt=metric_spec.get("fmt", ".2f"),
            cmap=cmap,
            ncols=ncols,
            subplot_size=subplot_size,
            max_x_ticks=max_x_ticks,
            max_y_ticks=max_y_ticks,
            tick_decimals=tick_decimals,
            y_descending=y_descending,
            vmin=metric_spec.get("vmin"),
            vmax=metric_spec.get("vmax"),
            cbar_label=metric_spec.get("cbar_label"),
            mark_max=mark_max,
        )
        figures.append((fig, axes))
    return figures
