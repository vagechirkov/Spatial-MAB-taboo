import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

def plot_reward_grid(grid, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.axis('off')
    plt.show()

def plot_reward_grids(reward_array, titles=None):
    if titles is None:
        titles = [f"Landscape {i + 1}" for i in range(reward_array.shape[0])]

    n_grids = reward_array.shape[0]
    fig, axes = plt.subplots(1, n_grids, figsize=(15, 4))

    for i, grid in enumerate(reward_array):
        axes[i].imshow(grid, cmap='viridis', origin='lower')
        axes[i].set_title(titles[i])

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    
    fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    return fig, axes

def plot_most_common_choice_trajectory(
    df_batch,
    reward,
    choice_col: str = "most_common_choice",
    figsize=(6, 6),
    heatmap_cmap: str = "viridis",
    trajectory_cmap: str = "cool",
    title: str = "Most Common Choice Trajectory",
):
    """
    Plot the trajectory of the most common (row, col) choice over time on top of a reward heatmap.

    Parameters
    ----------
    df_batch : pandas.DataFrame
        DataFrame containing a column of coordinate tuples/lists (x, y).
    reward : np.ndarray
        2D reward matrix to display as a heatmap.
    choice_col : str, default "most_common_choice"
        Name of the DataFrame column containing (row, col) choices.
    figsize : tuple, default (6, 6)
        Figure size.
    heatmap_cmap : str, default "viridis"
        Colormap for reward heatmap.
    trajectory_cmap : str, default "cool"
        Colormap for trajectory line (colored by step index).
    title : str, default "Most Common Choice Trajectory"
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Handles for further customization if needed.
    """
    # Extract coordinate sequence from DataFrame
    most_common_choices = df_batch[choice_col].tolist()
    if len(most_common_choices) < 2:
        raise ValueError("Need at least two points to plot a trajectory.")

    rows = [choice[0] for choice in most_common_choices]
    cols = [choice[1] for choice in most_common_choices]

    # Create figure/axes
    fig, ax = plt.subplots(figsize=figsize)

    # 1) Background: reward heatmap
    ax.imshow(reward, cmap=heatmap_cmap, origin="lower")

    # 2) Build line segments from consecutive points
    points = np.array([cols, rows]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 3) Color segments by time step
    lc = LineCollection(
        segments,
        cmap=trajectory_cmap,
        norm=plt.Normalize(0, len(most_common_choices) - 1),
    )
    lc.set_array(np.arange(len(most_common_choices) - 1))
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # Mark start and end points
    ax.scatter(cols[0], rows[0], color="red", label="Start", zorder=5)
    ax.scatter(cols[-1], rows[-1], color="blue", label="End", zorder=5)

    # Labels and style
    ax.set_title(title)
    ax.axis("off")
    ax.legend()

    # Colorbar for step index
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Step")

    plt.show()
    return fig, ax


def rbf(grid_size, length_scale, center):
    rows, cols = np.indices((grid_size, grid_size))
    dists = np.sqrt((rows - center[0])**2 + (cols - center[1])**2)
    return np.exp(-0.5 * (dists / length_scale)**2)


def animate_heatmap_trajectory(
    df,
    heatmap_col,
    choice_col='choice',
    figsize=(4, 4),
    cmap='viridis',
    origin='lower',
    marker_color='red',
    marker_size=100,
    edgecolor='white',
    start_idx=1,
    fps=4,
    save_path=None,
    repeat=False,
    title_prefix='Step',
    focal_agent=None,
    agent_id_col='AgentID',
    step_col='Step',
    match_cols=None,
    other_marker_color='cyan',
    other_marker_size=70,
    other_edgecolor='black',
    other_alpha=0.85,
):
    """
    Animate a focal agent's heatmap with choice markers over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with at least heatmap and choice columns.
    heatmap_col : str
        Column containing 2D arrays to render each frame.
    choice_col : str, default 'choice'
        Column containing (row, col) choices.
    focal_agent : optional
        Agent identifier to animate. If provided, choices of all other agents
        at each matching step are also plotted as dots.
    agent_id_col : str, default 'AgentID'
        Agent id column.
    step_col : str, default 'Step'
        Step column used to align other agents with the focal frame.
    match_cols : list[str] | None, default None
        Extra columns that must match between focal and other-agent rows
        (for example, RunId/iteration). If None, inferred as available
        columns among ['RunId', 'iteration'].
    """

    def _unpack_choice(choice, row_idx):
        if not hasattr(choice, '__len__') or len(choice) < 2:
            raise ValueError(
                f"Invalid choice at row {row_idx}: expected (row, col), got {choice!r}."
            )
        return choice[0], choice[1]

    if heatmap_col not in df.columns:
        raise KeyError(f"Column '{heatmap_col}' not found in dataframe.")
    if choice_col not in df.columns:
        raise KeyError(f"Column '{choice_col}' not found in dataframe.")

    if focal_agent is not None:
        if agent_id_col not in df.columns:
            raise KeyError(
                f"Column '{agent_id_col}' is required when focal_agent is provided."
            )
        if step_col not in df.columns:
            raise KeyError(
                f"Column '{step_col}' is required when focal_agent is provided."
            )
        focal_df = df[df[agent_id_col] == focal_agent].copy()
        if focal_df.empty:
            raise ValueError(
                f"No rows found for focal_agent={focal_agent!r} in column '{agent_id_col}'."
            )
    else:
        focal_df = df.copy()
        if (
            agent_id_col in focal_df.columns
            and focal_df[agent_id_col].nunique() > 1
        ):
            raise ValueError(
                "Dataframe contains multiple agents. "
                "Provide focal_agent to choose which agent heatmap to animate."
            )

    if step_col in focal_df.columns:
        focal_df = focal_df.sort_values(step_col).reset_index(drop=True)
    else:
        focal_df = focal_df.reset_index(drop=True)

    if match_cols is None:
        match_cols = [col for col in ('RunId', 'iteration') if col in focal_df.columns]
    else:
        missing_match_cols = [col for col in match_cols if col not in focal_df.columns]
        if missing_match_cols:
            raise KeyError(
                f"Columns {missing_match_cols} in match_cols were not found in dataframe."
            )

    heatmaps = focal_df[heatmap_col].to_numpy()
    if len(heatmaps) == 0:
        raise ValueError("Dataframe has no rows to animate.")
    if not (0 <= start_idx < len(heatmaps)):
        raise ValueError(f"start_idx must be in [0, {len(heatmaps)-1}], got {start_idx}.")

    other_choices_by_key = {}
    if focal_agent is not None:
        for row_idx, row in df.iterrows():
            if row[agent_id_col] == focal_agent:
                continue
            key = [row[step_col]]
            for col in match_cols:
                key.append(row[col])
            other_choices_by_key.setdefault(tuple(key), []).append(
                _unpack_choice(row[choice_col], row_idx)
            )

    def _frame_key(frame_idx):
        if focal_agent is None:
            return None
        row = focal_df.iloc[frame_idx]
        key = [row[step_col]]
        for col in match_cols:
            key.append(row[col])
        return tuple(key)

    fig, ax = plt.subplots(figsize=figsize)
    vmin = min(h.min() for h in heatmaps)
    vmax = max(h.max() for h in heatmaps)

    im = ax.imshow(
        heatmaps[start_idx], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax
    )
    row0, col0 = _unpack_choice(focal_df.iloc[start_idx][choice_col], start_idx)
    ax.scatter(
        col0,
        row0,
        color=marker_color,
        s=marker_size,
        edgecolor=edgecolor,
        label='Focal',
    )

    if focal_agent is not None:
        other_choices = other_choices_by_key.get(_frame_key(start_idx), [])
        if other_choices:
            other_rows, other_cols = zip(*other_choices)
            ax.scatter(
                other_cols,
                other_rows,
                color=other_marker_color,
                s=other_marker_size,
                edgecolor=other_edgecolor,
                alpha=other_alpha,
                label='Other agents',
            )
        ax.legend(loc='upper right', frameon=True)

    initial_step = (
        focal_df.iloc[start_idx][step_col]
        if step_col in focal_df.columns
        else start_idx
    )
    if focal_agent is None:
        ax.set_title(f'{title_prefix} {initial_step}')
    else:
        ax.set_title(f'{title_prefix} {initial_step} | focal {focal_agent}')

    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame):
        ax.clear()
        im = ax.imshow(
            heatmaps[frame], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax
        )
        row, col = _unpack_choice(focal_df.iloc[frame][choice_col], frame)
        ax.scatter(
            col,
            row,
            color=marker_color,
            s=marker_size,
            edgecolor=edgecolor,
            label='Focal',
        )

        if focal_agent is not None:
            other_choices = other_choices_by_key.get(_frame_key(frame), [])
            if other_choices:
                other_rows, other_cols = zip(*other_choices)
                ax.scatter(
                    other_cols,
                    other_rows,
                    color=other_marker_color,
                    s=other_marker_size,
                    edgecolor=other_edgecolor,
                    alpha=other_alpha,
                    label='Other agents',
                )
            ax.legend(loc='upper right', frameon=True)

        frame_step = (
            focal_df.iloc[frame][step_col]
            if step_col in focal_df.columns
            else frame
        )
        if focal_agent is None:
            ax.set_title(f'{title_prefix} {frame_step}')
        else:
            ax.set_title(f'{title_prefix} {frame_step} | focal {focal_agent}')
        ax.axis('off')
        return (im,)

    ani = FuncAnimation(
        fig,
        update,
        frames=range(start_idx, len(heatmaps)),
        repeat=repeat,
    )

    if save_path is not None:
        ani.save(save_path, writer='pillow', fps=fps)

    return ani, fig, ax


def sample_parameters_from_csv(
    csv_path: str,
    n_samples: int,
    param_columns: dict[str, str],
    rng: np.random.Generator | None = None,
    param_scaling: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """
    Sample parameter sets from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing parameter distributions.
    n_samples : int
        Number of parameter sets to sample.
    param_columns : dict[str, str]
        Mapping from model parameter names to CSV column names.
        Example: {'length_scale': 'lambda_0', 'tau': 'tau_0', 'beta': 'beta_0'}
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().
    param_scaling : dict[str, tuple[float, float]] | None, optional
        Scaling to apply to specific parameters after sampling.
        Keys are parameter names (must be in param_columns).
        Values are tuples of (source_lambda, target_lambda) - the sampled value
        will be scaled as: scaled_value = sample * target_lambda / source_lambda.
        Example: {'length_scale': (1.5, 4.5)} scales length_scale from an
        environment with lambda=1.5 to lambda=4.5.

    Returns
    -------
    list[dict]
        List of parameter dictionaries, each containing sampled (and optionally
        scaled) values for the specified parameters.

    Notes
    -----
    - For batches with multiple runs, each run will sample a new parameter set.
    - Parameters not specified in param_columns are not sampled (use fixed values
      in the model configuration).
    - Sampling is done with replacement if n_samples > number of rows in CSV.
    """
    if rng is None:
        rng = np.random.default_rng()

    df = pd.read_csv(csv_path)

    # Validate that all specified columns exist
    missing_cols = set(param_columns.values()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV columns not found: {missing_cols}")

    # Validate param_scaling keys are in param_columns
    if param_scaling is not None:
        invalid_keys = set(param_scaling.keys()) - set(param_columns.keys())
        if invalid_keys:
            raise ValueError(
                f"Parameters in param_scaling not in param_columns: {invalid_keys}"
            )

    # Sample rows from the dataframe
    sampled_indices = rng.integers(0, len(df), size=n_samples)
    sampled_rows = df.iloc[sampled_indices]

    # Build parameter dictionaries
    param_sets = []
    for _, row in sampled_rows.iterrows():
        param_set = {}
        for model_param, csv_col in param_columns.items():
            value = row[csv_col]
            # Apply scaling if specified
            if param_scaling is not None and model_param in param_scaling:
                source_lambda, target_lambda = param_scaling[model_param]
                value = value * target_lambda / source_lambda
            param_set[model_param] = value
        param_sets.append(param_set)

    return param_sets


def sample_parameters_from_distributions(
    n_samples: int,
    param_distributions: dict[str, dict],
    rng: np.random.Generator | None = None,
    param_scaling: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """
    Sample parameter sets from specified distributions.

    Supports log-normal distributions (for length_scale, beta, tau, alpha) and
    uniform distributions (for any parameter).

    Parameters
    ----------
    n_samples : int
        Number of parameter sets to sample.
    param_distributions : dict[str, dict]
        Mapping from model parameter names to distribution specifications.
        Each specification is a dict with 'distribution' key and distribution-specific
        parameters:

        For log-normal distribution:
            {'distribution': 'lognormal', 'mu': float, 'sigma': float}
            - mu: mean of the underlying normal distribution (log scale)
            - sigma: standard deviation of the underlying normal distribution

        For uniform distribution:
            {'distribution': 'uniform', 'a': float, 'b': float}
            - a: lower bound
            - b: upper bound

        Example:
            {
                'length_scale': {'distribution': 'lognormal', 'mu': 0.5, 'sigma': 0.3},
                'beta': {'distribution': 'lognormal', 'mu': -1.0, 'sigma': 0.5},
                'tau': {'distribution': 'uniform', 'a': 0.01, 'b': 0.1},
                'alpha': {'distribution': 'lognormal', 'mu': -0.5, 'sigma': 0.4},
            }

    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().
    param_scaling : dict[str, tuple[float, float]] | None, optional
        Scaling to apply to specific parameters after sampling.
        Keys are parameter names (must be in param_distributions).
        Values are tuples of (source_lambda, target_lambda) - the sampled value
        will be scaled as: scaled_value = sample * target_lambda / source_lambda.
        Example: {'length_scale': (1.5, 4.5)} scales length_scale from an
        environment with lambda=1.5 to lambda=4.5.

    Returns
    -------
    list[dict]
        List of parameter dictionaries, each containing sampled values for
        the specified parameters.

    Notes
    -----
    - Parameters not specified in param_distributions will not be sampled
      (use fixed values in the model configuration).
    - Log-normal is useful for positive parameters that span orders of magnitude.
    - Uniform is useful for bounded parameters where any value in range is equally likely.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    if not param_distributions:
        return [{}] * n_samples

    # Validate param_scaling keys are in param_distributions
    if param_scaling is not None:
        invalid_keys = set(param_scaling.keys()) - set(param_distributions.keys())
        if invalid_keys:
            raise ValueError(
                f"Parameters in param_scaling not in param_distributions: {invalid_keys}"
            )

    # Validate distribution specifications
    supported_distributions = {"lognormal", "uniform"}
    for param_name, dist_spec in param_distributions.items():
        if not isinstance(dist_spec, dict):
            raise TypeError(
                f"Parameter '{param_name}' must have a dict specification, "
                f"got {type(dist_spec).__name__}"
            )
        distribution_type = dist_spec.get("distribution")
        if distribution_type not in supported_distributions:
            raise ValueError(
                f"Parameter '{param_name}': unknown distribution '{distribution_type}'. "
                f"Supported: {supported_distributions}"
            )

    # Sample parameters
    param_sets = []
    for _ in range(n_samples):
        param_set = {}
        for param_name, dist_spec in param_distributions.items():
            distribution_type = dist_spec["distribution"]

            if distribution_type == "lognormal":
                mu = dist_spec["mu"]
                sigma = dist_spec["sigma"]
                value = rng.lognormal(mean=mu, sigma=sigma)

            elif distribution_type == "uniform":
                a = dist_spec["a"]
                b = dist_spec["b"]
                value = rng.uniform(low=a, high=b)

            # Apply scaling if specified
            if param_scaling is not None and param_name in param_scaling:
                source_lambda, target_lambda = param_scaling[param_name]
                value = value * target_lambda / source_lambda

            param_set[param_name] = value

        param_sets.append(param_set)

    return param_sets