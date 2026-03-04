import numpy as np
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
):
    if heatmap_col not in df.columns:
        raise KeyError(f"Column '{heatmap_col}' not found in dataframe.")
    if choice_col not in df.columns:
        raise KeyError(f"Column '{choice_col}' not found in dataframe.")

    heatmaps = df[heatmap_col].to_numpy()
    if len(heatmaps) == 0:
        raise ValueError("Dataframe has no rows to animate.")
    if not (0 <= start_idx < len(heatmaps)):
        raise ValueError(f"start_idx must be in [0, {len(heatmaps)-1}], got {start_idx}.")

    fig, ax = plt.subplots(figsize=figsize)
    vmin = min(h.min() for h in heatmaps)
    vmax = max(h.max() for h in heatmaps)

    im = ax.imshow(
        heatmaps[start_idx], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax
    )
    row0, col0 = df.iloc[0][choice_col]
    ax.scatter(col0, row0, color=marker_color, s=marker_size, edgecolor=edgecolor)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame):
        ax.clear()
        im = ax.imshow(
            heatmaps[frame], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax
        )
        row, col = df.iloc[frame][choice_col]
        ax.scatter(col, row, color=marker_color, s=marker_size, edgecolor=edgecolor)
        ax.set_title(f'{title_prefix} {frame}')
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