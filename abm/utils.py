import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_reward_grid(grid, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.axis('off')
    plt.show()

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
    Plot the trajectory of the most common (x, y) choice over time on top of a reward heatmap.

    Parameters
    ----------
    df_batch : pandas.DataFrame
        DataFrame containing a column of coordinate tuples/lists (x, y).
    reward : np.ndarray
        2D reward matrix to display as a heatmap.
    choice_col : str, default "most_common_choice"
        Name of the DataFrame column containing (x, y) choices.
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

    x_coords = [choice[0] for choice in most_common_choices]
    y_coords = [choice[1] for choice in most_common_choices]

    # Create figure/axes
    fig, ax = plt.subplots(figsize=figsize)

    # 1) Background: reward heatmap
    ax.imshow(reward, cmap=heatmap_cmap, origin="lower")

    # 2) Build line segments from consecutive points
    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
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
    ax.scatter(x_coords[0], y_coords[0], color="red", label="Start", zorder=5)
    ax.scatter(x_coords[-1], y_coords[-1], color="blue", label="End", zorder=5)

    # Labels and style
    ax.set_title(title)
    ax.axis("off")
    ax.legend()

    # Colorbar for step index
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Step")

    plt.show()
    return fig, ax