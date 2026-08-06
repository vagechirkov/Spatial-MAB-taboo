import numpy as np
import matplotlib.pyplot as plt

def _min_max(arr, min=None, max=None):
    """Normalize array to [0, 1]."""
    if min is None:
        min = arr.min()
    if max is None:
        max = arr.max()
    return (arr - min) / (max - min)

def _mexican_hat_DoG(
    n, 
    sigma_outer, 
    sigma_inner, 
    center=None
):
    """
    Difference of Gaussians (Mexican Hat) landscape.
    Parameters
    ----------
    n : int
        Size of the square grid
    sigma_outer : float
        Standard deviation of the outer Gaussian
    sigma_inner : float
        Standard deviation of the inner Gaussian
    center : tuple or None
        Center of the Mexican Hat
    Returns
    -------
    mh : ndarray
    """
    if sigma_outer <= sigma_inner:
        raise ValueError("sigma_outer must be > sigma_inner")
    if center is None:
        center = ((n - 1) / 2.0, (n - 1) / 2.0)

    rows, cols = np.indices((n, n))
    r2 = (rows - center[0])**2 + (cols - center[1])**2
    inner = np.exp(-r2 / (2 * sigma_inner**2))
    outer = np.exp(-r2 / (2 * sigma_outer**2))
    mh = inner - outer * (sigma_inner/sigma_outer)

    return mh

"""
Plotting and visualization functions
"""
def plot_MAB(grid, kernel, title=''):
        kernel_visualization = covariance_heatmap(kernel)

        plt.rcParams['mathtext.fontset'] = 'stix'

        fig, ax  = plt.subplots(2, 1, figsize=(4, 8))
        fig.tight_layout(pad=1)

        ax[0].imshow(kernel_visualization, cmap='viridis', interpolation="nearest",)
        ax[0].axis('off')
        ax[0].set_title(f'{title[0]}',
                fontsize=25,
        )
                        
        ax[1].imshow(grid, cmap='viridis', origin='lower')
        ax[1].axis('off')
        ax[1].set_title(f'{title[1]}',
                fontsize=25,
        )
        plt.show()

def covariance_heatmap(K, n=None, reference="center"):
    """
    Convert a covariance matrix for a flattened square grid into a natural
    2D covariance heatmap.

    Parameters
    ----------
    K : ndarray, shape (n*n, n*n)
        Covariance matrix. Grid points must have been flattened in row-major
        order, as with array.ravel() or reshape(-1).

    n : int, optional
        Grid width and height. Inferred from K when omitted.

    reference : {"center", "center_mean"} or tuple[int, int], default="center"
        Reference location:

        - "center": use grid point (n // 2, n // 2)
        - "center_mean": for even n, average the four central covariance maps
        - (row, col): use a specified grid point

    Returns
    -------
    heatmap : ndarray, shape (n, n)
        Covariance between the reference point and every grid location.
    """
    K = np.asarray(K)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2D matrix.")

    n_points = K.shape[0]

    if n is None:
        n = int(np.sqrt(n_points))
        if n * n != n_points:
            raise ValueError(
                "K does not correspond to a square grid; provide the grid shape."
            )
    elif n * n != n_points:
        raise ValueError(
            f"Expected K to have shape ({n*n}, {n*n}) for n={n}, "
            f"but got {K.shape}."
        )

    if reference == "center":
        row = col = n // 2
        reference_idx = row * n + col
        return K[reference_idx].reshape(n, n)

    if reference == "center_mean":
        if n % 2 == 1:
            center = n // 2
            reference_idx = center * n + center
            return K[reference_idx].reshape(n, n)

        centers = [
            (n // 2 - 1, n // 2 - 1),
            (n // 2 - 1, n // 2),
            (n // 2, n // 2 - 1),
            (n // 2, n // 2),
        ]

        center_indices = [row * n + col for row, col in centers]

        return K[center_indices].mean(axis=0).reshape(n, n)

    if isinstance(reference, tuple) and len(reference) == 2:
        row, col = reference

        if not (0 <= row < n and 0 <= col < n):
            raise ValueError(
                f"Reference point {(row, col)} is outside the {n}×{n} grid."
            )

        reference_idx = row * n + col
        return K[reference_idx].reshape(n, n)

    raise ValueError(
        "reference must be 'center', 'center_mean', or a (row, col) tuple."
    )