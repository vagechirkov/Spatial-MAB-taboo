"""
Utility functions for generating reward landscapes (GP, Gabor, Mexican Hat, etc).

Naming conventions:
- All functions use 'mexican_hat' for DoG/Mexican Hat family
- Approach is specified in function name (e.g. mexican_hat_gaussian, mexican_hat_rbf, mexican_hat_sinusoid)
- No '_2d' suffix
"""

import numpy as np
from sklearn.gaussian_process.kernels import RBF

def _min_max(arr):
    """Normalize array to [0, 1]."""
    return (arr - arr.min()) / (arr.max() - arr.min())

def _gabor_filter(grid_size, frequency, theta=0.0, sigma=None, phase=0.0, center=None):
    """
    Generate a 2D Gabor filter pattern.
    Parameters
    ----------
    grid_size : int
        Size of the grid (grid_size x grid_size)
    frequency : float
        Spatial frequency of the sinusoidal component (cycles per grid)
    theta : float, default=0.0
        Orientation angle in radians (0 = horizontal stripes)
    sigma : float, optional
        Standard deviation of the Gaussian envelope. If None, uses frequency-based default.
    phase : float, default=0.0
        Phase offset of the sinusoidal component in radians
    center : tuple of float, optional
        Center of the Gabor filter as (row, col). If None, uses grid center.
    Returns
    -------
    gabor : ndarray
        2D array of shape (grid_size, grid_size) with Gabor filter pattern
    """
    if center is None:
        center = (grid_size / 2.0, grid_size / 2.0)
    if sigma is None:
        wavelength = grid_size / frequency
        sigma = wavelength / 4.0
    rows, cols = np.indices((grid_size, grid_size))
    Xc = cols - center[1]
    Yc = rows - center[0]
    X_rot = Xc * np.cos(theta) + Yc * np.sin(theta)
    Y_rot = -Xc * np.sin(theta) + Yc * np.cos(theta)
    gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
    freq_pixel = frequency / grid_size
    sinusoidal = np.cos(2 * np.pi * freq_pixel * X_rot + phase)
    gabor = gaussian * sinusoidal
    return gabor

def _mexican_hat_gaussian(grid_size, frequency=2.0, sigma_inner=None, sigma_outer=None, center=None, outer_factor=2.0):
    """
    Difference-of-Gaussians (Mexican Hat) pattern.
    Parameters
    ----------
    grid_size : int
    frequency : float, default=2.0
    sigma_inner : float, optional
    sigma_outer : float, optional
    center : tuple of float, optional
    outer_factor : float, default=2.0
    Returns
    -------
    mh : ndarray
        2D array of shape (grid_size, grid_size) with zero-mean Mexican Hat values.
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be > 0, got {frequency}")
    if outer_factor <= 0:
        raise ValueError(f"outer_factor must be > 0, got {outer_factor}")
    if center is None:
        center = ((grid_size - 1) / 2.0, (grid_size - 1) / 2.0)
    if sigma_inner is None:
        wavelength = grid_size / frequency
        sigma_inner = wavelength / 4.0
    if sigma_outer is None:
        sigma_outer = outer_factor * float(sigma_inner)
    rows, cols = np.indices((grid_size, grid_size))
    Xc = cols - center[1]
    Yc = rows - center[0]
    r2 = Xc**2 + Yc**2
    def _gauss(r2_arr, s):
        return np.exp(-r2_arr / (2.0 * (s**2)))
    g_inner = _gauss(r2, float(sigma_inner))
    g_outer = _gauss(r2, float(sigma_outer)) * sigma_inner/sigma_outer * 1.5
    mh = g_inner - g_outer
    mh = mh - mh.mean()
    return mh

def mexican_hat_rbf(grid_size, sigma_inner, sigma_outer, center=None):
    """
    Difference-of-RBF (Mexican Hat) landscape.
    Parameters
    ----------
    grid_size : int
    sigma_inner : float
    sigma_outer : float
    center : tuple or None
    Returns
    -------
    mh : ndarray
    """
    if sigma_outer <= sigma_inner:
        raise ValueError("sigma_outer must be > sigma_inner")
    if center is None:
        center = ((grid_size - 1) / 2.0, (grid_size - 1) / 2.0)
    rows, cols = np.indices((grid_size, grid_size))
    r2 = (rows - center[0])**2 + (cols - center[1])**2
    inner = np.exp(-r2 / (2 * sigma_inner**2))
    outer = np.exp(-r2 / (2 * sigma_outer**2))
    mh = inner - outer * (sigma_inner/sigma_outer)
    return mh

def mexican_hat_sinusoid(grid_size, sigma_inner, sigma_outer, center=None):
    """
    Sinusoidal Mexican Hat landscape.
    Parameters
    ----------
    grid_size : int
    sigma_inner : float
    sigma_outer : float
    center : tuple or None
    Returns
    -------
    mh : ndarray
    """
    if sigma_outer <= 0:
        raise ValueError("sigma_outer must be > 0")
    if sigma_inner <= 0:
        raise ValueError("sigma_inner must be > 0")
    if center is None:
        center = ((grid_size - 1) / 2.0, (grid_size - 1) / 2.0)
    rows, cols = np.indices((grid_size, grid_size))
    r = np.sqrt((rows - center[0])**2 + (cols - center[1])**2)
    inner_wavelength = 2.0 * sigma_inner
    inner_phase = 0.0
    inner = np.full_like(r, np.nan, dtype=float)
    inner_mask = r <= sigma_inner
    inner_arg = (2.0 * np.pi * r[inner_mask] / inner_wavelength) + inner_phase
    inner[inner_mask] =  (0.5 + np.cos(inner_arg))
    outer_wavelength = sigma_outer
    outer_phase = np.pi
    outer = np.full_like(r, np.nan, dtype=float)
    outer_ramp_mask = (r > sigma_inner) & (r <= sigma_outer + 1)
    outer_arg = (
        (2.0 * np.pi * (r[outer_ramp_mask] - sigma_inner) / outer_wavelength)
        + outer_phase
    )
    outer[outer_ramp_mask] = 0.25 * (-1.0 + np.cos(outer_arg))
    outer[r > sigma_outer + 1] = np.nan
    result = np.nan_to_num(inner, nan=0.0) + np.nan_to_num(outer, nan=0.0)
    return result

def _cholesky_grid(rng=None, grid_size=20, length_scale=4.0, jitter=1e-10, normalize=True):
    """
    Generate a single 2D Gaussian Process grid using an RBF kernel.
    """
    if rng is None:
        rng = np.random.default_rng()
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    X = np.column_stack([x.ravel(), y.ravel()])
    kernel = RBF(length_scale=length_scale)
    K = kernel(X)
    L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
    z = rng.standard_normal(K.shape[0])
    sample = L @ z
    grid = sample.reshape(grid_size, grid_size)
    if normalize:
        grid = _min_max(grid)
    return grid

def build_correlation_matrix(n_total=4, r=0.6):
    """
    Build a simple correlation matrix of size n_total x n_total with off-diagonal r.
    """
    R = np.full((n_total, n_total), r)
    np.fill_diagonal(R, 1.0)
    return R

def check_correlations(parent, children, target_correlation, tol=0.1):
    """
    Utility to verify empirical correlations between parent and children match target.
    """
    parent_flat = parent.ravel()
    correlations = []
    for child in children:
        child_flat = child.ravel()
        corr = np.corrcoef(parent_flat, child_flat)[0, 1]
        correlations.append(corr)
    diffs = np.abs(np.array(correlations) - target_correlation)
    return np.all(diffs <= tol), correlations

def check_correlations_matrix(parent, children, R_target, tol=0.1):
    """
    Utility to verify empirical correlations match target matrix.
    """
    flats = [parent.ravel()] + [c.ravel() for c in children]
    C = np.corrcoef(flats)
    i, j = np.triu_indices(R_target.shape[0], k=1)
    diffs = np.abs(C[i, j] - R_target[i, j])
    return np.all(diffs <= tol)
