import numpy as np
from sklearn.gaussian_process.kernels import RBF

def _min_max(arr):
    """Normalize array to [0, 1]."""
    return (arr - arr.min()) / (arr.max() - arr.min())

def make_parent_and_children_cholesky(
    rng,
    grid_size=11,
    n_children=4,
    length_scale=2.0,
    corr_matrix=None,
):
    """
    Generate one parent map + n_children child maps with a specified
    spatial covariance (RBF kernel) and a task-level correlation structure.
    
    This is the core generator for correlated GP-based reward environments.
    """
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    Xstar = np.column_stack([x.ravel(), y.ravel()])
    Sigma = RBF(length_scale)(Xstar)
    # Add jitter for numerical stability
    LSigma = np.linalg.cholesky(Sigma + 1e-10 * np.eye(Sigma.shape[0]))
    M = Sigma.shape[0]

    if corr_matrix is not None:
        R = np.asarray(corr_matrix, dtype=float)
        n_total = R.shape[0]
        # In this mode, n_children is derived from R or expected to match
    else:
        n_total = n_children + 1
        # Default: full correlation of 0.6 between all tasks (parent and children)
        R = np.full((n_total, n_total), 0.6)
        np.fill_diagonal(R, 1.0)

    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 0:
        raise ValueError(f"Correlation matrix is not positive-definite. Min eigenvalue: {eigvals.min():.3g}")

    LR = np.linalg.cholesky(R)
    z = rng.standard_normal((n_total, M))
    Y = (LR @ z) @ LSigma.T

    parent = Y[0].reshape(grid_size, grid_size)
    children = [Y[i + 1].reshape(grid_size, grid_size) for i in range(n_total - 1)]
    
    return _min_max(parent), [_min_max(c) for c in children]

def build_corr_matrix_bare_bones(n_total=4):
    """Example of a simple correlation matrix."""
    R = np.full((n_total, n_total), 0.6)
    np.fill_diagonal(R, 1.0)
    return R

def check_correlations_matrix(parent, children, R_target, tol=0.1):
    """Utility to verify empirical correlations match target."""
    flats = [parent.ravel()] + [c.ravel() for c in children]
    C = np.corrcoef(flats)
    i, j = np.triu_indices(R_target.shape[0], k=1)
    diffs = np.abs(C[i, j] - R_target[i, j])
    return np.all(diffs <= tol)