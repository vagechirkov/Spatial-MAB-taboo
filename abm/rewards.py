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

def _gabor_filter_2d(
    grid_size,
    frequency,
    theta=0.0,
    sigma=None,
    phase=0.0,
    center=None,
):
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
        Center of the Gabor filter. If None, uses grid center.
    
    Returns
    -------
    gabor : ndarray
        2D array of shape (grid_size, grid_size) with Gabor filter pattern
    """
    if center is None:
        center = (grid_size / 2.0, grid_size / 2.0)
    
    if sigma is None:
        # Default sigma based on frequency: roughly 1/4 of wavelength
        wavelength = grid_size / frequency
        sigma = wavelength / 4.0
    
    # Create coordinate grids
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Center coordinates
    Xc = X - center[0]
    Yc = Y - center[1]
    
    # Rotate coordinates according to orientation
    X_rot = Xc * np.cos(theta) + Yc * np.sin(theta)
    Y_rot = -Xc * np.sin(theta) + Yc * np.cos(theta)
    
    # Gaussian envelope
    gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
    
    # Sinusoidal component
    # Convert frequency from cycles per grid to cycles per pixel
    freq_pixel = frequency / grid_size
    sinusoidal = np.cos(2 * np.pi * freq_pixel * X_rot + phase)
    
    # Gabor filter is the product
    gabor = gaussian * sinusoidal
    
    return gabor

def make_parent_and_children_gabor(
    rng,
    grid_size=11,
    n_children=4,
    frequency=2.0,
    theta_parent=0.0,
    sigma=None,
    phase_parent=None,
    correlation=0.6,
    theta_children=None,
    phase_children=None,
    center=None,
):
    """
    Generate one parent map + n_children child maps using Gabor filters.
    
    The parent is generated with a Gabor filter. Each child is generated to have
    a specified correlation r with the parent while maintaining the same frequency.
    Children are created by mixing the parent pattern with independent Gabor patterns.
    
    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator
    grid_size : int, default=11
        Size of the grid (grid_size x grid_size)
    n_children : int, default=4
        Number of child maps to generate
    frequency : float, default=2.0
        Spatial frequency of the Gabor filter (cycles per grid)
    theta_parent : float, default=0.0
        Orientation angle for parent in radians (0 = horizontal stripes)
    sigma : float, optional
        Standard deviation of the Gaussian envelope. If None, uses frequency-based default.
    phase_parent : float, optional
        Phase offset for parent in radians. If None, randomly sampled.
    correlation : float, default=0.6
        Target correlation coefficient between parent and each child
    theta_children : array-like of float, optional
        Orientation angles for children in radians. If None, randomly sampled.
    phase_children : array-like of float, optional
        Phase offsets for children in radians. If None, randomly sampled.
    center : tuple of float, optional
        Center of the Gabor filter. If None, uses grid center.
    
    Returns
    -------
    parent : ndarray
        2D array of shape (grid_size, grid_size) with normalized parent Gabor pattern
    children : list of ndarray
        List of n_children 2D arrays, each normalized to [0, 1]
    """
    if not (-1 <= correlation <= 1):
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")
    
    # Generate parent Gabor pattern
    if phase_parent is None:
        phase_parent = rng.uniform(0, 2 * np.pi)
    
    parent = _gabor_filter_2d(
        grid_size=grid_size,
        frequency=frequency,
        theta=theta_parent,
        sigma=sigma,
        phase=phase_parent,
        center=center,
    )
    
    # Normalize parent to zero mean and unit variance for correlation mixing
    parent_flat = parent.ravel()
    parent_mean = parent_flat.mean()
    parent_std = parent_flat.std()
    if parent_std < 1e-10:
        parent_std = 1.0
    parent_normalized = (parent_flat - parent_mean) / parent_std
    
    # Generate children
    children = []
    
    if theta_children is None:
        theta_children = [rng.uniform(0, 2 * np.pi) for _ in range(n_children)]
    elif len(theta_children) != n_children:
        raise ValueError(f"theta_children must have length {n_children}, got {len(theta_children)}")
    
    if phase_children is None:
        phase_children = [rng.uniform(0, 2 * np.pi) for _ in range(n_children)]
    elif len(phase_children) != n_children:
        raise ValueError(f"phase_children must have length {n_children}, got {len(phase_children)}")
    
    for i in range(n_children):
        # Generate independent Gabor pattern with same frequency
        child_independent = _gabor_filter_2d(
            grid_size=grid_size,
            frequency=frequency,
            theta=theta_children[i],
            sigma=sigma,
            phase=phase_children[i],
            center=center,
        )
        
        # Normalize independent pattern
        child_indep_flat = child_independent.ravel()
        child_indep_mean = child_indep_flat.mean()
        child_indep_std = child_indep_flat.std()
        if child_indep_std < 1e-10:
            child_indep_std = 1.0
        child_indep_normalized = (child_indep_flat - child_indep_mean) / child_indep_std
        
        # Mix parent and independent pattern to achieve target correlation
        # child = r * parent + sqrt(1 - r^2) * independent
        # This ensures correlation(child, parent) = r
        r = correlation
        child_mixed = r * parent_normalized + np.sqrt(1 - r**2) * child_indep_normalized
        
        # Reshape and normalize to [0, 1]
        child_grid = child_mixed.reshape(grid_size, grid_size)
        children.append(_min_max(child_grid))
    
    # Normalize parent to [0, 1] for consistency
    parent_normalized_grid = _min_max(parent)
    
    return parent_normalized_grid, children

def check_correlations(parent, children, target_correlation, tol=0.1):
    """
    Utility to verify empirical correlations between parent and children match target.
    
    Parameters
    ----------
    parent : ndarray
        2D parent grid
    children : list of ndarray
        List of 2D child grids
    target_correlation : float
        Expected correlation between parent and each child
    tol : float, default=0.1
        Tolerance for correlation difference
    
    Returns
    -------
    bool
        True if all correlations are within tolerance
    """
    parent_flat = parent.ravel()
    correlations = []
    for child in children:
        child_flat = child.ravel()
        corr = np.corrcoef(parent_flat, child_flat)[0, 1]
        correlations.append(corr)
    
    diffs = np.abs(np.array(correlations) - target_correlation)
    return np.all(diffs <= tol), correlations


def _dog_filter_2d(
    grid_size,
    frequency: float = 2.0,
    sigma_inner: float | None = None,
    sigma_outer: float | None = None,
    center=None,
    outer_factor: float = 2.0,
):
    """
    Difference-of-Gaussians (DoG) pattern, a Mexican-hat–like filter.

    Parameters
    ----------
    grid_size : int
        Size of the grid (grid_size x grid_size)
    frequency : float, default=2.0
        Controls base spatial scale (cycles per grid). Higher -> smaller sigmas.
    sigma_inner : float, optional
        Inner Gaussian width. If None, derived from (grid_size, frequency).
    sigma_outer : float, optional
        Outer Gaussian width. If None, set to outer_factor * sigma_inner.
    center : tuple of float, optional
        Center (x, y). If None, uses grid center.

    Returns
    -------
    dog : ndarray
        2D array of shape (grid_size, grid_size) with zero-mean DoG values.
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be > 0, got {frequency}")
    if outer_factor <= 0:
        raise ValueError(f"outer_factor must be > 0, got {outer_factor}")
    if center is None:
        center = ((grid_size - 1) / 2.0, (grid_size - 1) / 2.0)

    # If sigmas are not provided, derive them from frequency.
    if sigma_inner is None:
        # Match the Gabor convention: wavelength = grid_size / frequency; sigma ≈ wavelength / 4
        wavelength = grid_size / frequency
        sigma_inner = wavelength / 4.0
    if sigma_outer is None:
        sigma_outer = outer_factor * float(sigma_inner)

    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    Xc = X - center[0]
    Yc = Y - center[1]
    r2 = Xc**2 + Yc**2

    def _gauss(r2_arr, s):
        return np.exp(-r2_arr / (2.0 * (s**2)))

    g_inner = _gauss(r2, float(sigma_inner))
    g_outer = _gauss(r2, float(sigma_outer))
    dog = g_inner - g_outer
    dog = dog - dog.mean()
    return dog


def make_parent_and_children_mexican_hat(
    rng,
    grid_size=11,
    n_children=4,
    frequency: float = 2.0,
    sigma_inner=None,
    sigma_outer=None,
    correlation=0.6,
    center=None,
    jitter_center_sd=0.75,
    jitter_sigma_inner_sd=None,
    jitter_sigma_outer_sd=None,
):
    """
    Generate one parent map + n_children child maps using a Difference-of-Gaussians filter.

    The parent is generated with a DoG (inner minus outer Gaussian) pattern. Each
    child is generated to have a specified correlation r with the parent while
    maintaining a similar frequency/scale and inner/outer widths (with jitter).

    Children are created by mixing the parent pattern with independent Mexican-hat
    patterns:
        child = r * parent_z + sqrt(1 - r^2) * indep_z
    where *_z are standardized (zero-mean, unit-variance) flattened maps.
    """
    if not (-1 <= correlation <= 1):
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")

    parent = _dog_filter_2d(
        grid_size=grid_size,
        frequency=frequency,
        sigma_inner=sigma_inner,
        sigma_outer=sigma_outer,
        center=center,
    )

    # Standardize for correlation mixing
    parent_flat = parent.ravel()
    parent_mean = parent_flat.mean()
    parent_std = parent_flat.std()
    if parent_std < 1e-10:
        parent_std = 1.0
    parent_z = (parent_flat - parent_mean) / parent_std

    children = []
    r = correlation
    mix_b = np.sqrt(1 - r**2)

    for _ in range(n_children):
        # Create an "independent" DoG map with *almost* the same parameters,
        # but jittered so it's distinct even if the user provided a fixed `center`.
        base_center = center if center is not None else ((grid_size - 1) / 2.0, (grid_size - 1) / 2.0)
        c = (
            float(base_center[0] + rng.normal(0.0, jitter_center_sd)),
            float(base_center[1] + rng.normal(0.0, jitter_center_sd)),
        )

        # Base widths for jitter: use provided sigmas, otherwise derive from (grid_size, frequency)
        if sigma_inner is not None:
            base_sigma_inner = float(sigma_inner)
        else:
            wavelength = grid_size / frequency
            base_sigma_inner = wavelength / 4.0
        base_sigma_outer = float(sigma_outer) if sigma_outer is not None else 2.0 * base_sigma_inner

        s_in_sd = (0.05 * base_sigma_inner) if jitter_sigma_inner_sd is None else float(jitter_sigma_inner_sd)
        s_out_sd = (0.05 * base_sigma_outer) if jitter_sigma_outer_sd is None else float(jitter_sigma_outer_sd)

        sigma_inner_i = float(max(1e-6, base_sigma_inner + rng.normal(0.0, s_in_sd)))
        sigma_outer_i = float(max(1e-6, base_sigma_outer + rng.normal(0.0, s_out_sd)))

        indep = _dog_filter_2d(
            grid_size=grid_size,
            frequency=frequency,
            sigma_inner=sigma_inner_i,
            sigma_outer=sigma_outer_i,
            center=c
        )
        indep_flat = indep.ravel()
        indep_mean = indep_flat.mean()
        indep_std = indep_flat.std()
        if indep_std < 1e-10:
            indep_std = 1.0
        indep_z = (indep_flat - indep_mean) / indep_std

        # Orthogonalize indep against parent to guarantee corr(child, parent)=r.
        denom = float(np.dot(parent_z, parent_z))
        if denom < 1e-12:
            indep_orth = indep_z
        else:
            indep_orth = indep_z - (np.dot(parent_z, indep_z) / denom) * parent_z
        indep_orth_std = indep_orth.std()
        if indep_orth_std < 1e-10:
            indep_orth_std = 1.0
        indep_orth = (indep_orth - indep_orth.mean()) / indep_orth_std

        child_mixed = r * parent_z + mix_b * indep_orth
        child_grid = child_mixed.reshape(grid_size, grid_size)
        children.append(_min_max(child_grid))

    return _min_max(parent), children