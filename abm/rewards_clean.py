"""
Reward landscape generators (GP, Gabor, Mexican Hat families).

Main entry points:
- create_[family]_single: Generate a single landscape
- create_[family]_set: Generate a set of correlated landscapes (parent + children)

Utility functions are imported from rewards_utils.py.
"""

import numpy as np
from abm.rewards_utils import (
    _min_max,
    _gabor_filter,
    _mexican_hat_gaussian,
    mexican_hat_rbf,
    _cholesky_grid,
    build_correlation_matrix,
)


# ====================== GP (Gaussian Process) ======================

def create_gp_set(
    rng,
    grid_size=11,
    n_children=4,
    length_scale=2.0,
    corr_matrix=None,
):
    """
    Generate a set of correlated GP-based reward landscapes (parent + children).
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    grid_size : int
        Size of the grid (grid_size x grid_size)
    n_children : int
        Number of child landscapes to generate
    length_scale : float
        RBF kernel length scale
    corr_matrix : ndarray, optional
        Custom correlation matrix. If None, uses default 0.6 correlation.
    
    Returns
    -------
    parent : ndarray
        2D array of shape (grid_size, grid_size), normalized to [0, 1]
    children : list of ndarray
        List of n_children 2D arrays, each normalized to [0, 1]
    """
    from sklearn.gaussian_process.kernels import RBF
    
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    Xstar = np.column_stack([x.ravel(), y.ravel()])
    Sigma = RBF(length_scale)(Xstar)
    LSigma = np.linalg.cholesky(Sigma + 1e-10 * np.eye(Sigma.shape[0]))
    M = Sigma.shape[0]
    
    if corr_matrix is not None:
        R = np.asarray(corr_matrix, dtype=float)
        n_total = R.shape[0]
    else:
        n_total = n_children + 1
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


def create_gp_single(rng, grid_size=11, length_scale=2.0):
    """
    Generate a single GP-based reward landscape.
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    grid_size : int
        Size of the grid (grid_size x grid_size)
    length_scale : float
        RBF kernel length scale
    
    Returns
    -------
    grid : ndarray
        2D array of shape (grid_size, grid_size), normalized to [0, 1]
    """
    return _cholesky_grid(rng, grid_size, length_scale)


# ====================== Gabor ======================

def create_gabor_set(
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
    Generate a set of Gabor-based reward landscapes (parent + children).
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    grid_size : int
        Size of the grid
    n_children : int
        Number of child landscapes
    frequency : float
        Spatial frequency of Gabor filter
    theta_parent : float
        Orientation angle for parent (radians)
    sigma : float, optional
        Gaussian envelope width
    phase_parent : float, optional
        Phase offset for parent
    correlation : float
        Target correlation between parent and children
    theta_children : list of float, optional
        Orientation angles for children
    phase_children : list of float, optional
        Phase offsets for children
    center : tuple, optional
        Center of the filter
    
    Returns
    -------
    parent : ndarray
        2D array, normalized to [0, 1]
    children : list of ndarray
        List of child landscapes
    """
    if not (-1 <= correlation <= 1):
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")
    
    if phase_parent is None:
        phase_parent = rng.uniform(0, 2 * np.pi)
    
    parent = _gabor_filter(
        grid_size=grid_size,
        frequency=frequency,
        theta=theta_parent,
        sigma=sigma,
        phase=phase_parent,
        center=center,
    )
    
    parent_flat = parent.ravel()
    parent_mean = parent_flat.mean()
    parent_std = parent_flat.std()
    if parent_std < 1e-10:
        parent_std = 1.0
    parent_normalized = (parent_flat - parent_mean) / parent_std
    
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
        child_independent = _gabor_filter(
            grid_size=grid_size,
            frequency=frequency,
            theta=theta_children[i],
            sigma=sigma,
            phase=phase_children[i],
            center=center,
        )
        
        child_indep_flat = child_independent.ravel()
        child_indep_mean = child_indep_flat.mean()
        child_indep_std = child_indep_flat.std()
        if child_indep_std < 1e-10:
            child_indep_std = 1.0
        child_indep_normalized = (child_indep_flat - child_indep_mean) / child_indep_std
        
        r = correlation
        child_mixed = r * parent_normalized + np.sqrt(1 - r**2) * child_indep_normalized
        
        child_grid = child_mixed.reshape(grid_size, grid_size)
        children.append(_min_max(child_grid))
    
    parent_normalized_grid = _min_max(parent)
    return parent_normalized_grid, children


def create_gabor_single(rng, grid_size=11, frequency=2.0, theta=0.0, sigma=None, phase=None, center=None):
    """
    Generate a single Gabor-based reward landscape.
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    grid_size : int
        Size of the grid
    frequency : float
        Spatial frequency of Gabor filter
    theta : float
        Orientation angle (radians)
    sigma : float, optional
        Gaussian envelope width
    phase : float, optional
        Phase offset
    center : tuple, optional
        Center of the filter
    
    Returns
    -------
    grid : ndarray
        2D array, normalized to [0, 1]
    """
    if phase is None:
        phase = rng.uniform(0, 2 * np.pi)
    return _min_max(_gabor_filter(grid_size, frequency, theta, sigma, phase, center))


# ====================== Mexican Hat Simple ======================

def create_mexican_hat_simple_set(
    rng,
    grid_size=11,
    n_children=4,
    frequency=2.0,
    sigma_inner=None,
    sigma_outer=None,
    correlation=0.6,
    center=None,
    jitter_center_sd=0.75,
    jitter_sigma_inner_sd=None,
    jitter_sigma_outer_sd=None,
):
    """
    Generate a set of simple Mexican Hat (DoG) reward landscapes (parent + children).
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    grid_size : int
        Size of the grid
    n_children : int
        Number of child landscapes
    frequency : float
        Controls spatial scale
    sigma_inner : float, optional
        Inner Gaussian width
    sigma_outer : float, optional
        Outer Gaussian width
    correlation : float
        Target correlation between parent and children
    center : tuple, optional
        Center of the filter
    jitter_center_sd : float
        Standard deviation for center jitter
    jitter_sigma_inner_sd : float, optional
        Standard deviation for inner sigma jitter
    jitter_sigma_outer_sd : float, optional
        Standard deviation for outer sigma jitter
    
    Returns
    -------
    parent : ndarray
        2D array, normalized to [0, 1]
    children : list of ndarray
        List of child landscapes
    """
    if not (-1 <= correlation <= 1):
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")
    
    parent = _mexican_hat_gaussian(
        grid_size=grid_size,
        frequency=frequency,
        sigma_inner=sigma_inner,
        sigma_outer=sigma_outer,
        center=center,
    )
    
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
        base_center = center if center is not None else ((grid_size - 1) / 2.0, (grid_size - 1) / 2.0)
        c = (
            float(base_center[0] + rng.normal(0.0, jitter_center_sd)),
            float(base_center[1] + rng.normal(0.0, jitter_center_sd)),
        )
        
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
        
        indep = _mexican_hat_gaussian(
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


def create_mexican_hat_simple_single(rng, grid_size=11, frequency=2.0, sigma_inner=None, sigma_outer=None, center=None):
    """
    Generate a single simple Mexican Hat (DoG) reward landscape.
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    grid_size : int
        Size of the grid
    frequency : float
        Controls spatial scale
    sigma_inner : float, optional
        Inner Gaussian width
    sigma_outer : float, optional
        Outer Gaussian width
    center : tuple, optional
        Center of the filter
    
    Returns
    -------
    grid : ndarray
        2D array, normalized to [0, 1]
    """
    return _min_max(_mexican_hat_gaussian(grid_size, frequency, sigma_inner, sigma_outer, center))


# ====================== Mexican Hat GP (Mexican Hat on GP) ======================

def create_mexican_hat_gp_set(
    rng,
    grid_size=33,
    n_children=1,
    length_scale=4.5,
    target_correlation=1.0,
    sigma_inner=None,
    sigma_outer=None,
    fixed_min_coords=False,
    local_global_max_ratio=1.2,
):
    """
    Generate a set of Mexican Hat over correlated GP reward landscapes (parent + children).
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    grid_size : int
        Size of the grid
    n_children : int
        Number of child landscapes
    length_scale : float
        GP RBF kernel length scale
    target_correlation : float
        Target correlation between parent and children
    sigma_inner : float, optional
        Inner Gaussian width for Mexican Hat kernel
    sigma_outer : float, optional
        Outer Gaussian width for Mexican Hat kernel
    fixed_min_coords : bool
        If True, use same minimum coordinates for all landscapes
    local_global_max_ratio : float
        Ratio of local (GP) max to global (Mexican Hat kernel) max amplitude.
        The GP landscape is scaled so its max = 1, then the Mexican Hat kernel
        is scaled so its max = local_global_max_ratio.
    
    Returns
    -------
    parent : ndarray
        2D array, normalized to [0, 1]
    children : list of ndarray
        List of child landscapes
    min_coords : tuple
        Coordinates of the global minimum
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if not (-1 <= target_correlation <= 1):
        raise ValueError(f"target_correlation must be in [-1, 1], got {target_correlation}")
    
    if target_correlation == 1.0:
        parent_mix, min_coords = create_mexican_hat_gp_single(
            rng=rng,
            grid_size=grid_size,
            length_scale=length_scale,
            sigma_inner=sigma_inner,
            sigma_outer=sigma_outer,
            local_global_max_ratio=local_global_max_ratio,
        )
        children_mix = [parent_mix.copy() for _ in range(n_children)]
        return parent_mix, children_mix, min_coords
    
    corr_matrix = build_correlation_matrix(n_total=n_children + 1, r=target_correlation)
    parent, children = create_gp_set(
        rng,
        grid_size=grid_size,
        n_children=n_children,
        length_scale=length_scale,
        corr_matrix=corr_matrix,
    )
    
    parent_mix, min_coords = create_mexican_hat_gp_single(
        parent,
        length_scale=length_scale,
        sigma_inner=sigma_inner,
        sigma_outer=sigma_outer,
        local_global_max_ratio=local_global_max_ratio,
    )
    
    mh_center = min_coords if fixed_min_coords else None
    
    children_mix = []
    for child in children:
        mix, _ = create_mexican_hat_gp_single(
            child,
            length_scale=length_scale,
            sigma_inner=sigma_inner,
            sigma_outer=sigma_outer,
            min_coords=mh_center,
            local_global_max_ratio=local_global_max_ratio,
        )
        children_mix.append(mix)
    
    return parent_mix, children_mix, min_coords


def create_mexican_hat_gp_single(
    parent=None,
    rng=None,
    grid_size=25,
    length_scale=10.0,
    sigma_inner=None,
    sigma_outer=None,
    min_coords=None,
    local_global_max_ratio=1.2,
):
    """
    Generate a single Mexican Hat over correlated GP reward landscape.
    
    Parameters
    ----------
    parent : ndarray, optional
        Existing GP landscape to overlay Mexican Hat kernel on
    rng : np.random.Generator, optional
        Random number generator (used if parent is None)
    grid_size : int
        Size of the grid (used if parent is None)
    length_scale : float
        GP RBF kernel length scale
    sigma_inner : float, optional
        Inner Gaussian width for Mexican Hat kernel
    sigma_outer : float, optional
        Outer Gaussian width for Mexican Hat kernel
    min_coords : tuple, optional
        Coordinates for Mexican Hat kernel center
    local_global_max_ratio : float
        Ratio of local (GP) max to global (Mexican Hat kernel) max amplitude.
        The GP landscape is scaled so its max = 1, then the Mexican Hat kernel
        is scaled so its max = local_global_max_ratio.
    
    Returns
    -------
    mix : ndarray
        Combined GP + Mexican Hat landscape, normalized to [0, 1]
    mh_kernel : ndarray
        The Mexican Hat kernel component
    min_coords : tuple
        Coordinates of the global minimum
    """
    if parent is None:
        parent = _cholesky_grid(rng, grid_size, length_scale)
    
    grid_size = parent.shape[0]
    
    # Scale GP so its max amplitude = 1 (local max = 1)
    gp_scaled = _min_max(parent)
    gp_max = gp_scaled.max()
    if gp_max > 0:
        gp_scaled = gp_scaled / gp_max
    
    if sigma_inner is None and sigma_outer is None:
        sigma_outer = length_scale
        sigma_inner = sigma_outer / 2.0
    
    if min_coords is None:
        min_coords = np.unravel_index(np.argmin(gp_scaled), gp_scaled.shape)
    
    # Generate Mexican Hat kernel
    mh_kernel = mexican_hat_rbf(
        grid_size=grid_size,
        sigma_inner=sigma_inner,
        sigma_outer=sigma_outer,
        center=min_coords
    )
    
    # Scale MH kernel so its max amplitude = local_global_max_ratio
    # (since GP/local max is already scaled to 1)
    mh_kernel = mh_kernel * local_global_max_ratio
    
    return gp_scaled + mh_kernel, min_coords


# ====================== Two Valley ======================

def create_two_valley_set(
    rng=None,
    n_children=1,
    grid_size=33,
    length_scale=4.5,
    sigma_inner=None,
    sigma_outer=None,
    secondary_sigma=None,
    secondary_amplitude=0.8,
    mh_exclusion_radius=None,
):
    """
    Generate a set of Mexican Hat with two valleys reward landscapes (parent + children).
    
    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator
    n_children : int
        Number of child landscapes
    grid_size : int
        Size of the grid
    length_scale : float
        GP RBF kernel length scale
    sigma_inner : float, optional
        Inner Gaussian width for Mexican Hat kernel
    sigma_outer : float, optional
        Outer Gaussian width for Mexican Hat kernel
    secondary_sigma : float, optional
        Sigma for secondary valley
    secondary_amplitude : float
        Amplitude of secondary valley
    mh_exclusion_radius : float, optional
        Radius to exclude when finding second minimum
    
    Returns
    -------
    env : ndarray
        Two-valley landscape, normalized to [0, 1]
    children : list of ndarray
        List of child landscapes (copies of env)
    min_coords : tuple
        Coordinates of the primary minimum
    """
    env, _, _, min_coords, _, _ = create_two_valley_single(
        rng=rng,
        grid_size=grid_size,
        length_scale=length_scale,
        sigma_inner=sigma_inner,
        sigma_outer=sigma_outer,
        secondary_sigma=secondary_sigma,
        secondary_amplitude=secondary_amplitude,
        mh_exclusion_radius=mh_exclusion_radius,
    )

    return env, [env.copy() for _ in range(n_children)], min_coords


def create_two_valley_single(
    rng=None,
    grid_size=25,
    length_scale=10.0,
    sigma_inner=None,
    sigma_outer=None,
    secondary_sigma=None,
    secondary_amplitude=0.8,
    mh_exclusion_radius=None,
):
    """
    Generate a single Mexican Hat with two valleys reward landscape.
    
    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator
    grid_size : int
        Size of the grid
    length_scale : float
        GP RBF kernel length scale
    sigma_inner : float, optional
        Inner Gaussian width for Mexican Hat kernel
    sigma_outer : float, optional
        Outer Gaussian width for Mexican Hat kernel
    secondary_sigma : float, optional
        Sigma for secondary valley
    secondary_amplitude : float
        Amplitude of secondary valley
    mh_exclusion_radius : float, optional
        Radius to exclude when finding second minimum
    
    Returns
    -------
    mix : ndarray
        Two-valley landscape, normalized to [0, 1]
    gp : ndarray
        The GP component
    mh_kernel : ndarray
        The Mexican Hat kernel component
    second_valley : ndarray
        The secondary valley component
    min_coords : tuple
        Coordinates of the primary minimum
    second_min_coords : tuple or None
        Coordinates of the secondary minimum
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if sigma_inner is None and sigma_outer is None:
        sigma_outer = length_scale
        sigma_inner = sigma_outer / 2.0
    
    if secondary_sigma is None:
        secondary_sigma = sigma_inner
    
    if mh_exclusion_radius is None:
        mh_exclusion_radius = float(sigma_outer)
    
    gp = _cholesky_grid(rng, grid_size, length_scale=length_scale)
    min_coords = np.unravel_index(np.argmin(gp), gp.shape)
    
    mh_kernel = mexican_hat_rbf(
        grid_size=grid_size,
        sigma_inner=sigma_inner,
        sigma_outer=sigma_outer,
        center=min_coords,
    )
    mh_kernel = mh_kernel / np.abs(mh_kernel).max() * 1.2
    mh_kernel = mh_kernel - mh_kernel.mean()
    
    rows, cols = np.indices(gp.shape)
    r2_from_primary = (rows - min_coords[0])**2 + (cols - min_coords[1])**2
    exclusion_mask = r2_from_primary <= float(mh_exclusion_radius) ** 2
    candidates = ~exclusion_mask
    
    second_min_coords = None
    second_valley = np.zeros_like(gp)
    
    if np.any(candidates):
        second_index = np.argmin(np.where(candidates, gp, np.inf))
        second_min_coords = np.unravel_index(second_index, gp.shape)
        
        r2 = (rows - second_min_coords[0])**2 + (cols - second_min_coords[1])**2
        second_valley = -float(secondary_amplitude) * np.exp(
            -r2 / (2.0 * float(secondary_sigma) ** 2)
        )
    
    mix = _min_max(gp + mh_kernel + second_valley)
    
    return mix, gp, mh_kernel, second_valley, min_coords, second_min_coords