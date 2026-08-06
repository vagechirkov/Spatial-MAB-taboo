import numpy as np
from thesis.utils import _min_max, _mexican_hat_DoG
from sklearn.gaussian_process.kernels import RBF

"""
3.2.2 Sampling environments from GP priors
"""
def sample_RBF_GP(
    rng,
    n=20,
    length_scale=3.0,
    jitter=1e-8,
    normalize=True,
):
    """
    Generate a single 2D Gaussian Process grid using an RBF kernel.
    
    Parameters
    ----------
    rng : np.random.Generator
    n : int
        Size of square grid (n x n)
    length_scale : float
        RBF kernel length scale
    jitter : float
        Small diagonal term for numerical stability
    normalize : bool
        If True, apply min-max scaling to [0,1]
        
    Returns
    -------
    f : ndarray (n, n)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    X = np.column_stack([x.ravel(), y.ravel()])

    # Covariance matrix from sklearn RBF kernel
    kernel = RBF(length_scale=length_scale)
    K = kernel(X)

    # Cholesky decomposition
    L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))

    # Sample from GP prior
    z = rng.standard_normal(K.shape[0])
    sample = L @ z

    f = sample.reshape(n, n)
    if normalize:
        f = _min_max(f)

    return f, K

def sample_Gabor_GP(
    rng,
    n=20,
    frequency=0.1,
    sigma=3.0,
    theta=0.0,
    amplitude=1.0,
    jitter=1e-8,
):
    """Sample a 2-D function from a GP with a Gabor covariance kernel.

    The kernel is a squared-exponential envelope multiplied by a cosine:
    k(x, x') = amplitude**2 * exp(-||x-x'||**2 / (2*sigma**2))
                 * cos(2*pi*frequency*u_theta.T@(x-x')).
    """
    if n < 1:
        raise ValueError("n must be positive")
    if sigma <= 0 or amplitude <= 0:
        raise ValueError("sigma and amplitude must be positive")

    rows, cols = np.mgrid[:n, :n]
    points = np.column_stack((rows.ravel(), cols.ravel()))
    differences = points[:, None, :] - points[None, :, :]

    squared_distances = np.sum(differences**2, axis=-1)
    direction = np.array([np.sin(theta), np.cos(theta)])
    projected_distances = differences @ direction
    covariance = amplitude**2 * np.exp(
        -squared_distances / (2 * sigma**2)
    ) * np.cos(2 * np.pi * frequency * projected_distances)
    covariance.flat[:: covariance.shape[0] + 1] += jitter

    sample = rng.multivariate_normal(np.zeros(points.shape[0]), covariance)
    return sample.reshape(n, n), covariance

def sample_DoG_GP(
    rng=None,
    n=20,
    sigma_outer=3.0,
    sigma_inner=1.5,
    amplitude=1.0,
    jitter=1e-8,
):
    """Sample a 2-D GP with a difference-of-Gaussians prior kernel.

    Let ``h = G(sigma_inner) - G(sigma_outer)``, where each ``G`` is a
    normalized isotropic Gaussian density.  The covariance is the stationary
    autocorrelation kernel

    ``k(x, x') = amplitude**2 * <h(x - .), h(x' - .)> / <h, h>``.

    Expanding the convolution gives a difference-of-Gaussians-shaped kernel
    with three Gaussian terms.  Defining the kernel as an autocorrelation is
    important: it guarantees positive semidefiniteness, whereas simply
    subtracting two RBF covariance kernels does not generally define a valid
    GP.  ``amplitude`` is the marginal standard deviation before adding
    numerical jitter.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator. A new generator is used when omitted.
    n : int, default=20
        Side length of the square sampling grid.
    sigma_inner, sigma_outer : float
        Spatial standard deviations of the normalized Gaussian filters, in
        grid-cell units. ``sigma_outer`` must exceed ``sigma_inner``.
    amplitude : float, default=1.0
        Marginal standard deviation of the GP.
    jitter : float, default=1e-10
        Non-negative diagonal term used for numerical stability.

    Returns
    -------
    ndarray
        A sample with shape ``(grid_size, grid_size)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(sigma_inner) or sigma_inner <= 0:
        raise ValueError("sigma_inner must be finite and positive")
    if not np.isfinite(sigma_outer) or sigma_outer <= sigma_inner:
        raise ValueError("sigma_outer must be finite and greater than sigma_inner")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and non-negative")

    rows, cols = np.mgrid[:n, :n]
    points = np.column_stack((rows.ravel(), cols.ravel()))
    differences = points[:, None, :] - points[None, :, :]
    squared_distances = np.sum(differences**2, axis=-1).astype(np.longdouble)

    def gaussian_overlap(variance):
        # Density of N(0, variance * I_2), evaluated at pairwise offsets.
        variance = np.longdouble(variance)
        return np.exp(-squared_distances / (2.0 * variance)) / (
            2.0 * np.longdouble(np.pi) * variance
        )

    inner_variance = 2.0 * sigma_inner**2
    cross_variance = sigma_inner**2 + sigma_outer**2
    outer_variance = 2.0 * sigma_outer**2
    covariance = (
        gaussian_overlap(inner_variance)
        - 2.0 * gaussian_overlap(cross_variance)
        + gaussian_overlap(outer_variance)
    )

    # Normalize k(0) to amplitude**2. All diagonal entries are identical
    # because this is a stationary kernel.
    covariance *= np.longdouble(amplitude)**2 / covariance[0, 0]
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0

    # Project away negative eigenvalues caused solely by cancellation in the
    # three-term DoG expansion. Analytically all eigenvalues are non-negative
    # because the kernel is an autocorrelation.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0) + jitter
    sample = eigenvectors @ (
        np.sqrt(eigenvalues) * rng.standard_normal(points.shape[0])
    )
    return sample.reshape(n, n), covariance


"""
3.2.6 Mexican Hat manipulation
"""
def create_mexican_hat_MAB(
    rng=None,
    n=20,
    length_scale=3.0,
    sigma_outer=None,
    sigma_inner=None,
    r=1.2,
):
    """
    Returns
    -------
    f_MH : ndarray
        Combined GP + Mexican Hat landscape, normalized to [0, 1].

    x_min : tuple
        Coordinates of the Mexican Hat center.

    mh_kernel_normalized : ndarray
        The scaled Mexican Hat component, normalized using the same
        min/max as the combined landscape. This is the deterministic
        prior mean used by oracle GP-UCB.
    """
    if sigma_inner is None and sigma_outer is None:
        sigma_outer = length_scale
        sigma_inner = sigma_outer / 2.0

    # Sample a GP landscape using an RBF kernel
    f_RBF, _ = sample_RBF_GP(
        rng=rng,
        n=n,
        length_scale=length_scale,
    )

    # Find the minimum and maximum of f_RBF
    x_min = np.unravel_index(np.argmin(f_RBF), f_RBF.shape)
    x_max = np.unravel_index(np.argmax(f_RBF), f_RBF.shape)
    f_RBF_max = f_RBF[x_max]
    f_RBF_min = f_RBF[x_min]

    # Generate DoG kernel centered at the minimum of f_RBF
    K_DoG = _mexican_hat_DoG(
        n=n,
        sigma_inner=sigma_inner,
        sigma_outer=sigma_outer,
        center=x_min,
    )
    K_DoG_at_x_max = K_DoG[x_max]

    # Check that the f_RBF max is not too close to the Mexican Hat center
    denom = 1.0 - r * K_DoG_at_x_max
    if denom <= 0:
        raise ValueError(
            "Requested local_global_max_ratio is infeasible because the "
            "original GP maximum lies too close to the Mexican Hat center."
        )

    # Solve for A such that f_MH[x_min] / f_MH[x_max] = r
    amplitude = (r * f_RBF_max - f_RBF_min) / denom
    f_MH = f_RBF + amplitude * K_DoG
    # sanity check
    f_MH_max = f_MH[x_min]
    f_MH_at_f_RBF_max = f_MH[x_max]
    assert np.isclose(
        f_MH_max / f_MH_at_f_RBF_max,
        r,
        atol=1e-10,
    )

    # Normalize everything using f_MH and return
    return (
        _min_max(f_MH),
        x_min,
        _min_max(K_DoG, f_MH.min(), f_MH.max()),
        amplitude,
    )


"""
3.3 Results
"""
def posterior_learning_trap(
    mu, 
    K, 
    f,
):
    """
    Returns L, a measure of false beliefs in the GP posterior, given the current GP posterior mean and
    covariance.

    Parameters
    ----------
    mu : ndarray, shape (n, n)
        Current GP posterior mean.
    K : ndarray, shape (n*n, n*n)
        Current GP posterior covariance matrix.
    f : ndarray, shape (n, n)
        True underlying function values.

    Returns
    -------
    L : float
        Measure of false beliefs in the GP posterior.
    """
    n = mu.shape[0]
    if K.shape != (n*n, n*n):
        raise ValueError("K must have shape (n*n, n*n) where n is the grid size.")

    