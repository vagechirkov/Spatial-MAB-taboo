import numpy as np
from mesa.discrete_space import CellAgent
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

def gp_base_generalization(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_predict: np.ndarray,
    kernel: Kernel,
    observation_noise: np.ndarray | float,
    rng,
):
    """Fit a zero-mean GP and return μ, σ on the prediction grid."""
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=observation_noise,
        random_state=rng,
        optimizer=None,
        normalize_y=False,
    )
    gpr.fit(X_obs, y_obs)
    return gpr.predict(X_predict, return_std=True)

def make_mh_prior_mean_fn(mh_kernel_scaled: np.ndarray) -> callable:
    """
    Wrap the scaled Mexican Hat grid as a callable prior mean function.

    Parameters
    ----------
    mh_kernel_scaled : ndarray of shape (grid_size, grid_size)
        The mh_kernel_scaled returned by create_mexican_hat_gp_single.
        If any normalization is applied to the combined landscape after
        generation, the same normalization must be applied here first,
        otherwise the prior mean will be on a different scale than the
        observations the GP receives.

    Returns
    -------
    prior_mean_fn : callable (n, 2) -> (n,)
        Evaluates the Mexican Hat prior mean at arbitrary grid coordinates.
        Coordinates are cast to int, so float inputs are floored — pass
        exact integer coordinates to avoid ambiguity.
    """
    def prior_mean_fn(X: np.ndarray) -> np.ndarray:
        rows = X[:, 0].astype(int)
        cols = X[:, 1].astype(int)
        return mh_kernel_scaled[rows, cols]

    return prior_mean_fn

def gp_oracle_generalization(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_predict: np.ndarray,
    kernel: Kernel,
    observation_noise: np.ndarray | float,
    rng,
    prior_mean_fn: callable = None,
):
    """
    Fit a GP with an optional prior mean function and return μ, σ on the
    prediction grid.

    Standard agent  : omit prior_mean_fn → zero-mean GP, original behavior.
    Oracle GP-UCB   : pass prior_mean_fn from make_mh_prior_mean_fn.

    The GP is fit on the mean-subtracted residuals (y - μ₀(X_obs)).
    The prior mean is added back to the posterior mean at prediction time.
    Posterior variance is unaffected by the prior mean — it depends only
    on the kernel and the spatial configuration of observations.

    Parameters
    ----------
    prior_mean_fn : callable (n, 2) -> (n,), optional
        Evaluates the deterministic prior mean at a set of grid coordinates.
        When None, the prior mean is zero everywhere (standard behavior).
    """
    if prior_mean_fn is not None:
        y_residual = y_obs - prior_mean_fn(X_obs)
        mu_prior_pred = prior_mean_fn(X_predict)
    else:
        y_residual = y_obs
        mu_prior_pred = 0.0

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=observation_noise,
        random_state=rng,
        optimizer=None,
        normalize_y=False,
    )
    gpr.fit(X_obs, y_residual)
    mu_pred, sigma_pred = gpr.predict(X_predict, return_std=True)

    return mu_pred + mu_prior_pred, sigma_pred

def value_shaping(
    X_obs_private: np.ndarray,
    y_obs_private: np.ndarray,
    X_obs_social: list[np.ndarray],
    y_obs_social: list[np.ndarray],
    X_predict: np.ndarray,
    length_scale_private: float,
    length_scale_social: float,
    observation_noise_private: float,
    observation_noise_social: float,
    beta_private: float,
    beta_social: float,
    alpha: float,
    tau: float,
    random_state,
) -> np.ndarray:
    """
    GP-Value Shaping: Weighted average of private UCB and social UCB (average of neighbors).
    """
    # Private GP
    gp_mean_p, gp_std_p = gp_base_generalization(
        X_obs_private,
        y_obs_private,
        X_predict,
        RBF(length_scale=length_scale_private),
        np.ones(len(X_obs_private)) * observation_noise_private,
        random_state,
    )
    value_ucb_private = gp_mean_p + beta_private * gp_std_p

    # Social GPs (one per neighbor)
    if len(X_obs_social) > 0 and any(len(xs) > 0 for xs in X_obs_social):
        ucb_s_list = []
        for xs, ys in zip(X_obs_social, y_obs_social):
            if len(xs) == 0:
                continue
            gp_mean_s, gp_std_s = gp_base_generalization(
                xs,
                ys,
                X_predict,
                RBF(length_scale=length_scale_social),
                np.ones(len(xs)) * observation_noise_social,
                random_state,
            )
            ucb_s_list.append(gp_mean_s + beta_social * gp_std_s)
        
        if ucb_s_list:
            value_ucb_social = np.mean(np.vstack(ucb_s_list), axis=0)
            value_final = (1.0 - alpha) * value_ucb_private + alpha * value_ucb_social
        else:
            value_final = value_ucb_private
    else:
        value_final = value_ucb_private

    # Edited 2026.02.25: Return raw values instead of probabilities for visualization. Original logic added to line 198-200
    return value_final
    # logits = value_final / tau
    # logits = np.clip(logits, -40, 40)
    # return np.exp(logits)


def social_generalization(
    X_obs_private: np.ndarray,
    y_obs_private: np.ndarray,
    X_obs_social: list[np.ndarray],
    y_obs_social: list[np.ndarray],
    X_predict: np.ndarray,
    length_scale: float,
    observation_noise: float,
    sigma_social: float,
    beta: float,
    random_state,
) -> np.ndarray:
    """Pool private and social observations in one GP-UCB model.

    ``sigma_social`` is an additive observation-noise variance. Private rows
    receive ``observation_noise`` and social rows receive
    ``observation_noise + sigma_social``.
    """
    social_pairs = [
        (np.asarray(xs), np.asarray(ys).reshape(-1))
        for xs, ys in zip(X_obs_social, y_obs_social)
        if len(xs) > 0
    ]

    X_private = np.asarray(X_obs_private)
    y_private = np.asarray(y_obs_private).reshape(-1)

    if social_pairs:
        X_social = np.concatenate([xs for xs, _ in social_pairs], axis=0)
        y_social = np.concatenate([ys for _, ys in social_pairs], axis=0)
        X_observed = np.concatenate((X_private, X_social), axis=0)
        y_observed = np.concatenate((y_private, y_social), axis=0)
        observation_noise_by_row = np.concatenate(
            (
                np.full(len(X_private), observation_noise, dtype=float),
                np.full(
                    len(X_social),
                    observation_noise + sigma_social,
                    dtype=float,
                ),
            )
        )
    else:
        X_observed = X_private
        y_observed = y_private
        observation_noise_by_row = np.full(
            len(X_private), observation_noise, dtype=float
        )

    gp_mean, gp_std = gp_base_generalization(
        X_observed,
        y_observed,
        X_predict,
        RBF(length_scale=length_scale),
        observation_noise_by_row,
        random_state,
    )
    return gp_mean + beta * gp_std


class SocialGPAgent(CellAgent):
    """GP-based explorer using a model-selected social-information mechanism."""

    def __init__(
        self,
        model,
        cell,
        reward_environment: np.ndarray,
        length_scale_private: float,
        length_scale_social: float,
        observation_noise_private: float,
        observation_noise_social: float,
        beta_private: float,
        beta_social: float,
        tau: float,
        alpha: float,
        sigma_social: float = 0.0,
        social_information_mode: str = "value_shaping",
    ):
        super().__init__(model)
        self.cell = cell
        self.reward_environment = reward_environment

        # Hyperparameters
        self.length_scale_private = length_scale_private
        self.length_scale_social = length_scale_social
        self.observation_noise_private = observation_noise_private
        self.observation_noise_social = observation_noise_social
        self.beta_private = beta_private
        self.beta_social = beta_social
        self.tau = tau
        self.alpha = alpha
        self.sigma_social = sigma_social
        self.social_information_mode = social_information_mode

        # Memory buffers
        self.X_observations: list[tuple[int, int]] = []
        self.y_observations: list[float] = []

        # Prediction grid
        rows, cols = np.indices(reward_environment.shape, dtype=np.int32)
        self.meshgrid_flatten = np.column_stack((rows.ravel(), cols.ravel()))
        self.meshgrid_dict = {tuple(coord): i for i, coord in enumerate(self.meshgrid_flatten)}
        self.ucb = np.zeros(len(self.meshgrid_flatten))
        self.policy = np.ones(len(self.meshgrid_flatten)) / len(self.meshgrid_flatten)

    @property
    def last_choice(self) -> tuple[int, int]:
        return self.X_observations[-1] if self.X_observations else None

    @property
    def last_reward(self) -> float:
        return self.y_observations[-1] if self.y_observations else 0.0

    @property
    def total_reward(self) -> float:
        return np.sum(self.y_observations)
    
    @property
    def policy_grid(self) -> np.ndarray:
        grid = np.zeros(self.reward_environment.shape)
        for coord, prob in zip(self.meshgrid_flatten, self.policy):
            grid[tuple(coord)] = prob
        return grid
    
    @property
    def ucb_grid(self) -> np.ndarray:
        grid = np.zeros(self.reward_environment.shape)
        for coord, ucb in zip(self.meshgrid_flatten, self.ucb):
            grid[tuple(coord)] = ucb
        return grid

    def _gather_social_info(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        neighbours = list(self.model.grid[self.cell.coordinate].neighborhood)
        X_soc, y_soc = [], []
        history_horizon = self.model.steps - 1
        for neighbour in neighbours:
            neighbor_agent = neighbour.agents[0]
            X_soc.append(np.array(neighbor_agent.X_observations[:history_horizon]))
            y_soc.append(np.array(neighbor_agent.y_observations[:history_horizon]).reshape(-1, 1))
        return X_soc, y_soc

    def _add_noise_to_reward(self, reward: float):
        if hasattr(self.model, "reward_noise_sd") and (self.model.reward_noise_sd > 0):
            return reward + self.model.rng.normal(0, self.model.reward_noise_sd)
        return reward

    def _random_choice(self) -> None:
        idx = self.model.rng.integers(0, len(self.meshgrid_flatten))
        coord = tuple(self.meshgrid_flatten[idx])
        reward = self._add_noise_to_reward(float(self.reward_environment[coord]))
        self.X_observations.append(coord)
        self.y_observations.append(reward)

    def _make_choice(self):
        X_priv = np.array(self.X_observations)
        y_priv = np.array(self.y_observations).reshape(-1, 1)
        X_soc, y_soc = self._gather_social_info()

        if self.social_information_mode == "social_generalization":
            self.ucb = social_generalization(
                X_priv,
                y_priv,
                X_soc,
                y_soc,
                self.meshgrid_flatten,
                length_scale=self.length_scale_private,
                observation_noise=self.observation_noise_private,
                sigma_social=self.sigma_social,
                beta=self.beta_private,
                random_state=self.model.rng.__getstate__(),
            )
        else:
            self.ucb = value_shaping(
                X_priv,
                y_priv,
                X_soc,
                y_soc,
                self.meshgrid_flatten,
                length_scale_private=self.length_scale_private,
                length_scale_social=self.length_scale_social,
                observation_noise_private=self.observation_noise_private,
                observation_noise_social=self.observation_noise_social,
                beta_private=self.beta_private,
                beta_social=self.beta_social,
                alpha=self.alpha,
                tau=self.tau,
                random_state=self.model.rng.__getstate__(),
            )

        # Edited 2026.02.25
        logits = self.ucb / self.tau
        logits = np.exp(np.clip(logits, -40, 40))

        probs = logits.ravel()
        probs = np.nan_to_num(probs, nan=1e-12)
        probs += 1e-12
        probs /= probs.sum()
        self.policy = probs

        idx = self.model.rng.choice(len(self.policy), p=self.policy)
        coord = tuple(self.meshgrid_flatten[idx])
        reward = self._add_noise_to_reward(float(self.reward_environment[coord]))
        self.X_observations.append(coord)
        self.y_observations.append(reward)

    def step(self):
        if len(self.X_observations) == 0:
            self._random_choice()
        else:
            self._make_choice()
