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

    logits = value_final / tau
    logits = np.clip(logits, -40, 40)
    return np.exp(logits)


class SocialGPAgent(CellAgent):
    """GP-based explorer living on a Network grid using Value-Shaping."""

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

        # Memory buffers
        self.X_observations: list[tuple[int, int]] = []
        self.y_observations: list[float] = []

        # Prediction grid
        self.meshgrid = np.meshgrid(
            range(reward_environment.shape[0]),
            range(reward_environment.shape[1])
        )
        self.meshgrid_flatten = np.array(self.meshgrid, dtype=np.int32).reshape(2, -1).T
        self.meshgrid_dict = {tuple(coord): i for i, coord in enumerate(self.meshgrid_flatten)}
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

        logits = value_shaping(
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