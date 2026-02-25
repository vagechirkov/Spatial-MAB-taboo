import mesa
import networkx as nx
import numpy as np
import pandas as pd

from .agent import SocialGPAgent
from mesa import DataCollector
from mesa.discrete_space import Network
from .rewards import (
    make_parent_and_children_cholesky,
    make_parent_and_children_gabor,
    make_parent_and_children_mexican_hat,
    build_corr_matrix_bare_bones,
    make_parent_and_children_correlated_dog,
)

def _build_network(network_type, n):
    if network_type == "fully_connected":
        return nx.complete_graph(n)
    raise ValueError(f"Unknown network_type '{network_type}'")

class SocialGPModel(mesa.Model):
    """
    Bare-bones GP-Value Shaping model on correlated landscapes.
    """
    def __init__(
        self,
        *,
        n: int = 4,
        grid_size: int = 11,
        length_scale: float = 1.11, # for agents
        observation_noise: float = 0.01,
        beta: float = 0.5,
        tau: float = 0.01,
        alpha: float = 0.6,
        network_type: str = "fully_connected",
        reward_noise_sd : float = 0.001,
        reward_env_type: str = "gp",
        reward_env_params: dict | None = None,
        corr_matrix: np.ndarray | None = None,
        **kwargs,
    ):      
        # Handle seed/rng from kwargs to support mesa.batch_run
        # If 'seed' is passed in parameters, it ends up in kwargs.
        seed = kwargs.pop("seed", None)
        rng = kwargs.pop("rng", None)
        
        # If seed is provided, valid rng is derived from it, so we can ignore any passed rng 
        # to avoid "both seed and rng provided" error in Model.__init__
        if seed is not None:
            rng = None
            
        super().__init__(seed=seed, rng=rng, **kwargs)

        self.num_agents = n
        self.grid_size = grid_size
        self.reward_noise_sd = reward_noise_sd
        # Set reward peak location (used for gabor and mexican hat)
        if reward_env_params and 'center' in reward_env_params:
            self.reward_peak = np.array(reward_env_params['center'])
        # Generate reward environments
        reward_env_params = {} if reward_env_params is None else dict(reward_env_params)

        if reward_env_type == "gp":
            # Correlated GP landscapes via a task correlation matrix
            if corr_matrix is None:
                corr_matrix = build_corr_matrix_bare_bones(n + 1)
            env_length_scale = float(reward_env_params.pop("length_scale", 2.0))
            parent, child_maps = make_parent_and_children_cholesky(
                rng=self.rng,
                grid_size=grid_size,
                n_children=n,
                length_scale=env_length_scale,
                corr_matrix=corr_matrix,
                **reward_env_params,
            )
        elif reward_env_type == "gabor":
            # Parent + children with target correlation (scalar) and shared frequency
            parent, child_maps = make_parent_and_children_gabor(
                rng=self.rng,
                grid_size=grid_size,
                n_children=n,
                **reward_env_params,
            )
        elif reward_env_type in ("mexican_hat", "dog"):
            # Currently implemented as DoG-based mexican hat
            parent, child_maps = make_parent_and_children_mexican_hat(
                rng=self.rng,
                grid_size=grid_size,
                n_children=n,
                **reward_env_params,
            )
        elif reward_env_type == "corr_dog":
            # Correlated DoG landscapes
            parent, child_maps, self.reward_peak = make_parent_and_children_correlated_dog(
                rng=self.rng,
                grid_size=grid_size,
                n_children=n,
                **reward_env_params,
            )
            if 'lambda_inner' in reward_env_params:
                self.peak_radius = reward_env_params['lambda_inner']
            else:
                # NOTE: assumes reward environment generation maintains this ratio.  Must change if we change reward env generation logic.
                self.peak_radius = reward_env_params['length_scale'] // 5.0
                self.moat_radius = reward_env_params['length_scale'] // 2.0

        else:
            raise ValueError(
                "Unknown reward_env_type. Expected one of: "
                "'gp', 'gabor', 'mexican_hat' (alias: 'dog'). "
                f"Got: {reward_env_type!r}"
            )

        # Keep reference (handy for debugging/visualization)
        self.reward_parent = parent
        self.reward_maps = child_maps

        # Build social network
        G = _build_network(network_type, n)
        self.grid = Network(G, random=self.random)

        # Create agents
        # Here we shift maps to be zero-centered around 0.5 (original maps are in [0, 1])
        # This matches the zero-mean GP assumption if we consider 0.5 as the baseline.
        # Alternatively, we just use them as is. Keeping consistent with previous logic.
        child_maps = [c - 0.5 for c in child_maps]

        SocialGPAgent.create_agents(
            self,
            n,
            cell=self.grid.all_cells.cells,
            reward_environment=child_maps,
            length_scale_private=length_scale,
            length_scale_social=length_scale,
            observation_noise_private=observation_noise,
            observation_noise_social=observation_noise,
            beta_private=beta,
            beta_social=beta,
            tau=tau,
            alpha=alpha,
        )

        def dist_to_peak(agent):
            '''
            Returns the distance of the agent's last choice to the edge the reward peak
            '''
            # minimum distance from agent.last_choice to circle defined by reward_peak and peak_radius
            if (
                not hasattr(self, 'reward_peak')
                or self.reward_peak is None
                or not hasattr(self, 'peak_radius')
                or self.peak_radius is None
            ):
                return np.inf  # If reward_peak is not defined, return infinity
            choice = np.array(agent.last_choice)
            peak = np.array(self.reward_peak)
            radius = self.peak_radius
            dist_to_center = np.linalg.norm(choice - peak)
            dist_to_edge = max(0.0, dist_to_center - radius)
            return dist_to_edge
        
        def prob_global_max(agents):
            '''
            Returns the probability of agents being near the reward peak
            '''
            near_peaks = [dist_to_peak(a) <= 1.0 for a in agents] # Distance on grid
            return np.mean(near_peaks)

        def is_local_max(agent, threshold=0.55):
            '''
            Returns whether an individual agent is at a local max
            '''
            if (
                not hasattr(self, 'reward_peak')
                or self.reward_peak is None
                or not hasattr(self, 'moat_radius')
                or self.moat_radius is None
            ):
                return False

            choice = np.array(agent.last_choice)
            peak = np.array(self.reward_peak)
            dist_to_center = np.linalg.norm(choice - peak)
            outside_moat = dist_to_center > self.moat_radius
            return outside_moat and (agent.last_reward + 0.5 > threshold)
        
        def prob_local_max(agents):
            '''
            Returns the probability of agents being at a local max
            '''
            local_maxes = [1.0 if is_local_max(agent) else 0.0 for agent in agents]
            return np.mean(local_maxes)
        
        def most_common_choice(agents):
            '''
            Returns the most common choice among agents in the last step
            '''
            choices = [a.last_choice for a in agents]
            if not choices:
                return None
            return pd.Series(choices).mode()[0]

        self.datacollector = DataCollector(
            model_reporters={
                # "avg_cumulative_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) + 0.5 * m.steps,
                "avg_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) / m.steps + 0.5,
                # "prob_global_max": lambda m: prob_global_max(m.grid.agents),
                # "prob_local_max": lambda m: prob_local_max(m.grid.agents),
                # "most_common_choice": lambda m: most_common_choice(m.grid.agents)
            },
            agent_reporters={
                "policy": lambda a: a.policy_grid,
                "value": lambda a: a.ucb_grid,
                "choice": lambda a: a.last_choice,
                "reward": lambda a: a.last_reward + 0.5,
                # "cumulative_reward": lambda a: a.total_reward + 0.5 * a.model.steps,
                'distance_to_peak': lambda a: dist_to_peak(a),
                "global_max": lambda a: dist_to_peak(a) <= 1.0,
                "local_max": lambda a: is_local_max(a, threshold=0.55), 
            },
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

if __name__ == "__main__":
    m = SocialGPModel(n=5, grid_size=11, alpha=0.5, seed=42)
    for _ in range(20):
        m.step()
    
    results = m.datacollector.get_model_vars_dataframe()
    agent_results = m.datacollector.get_agent_vars_dataframe()
