import mesa
import networkx as nx
import numpy as np
import pandas as pd

from .agent import SocialGPAgent
from mesa import DataCollector
from mesa.discrete_space import Network
from .rewards import (
    make_parent_and_children_cholesky,
    build_corr_matrix_bare_bones,
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
        length_scale: float = 1.11,
        observation_noise: float = 0.01,
        beta: float = 0.5,
        tau: float = 0.01,
        alpha: float = 0.6,
        network_type: str = "fully_connected",
        reward_noise_sd : float = 0.001,
        seed: int | None = None,
        corr_matrix: np.ndarray | None = None,
    ):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid_size = grid_size
        self.reward_noise_sd = reward_noise_sd

        # Generate correlated reward environments
        if corr_matrix is None:
            corr_matrix = build_corr_matrix_bare_bones(n + 1)
        
        parent, child_maps = make_parent_and_children_cholesky(
            rng=self.rng,
            grid_size=grid_size,
            n_children=n,
            length_scale=2.0,
            corr_matrix=corr_matrix,
        )

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

        self.datacollector = DataCollector(
            model_reporters={
                "avg_cumulative_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) + 0.5 * m.steps,
                "avg_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) / m.steps + 0.5,
            },
            agent_reporters={
                "choice": lambda a: a.last_choice,
                "reward": lambda a: a.last_reward + 0.5,
                "cumulative_reward": lambda a: a.total_reward + 0.5 * a.model.steps,
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
