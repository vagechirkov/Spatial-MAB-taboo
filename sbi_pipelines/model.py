import mesa
import numpy as np
import networkx as nx
from mesa import DataCollector
from mesa.discrete_space import Network
from abm.agent import SocialGPAgent
from abm.rewards import make_parent_and_children_cholesky

def last_choice_distance_private(agent):
    x = np.asarray(agent.X_observations)
    if len(x) < 2:
        return 0.0
    return float(np.linalg.norm(x[-1] - x[-2], axis=-1))

def last_choice_distance_social(agent):
    X_soc, _ = agent._gather_social_info()
    if len(X_soc) < 1 or len(X_soc[0]) < 1:
        return 0.0
    social_last_choices = np.array([_x_soc[-1] for _x_soc in X_soc])
    cur = np.asarray(agent.X_observations[-1])
    return float(np.mean(np.linalg.norm(cur - social_last_choices, axis=-1)))

def nearest_choice_distance_private(agent):
    x = np.asarray(agent.X_observations)
    if len(x) < 2:
        return 0.0
    return float(np.min(np.linalg.norm(x[-1] - x[:-1], axis=-1)))

def avg_choice_distance_private(agent):
    x = np.asarray(agent.X_observations)
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(x[-1] - x[:-1], axis=-1)))

def nearest_choice_distance_social(agent):
    X_soc, _ = agent._gather_social_info()
    if len(X_soc) < 1 or len(X_soc[0]) < 1:
        return 0.0
    social_choices = np.vstack(X_soc)
    cur = np.asarray(agent.X_observations[-1])
    return float(np.min(np.linalg.norm(cur - social_choices, axis=-1)))

def avg_choice_distance_social(agent):
    X_soc, _ = agent._gather_social_info()
    if len(X_soc) < 1 or len(X_soc[0]) < 1:
        return 0.0
    social_choices = np.vstack(X_soc)
    cur = np.asarray(agent.X_observations[-1])
    return float(np.mean(np.linalg.norm(cur - social_choices, axis=-1)))

def neg_log_likelihood(agent):
    if agent.model.steps <= 1: # Assuming step 1 is random
        return 0.0
    return -np.log(agent.policy[agent.meshgrid_dict[agent.X_observations[-1]]] + 1e-12)


class SBIModel(mesa.Model):
    def __init__(
            self,
            child_maps,
            rng=None,
            n: int = 4,
            grid_size: int = 11,
            length_scale: float = 1.11,
            beta: float = 0.5,
            tau: float = 0.03,
            alpha: float = 0.6,
            observation_noise: float = 0.0001,
            reward_noise_sd: float = 0.0,
            individual_choices = None,
            individual_rewards = None,
            fitting_mode: bool = False
    ):
        super().__init__(rng=rng)
        
        self.num_agents = n
        self.grid_size = grid_size
        self.reward_noise_sd = reward_noise_sd
        self.fitting_mode = fitting_mode
        self.individual_choices = individual_choices
        self.individual_rewards = individual_rewards
        
        G = nx.complete_graph(n)
        self.grid = Network(G)

        length_scale_by_agent = [length_scale] * n
        observation_noise_by_agent = [observation_noise] * n
        beta_by_agent = [beta] * n
        tau_by_agent = [tau] * n
        alpha_by_agent = [alpha] * n

        SocialGPAgent.create_agents(
            self,
            n,
            cell=self.grid.all_cells.cells,
            reward_environment=child_maps,
            length_scale_private=length_scale_by_agent,
            length_scale_social=length_scale_by_agent,
            observation_noise_private=observation_noise_by_agent,
            observation_noise_social=observation_noise_by_agent,
            beta_private=beta_by_agent,
            beta_social=beta_by_agent,
            tau=tau_by_agent,
            alpha=alpha_by_agent,
        )
        self.agents_list = list(self.agents)

        self.datacollector = DataCollector(
            model_reporters={
                "avg_reward": lambda m: np.mean([a.last_reward for a in m.agents_list]),
                "avg_cumulative_reward": lambda m: np.mean([a.total_reward for a in m.agents_list]),
                "last_choice_distance_private": lambda m: np.mean([last_choice_distance_private(a) for a in m.agents_list]),
                "last_choice_distance_social": lambda m: np.mean([last_choice_distance_social(a) for a in m.agents_list]),
                "nearest_choice_distance_private": lambda m: np.mean([nearest_choice_distance_private(a) for a in m.agents_list]),
                "avg_choice_distance_private": lambda m: np.mean([avg_choice_distance_private(a) for a in m.agents_list]),
                "nearest_choice_distance_social": lambda m: np.mean([nearest_choice_distance_social(a) for a in m.agents_list]),
                "avg_choice_distance_social": lambda m: np.mean([avg_choice_distance_social(a) for a in m.agents_list]),
                "nll": lambda m: np.mean([neg_log_likelihood(a) for a in m.agents_list])
            },
            agent_reporters={
                "choice": lambda a: a.last_choice,
                "reward": lambda a: a.last_reward
            }
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
