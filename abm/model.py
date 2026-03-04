from collections import deque
from collections.abc import Iterable

import mesa
import networkx as nx
import numpy as np

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


def _find_local_peak_coordinates(reward_map: np.ndarray) -> list[tuple[int, int]]:
    """
    Return coordinates of local maxima in a 2D reward map (8-neighborhood).

    The check uses >= comparisons so flat plateaus are treated as peak regions.
    """
    n_rows, n_cols = reward_map.shape
    padded_map = np.pad(reward_map, 1, mode="constant", constant_values=-np.inf)
    center = padded_map[1:-1, 1:-1]

    is_local_peak = np.ones_like(reward_map, dtype=bool)
    for row_shift in (-1, 0, 1):
        for col_shift in (-1, 0, 1):
            if row_shift == 0 and col_shift == 0:
                continue
            neighbor = padded_map[
                1 + row_shift : 1 + row_shift + n_rows,
                1 + col_shift : 1 + col_shift + n_cols,
            ]
            is_local_peak &= center >= neighbor

    return [tuple(coord) for coord in np.argwhere(is_local_peak)]


def _find_global_peak_coordinates(reward_map: np.ndarray) -> list[tuple[int, int]]:
    """Return coordinates of global maxima in a 2D reward map."""
    max_value = np.max(reward_map)
    is_global_peak = np.isclose(reward_map, max_value)
    return [tuple(coord) for coord in np.argwhere(is_global_peak)]


def _min_distance_to_points(
    point: tuple[int, int] | None,
    points: list[tuple[int, int]],
) -> float:
    """Return the minimum Euclidean distance from `point` to a set of grid points."""
    if point is None or len(points) == 0:
        return np.inf

    point_arr = np.asarray(point, dtype=float)
    points_arr = np.asarray(points, dtype=float)
    return float(np.min(np.linalg.norm(points_arr - point_arr, axis=1)))


def _normalize_reporter_selection(
    selection: str | Iterable[str] | None,
    parameter_name: str,
) -> list[str] | None:
    """
    Normalize reporter selection arguments to a list of reporter names.

    Supports both direct model construction and mesa.batch_run behavior where
    a single-item list in the parameter grid may arrive as a scalar string.
    """
    if selection is None:
        return None

    if isinstance(selection, str):
        return [selection]

    selection_list = list(selection)

    # Allow one extra wrapper level (e.g., [["avg_reward", "cumulative_reward"]]).
    if (
        len(selection_list) == 1
        and isinstance(selection_list[0], Iterable)
        and not isinstance(selection_list[0], str)
    ):
        selection_list = list(selection_list[0])

    if not all(isinstance(item, str) for item in selection_list):
        raise TypeError(
            f"{parameter_name} must be a string or a sequence of strings. "
            f"Got: {selection!r}"
        )

    return selection_list

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
        summary_window: int = 5,
        collect_agent_reporters: bool = True,
        model_reporters_to_collect: list[str] | tuple[str, ...] | None = None,
        agent_reporters_to_collect: list[str] | tuple[str, ...] | None = None,
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
        if summary_window <= 0:
            raise ValueError("summary_window must be a positive integer")
        self.summary_window = int(summary_window)

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
            if 'sigma_inner' in reward_env_params:
                self.peak_radius = reward_env_params['sigma_inner']
            else:
                # NOTE: assumes reward environment generation maintains this ratio.  Must change if we change reward env generation logic.
                self.peak_radius = reward_env_params['length_scale'] // 2.0

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

        # Reporter radii are in grid-distance units.
        self.global_peak_report_radius = 1.0
        self.local_peak_report_radius  = 3.0

        # Local peak locations are static because reward maps are static.
        self.local_peak_coordinates = {
            agent.unique_id: _find_local_peak_coordinates(agent.reward_environment)
            for agent in self.grid.agents
        }
        self.global_peak_coordinates = {
            agent.unique_id: _find_global_peak_coordinates(agent.reward_environment)
            for agent in self.grid.agents
        }

        def distance_choice_to_global_peak_region(agent, choice):
            """Distance from a choice coordinate to the global-peak region boundary."""
            if choice is None:
                return np.inf

            has_global_peak_region = (
                hasattr(self, 'reward_peak')
                and self.reward_peak is not None
                and hasattr(self, 'peak_radius')
                and self.peak_radius is not None
            )

            if has_global_peak_region:
                choice_arr = np.array(choice)
                peak = np.array(self.reward_peak)
                radius = self.peak_radius
                dist_to_center = np.linalg.norm(choice_arr - peak)
                return max(0.0, dist_to_center - radius)

            peak_coords = self.global_peak_coordinates.get(agent.unique_id, [])
            return _min_distance_to_points(choice, peak_coords)

        def distance_to_global_peak_region(agent):
            """Distance from last choice to the global-peak region boundary."""
            return distance_choice_to_global_peak_region(agent, agent.last_choice)

        def is_choice_at_global_max(agent, choice):
            """1 if a choice is in the global-peak neighborhood, else 0."""
            return int(
                distance_choice_to_global_peak_region(agent, choice)
                <= self.global_peak_report_radius
            )

        def is_at_global_max(agent):
            """1 if the last choice is in the global-peak neighborhood, else 0."""
            return is_choice_at_global_max(agent, agent.last_choice)

        def distance_choice_to_nearest_local_peak(agent, choice):
            """Distance from a choice coordinate to the nearest detected local peak."""
            peak_coords = self.local_peak_coordinates.get(agent.unique_id, [])
            return _min_distance_to_points(choice, peak_coords)

        def distance_to_nearest_local_peak(agent):
            """Distance from last choice to nearest detected local peak."""
            return distance_choice_to_nearest_local_peak(agent, agent.last_choice)

        def is_choice_at_local_max(agent, choice):
            """1 if a choice is near a local peak and not near the global max."""
            if is_choice_at_global_max(agent, choice):
                return 0
            return int(
                distance_choice_to_nearest_local_peak(agent, choice) <= self.local_peak_report_radius
            )

        def is_at_local_max(agent):
            """
            1 if the last choice is near a detected local peak.

            Global max has priority: if an agent is classified as global, local is forced to 0.
            """
            return is_choice_at_local_max(agent, agent.last_choice)

        def is_not_at_any_max(agent):
            """1 if the last choice is not near any global/local peak, else 0."""
            return int((is_at_global_max(agent) == 0) and (is_at_local_max(agent) == 0))

        available_model_reporters = {
            "mean_cumulative_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) + 0.5 * m.steps,
            "mean_reward": lambda m: np.mean([a.total_reward for a in m.grid.agents]) / m.steps + 0.5,
        }

        if model_reporters_to_collect is None:
            model_reporters = dict(available_model_reporters)
        else:
            requested_model_reporters = _normalize_reporter_selection(
                model_reporters_to_collect,
                "model_reporters_to_collect",
            )
            unknown_model_reporters = sorted(
                set(requested_model_reporters) - set(available_model_reporters)
            )
            if unknown_model_reporters:
                raise ValueError(
                    "Unknown model reporters requested: "
                    f"{unknown_model_reporters}. "
                    f"Available: {sorted(available_model_reporters)}"
                )
            model_reporters = {
                reporter_name: available_model_reporters[reporter_name]
                for reporter_name in requested_model_reporters
            }

        agent_reporters = {}
        if collect_agent_reporters:
            available_agent_reporters = {
                "reward": lambda a: a.last_reward + 0.5,
                "global_max": lambda a: is_at_global_max(a),
                "local_max": lambda a: is_at_local_max(a),
                "no_max": lambda a: is_not_at_any_max(a),
                "cumulative_reward": lambda a: a.total_reward + 0.5 * a.model.steps,
            }

            if agent_reporters_to_collect is None:
                requested_agent_reporters = ["reward", "global_max", "local_max", "no_max", "cumulative_reward"]
            else:
                requested_agent_reporters = _normalize_reporter_selection(
                    agent_reporters_to_collect,
                    "agent_reporters_to_collect",
                )

            unknown_agent_reporters = sorted(
                set(requested_agent_reporters) - set(available_agent_reporters)
            )
            if unknown_agent_reporters:
                raise ValueError(
                    "Unknown agent reporters requested: "
                    f"{unknown_agent_reporters}. "
                    f"Available: {sorted(available_agent_reporters)}"
                )
            agent_reporters = {
                reporter_name: available_agent_reporters[reporter_name]
                for reporter_name in requested_agent_reporters
            }

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
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
