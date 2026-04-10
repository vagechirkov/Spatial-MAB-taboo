from collections.abc import Sequence

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
from .reporter_helpers import (
    find_global_peak_coordinates,
    find_local_peak_coordinates,
    min_distance_to_points,
    normalize_reporter_selection,
)

def _build_network(network_type, n):
    if network_type == "fully_connected":
        return nx.complete_graph(n)
    raise ValueError(f"Unknown network_type '{network_type}'")


def as_batch_fixed(value):
    """
    Wrap a value so mesa.batch_run treats it as a single fixed value.

    Mesa interprets iterables as sweep axes. Use this for per-agent vectors,
    e.g. `beta=as_batch_fixed([0.05, 0.10])`.
    """
    return [value]


def _unwrap_singleton_sequence(value):
    """Unwrap one-level singleton wrappers used to bypass mesa.batch_run sweeps."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        value = value.tolist()

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 1
    ):
        inner_value = value[0]
        if isinstance(inner_value, (list, tuple, np.ndarray)):
            return _unwrap_singleton_sequence(inner_value)

    return value


def _normalize_agent_parameter(value, n: int, parameter_name: str) -> list[float]:
    """
    Normalize agent-level parameters to length `n`.

    Accepted input formats:
    - scalar: shared across all agents
    - length-1 sequence: broadcast to all agents
    - length-n sequence: per-agent values
    """
    value = _unwrap_singleton_sequence(value)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [float(value.item())] * n
        values = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        try:
            scalar_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{parameter_name} must be numeric or a numeric sequence. "
                f"Got: {value!r}"
            ) from exc
        return [scalar_value] * n

    if len(values) == 0:
        raise ValueError(f"{parameter_name} cannot be empty")

    if len(values) == 1:
        values = values * n
    elif len(values) != n:
        raise ValueError(
            f"{parameter_name} must be a scalar, length-1 sequence, "
            f"or length-{n} sequence. Got length {len(values)}"
        )

    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{parameter_name} must contain numeric values. "
            f"Got: {values!r}"
        ) from exc

class SocialGPModel(mesa.Model):
    """
    Bare-bones GP-Value Shaping model on correlated landscapes.

    Agent parameters (`length_scale`, `observation_noise`, `beta`, `tau`, `alpha`)
    can be provided as scalars (shared), length-1 sequences (broadcast), or
    length-n sequences (heterogeneous values per agent).

    For mesa.batch_run, wrap heterogeneous vectors with `as_batch_fixed(...)`
    so they are passed as fixed values instead of sweep dimensions.
    """
    def __init__(
        self,
        *,
        n: int = 4,
        grid_size: int = 11,
        length_scale: float | Sequence[float] | np.ndarray = 1.11, # for agents
        observation_noise: float | Sequence[float] | np.ndarray = 0.01,
        beta: float | Sequence[float] | np.ndarray = 0.5,
        tau: float | Sequence[float] | np.ndarray = 0.01,
        alpha: float | Sequence[float] | np.ndarray = 0.6,
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

        length_scale_by_agent = _normalize_agent_parameter(length_scale, n, "length_scale")
        observation_noise_by_agent = _normalize_agent_parameter(
            observation_noise,
            n,
            "observation_noise",
        )
        beta_by_agent = _normalize_agent_parameter(beta, n, "beta")
        tau_by_agent = _normalize_agent_parameter(tau, n, "tau")
        alpha_by_agent = _normalize_agent_parameter(alpha, n, "alpha")

        self.agent_hyperparameters = {
            "length_scale": length_scale_by_agent,
            "observation_noise": observation_noise_by_agent,
            "beta": beta_by_agent,
            "tau": tau_by_agent,
            "alpha": alpha_by_agent,
        }

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

        # Reporter radii are in grid-distance units.
        self.global_peak_report_radius = 1.0
        self.local_peak_report_radius  = 1.0

        # Local peak locations are static because reward maps are static.
        self.local_peak_coordinates = {
            agent.unique_id: find_local_peak_coordinates(agent.reward_environment)
            for agent in self.grid.agents
        }
        self.global_peak_coordinates = {
            agent.unique_id: find_global_peak_coordinates(agent.reward_environment)
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
            return min_distance_to_points(choice, peak_coords)

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
            return min_distance_to_points(choice, peak_coords)

        def distance_to_nearest_local_peak(agent):
            """Distance from last choice to nearest detected local peak."""
            return distance_choice_to_nearest_local_peak(agent, agent.last_choice)

        def is_at_local_max(agent):
            """
            1 if the last choice is near a detected local peak.

            Global max has priority: if an agent is classified as global, local is forced to 0.
            """
            if is_choice_at_global_max(agent, agent.last_choice):
                return 0
            return int(
                distance_choice_to_nearest_local_peak(agent, agent.last_choice) <= self.local_peak_report_radius
            )

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
            requested_model_reporters = normalize_reporter_selection(
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
                "policy": lambda a: a.policy_grid,
                "value": lambda a: a.ucb_grid,
                "choice": lambda a: a.last_choice,
                "global_max": lambda a: is_at_global_max(a),
                "local_max": lambda a: is_at_local_max(a),
                "no_max": lambda a: is_not_at_any_max(a),
                "cumulative_reward": lambda a: a.total_reward + 0.5 * a.model.steps,
                "distance_to_global_peak": lambda a: distance_to_global_peak_region(a),
                "distance_to_local_peak": lambda a: distance_to_nearest_local_peak(a),
            }

            if agent_reporters_to_collect is None:
                requested_agent_reporters = ["reward", "global_max", "local_max", "no_max", "cumulative_reward"]
            else:
                requested_agent_reporters = normalize_reporter_selection(
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
