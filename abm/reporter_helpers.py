from collections.abc import Iterable

import numpy as np


DEFAULT_TOP_LOCAL_PEAKS = 3


def find_local_peak_coordinates(
    reward_map: np.ndarray,
    top_n: int = DEFAULT_TOP_LOCAL_PEAKS,
) -> list[tuple[int, int]]:
    """
    Return coordinates of the top local maxima in a 2D reward map (8-neighborhood).

    Peaks are sorted by reward value (descending). The global maximum is excluded,
    then at most `top_n` remaining local peaks are returned.
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

    peaks = np.argwhere(is_local_peak)
    if len(peaks) == 0:
        return []

    max_value = np.max(reward_map)
    peak_values = reward_map[peaks[:, 0], peaks[:, 1]]
    local_peaks = peaks[~np.isclose(peak_values, max_value)]
    if len(local_peaks) == 0:
        return []

    local_values = reward_map[local_peaks[:, 0], local_peaks[:, 1]]
    top_peaks = local_peaks[np.argsort(-local_values)][:top_n]
    return [(int(row), int(col)) for row, col in top_peaks]


def find_global_peak_coordinates(reward_map: np.ndarray) -> list[tuple[int, int]]:
    """Return coordinates of global maxima in a 2D reward map."""
    max_value = np.max(reward_map)
    is_global_peak = np.isclose(reward_map, max_value)
    return [tuple(coord) for coord in np.argwhere(is_global_peak)]


def min_distance_to_points(
    point: tuple[int, int] | None,
    points: list[tuple[int, int]],
) -> float:
    """Return the minimum Euclidean distance from `point` to a set of grid points."""
    if point is None or len(points) == 0:
        return np.inf

    point_arr = np.asarray(point, dtype=float)
    points_arr = np.asarray(points, dtype=float)
    return float(np.min(np.linalg.norm(points_arr - point_arr, axis=1)))


def make_peak_agent_reporters(
    global_peak_coordinates: dict[int, list[tuple[int, int]]],
    local_peak_coordinates: dict[int, list[tuple[int, int]]],
    *,
    global_radius: float = 1.0,
    local_radius: float = 3.0,
) -> dict[str, object]:
    """
    Build agent reporters for global/local peak proximity on static reward maps.

    Local peaks are expected to be the top local maxima excluding the global maximum
    (see `find_local_peak_coordinates`).
    """
    def _global_coords(agent):
        return global_peak_coordinates.get(agent.unique_id, [])

    def _local_coords(agent):
        return local_peak_coordinates.get(agent.unique_id, [])

    def _distance_to_peaks(agent, peak_coords):
        return min_distance_to_points(agent.last_choice, peak_coords)

    def _is_near_peaks(agent, peak_coords, radius):
        return _distance_to_peaks(agent, peak_coords) <= radius

    def global_max(agent):
        return int(_is_near_peaks(agent, _global_coords(agent), global_radius))

    def local_max(agent):
        if _is_near_peaks(agent, _global_coords(agent), global_radius):
            return 0
        return int(_is_near_peaks(agent, _local_coords(agent), local_radius))

    def no_max(agent):
        return int(global_max(agent) == 0 and local_max(agent) == 0)

    return {
        "global_max": global_max,
        "local_max": local_max,
        "no_max": no_max,
        "distance_to_global_peak": lambda agent: _distance_to_peaks(agent, _global_coords(agent)),
        "distance_to_local_peak": lambda agent: _distance_to_peaks(agent, _local_coords(agent)),
    }


def normalize_reporter_selection(
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
