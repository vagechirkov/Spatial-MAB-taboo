from collections.abc import Iterable

import numpy as np


def find_local_peak_coordinates(reward_map: np.ndarray) -> list[tuple[int, int]]:
    """Return coordinates of local maxima in a 2D reward map (8-neighborhood)."""
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
