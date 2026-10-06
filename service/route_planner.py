"""Constrained, transparent path search over a single observed SIC raster.

This is a research comparison primitive, not a vessel-performance or safety model.
"""
from __future__ import annotations

from heapq import heappop, heappush
import math
from typing import Any

import numpy as np


NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1))


def research_transit_sensitivity(field: np.ndarray, path: list[tuple[int, int]],
                                 pixel_x_m: float, pixel_y_m: float,
                                 open_water_knots: float, ice_affected_knots: float) -> dict[str, Any]:
    """Hypothetical travel-time sensitivity, never a ship ETA or passability test.

    This intentionally simple SIC-to-speed relation is an assumption, not a
    fitted vessel performance curve. The ±20% speed sweep is not a confidence
    interval. Abstain on high-concentration cells rather than imply passage.
    """
    sic = np.asarray(field, dtype=np.float64)
    if any(not math.isfinite(value) or value <= 0 for value in
           (pixel_x_m, pixel_y_m, open_water_knots, ice_affected_knots)):
        raise ValueError("Pixel spacing and assumed speeds must be positive and finite")
    if ice_affected_knots > open_water_knots:
        raise ValueError("Assumed ice-affected speed cannot exceed open-water speed")
    if len(path) < 2:
        return {"status": "unavailable", "reason": "A path needs at least two grid cells."}
    if any(float(sic[cell]) >= 0.7 for cell in path):
        return {"status": "unavailable", "reason": "Observed SIC reaches 70% on this path; generic speed assumptions are withheld."}
    hours = 0.0
    for first, second in zip(path, path[1:]):
        concentration = float((sic[first] + sic[second]) / 2.0)
        if not math.isfinite(concentration) or concentration < 0 or concentration > 1:
            return {"status": "unavailable", "reason": "The path crosses invalid observed ice data."}
        distance_nm = math.hypot((second[1] - first[1]) * pixel_x_m,
                                 (second[0] - first[0]) * pixel_y_m) / 1852.0
        assumed_knots = open_water_knots + (ice_affected_knots - open_water_knots) * concentration
        hours += distance_nm / assumed_knots
    return {"status": "hypothetical", "central_hours": round(hours, 2),
            "fast_hours": round(hours / 1.2, 2), "slow_hours": round(hours / 0.8, 2)}


def astar_path(field: np.ndarray, valid: np.ndarray, start: tuple[int, int],
               goal: tuple[int, int], pixel_x_m: float, pixel_y_m: float,
               concentration_weight: float, expansion_limit: int = 40_000) -> tuple[list[tuple[int, int]], int]:
    """Find a valid-cell path minimizing projected distance plus SIC penalty.

    Edge cost is projected grid distance in km multiplied by
    ``1 + weight * (midpoint SIC squared)``. Missing cells are impassable.
    Diagonal steps cannot cut across a missing-data corner.
    """
    sic = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).copy()
    if sic.ndim != 2 or sic.shape != mask.shape or min(sic.shape) < 1:
        raise ValueError("SIC field and valid mask must be matching non-empty 2-D arrays")
    if (not math.isfinite(pixel_x_m) or not math.isfinite(pixel_y_m)
            or pixel_x_m <= 0 or pixel_y_m <= 0
            or not math.isfinite(concentration_weight) or concentration_weight < 0):
        raise ValueError("Pixel spacing and concentration weight must be finite and non-negative")
    mask &= np.isfinite(sic) & (sic >= 0) & (sic <= 1)
    height, width = sic.shape
    for point, label in ((start, "origin"), (goal, "destination")):
        row, col = point
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label} is outside the observed grid")
        if not mask[row, col]:
            raise ValueError(f"{label} maps to a cell without valid observed SIC")
    if start == goal:
        return [start], 0

    def heuristic(row: int, col: int) -> float:
        return math.hypot((goal[1] - col) * pixel_x_m,
                          (goal[0] - row) * pixel_y_m) / 1000.0

    best: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    serial = 0
    frontier: list[tuple[float, int, float, tuple[int, int]]] = []
    heappush(frontier, (heuristic(*start), serial, 0.0, start))
    expanded = 0
    while frontier:
        _, _, cost_so_far, current = heappop(frontier)
        if cost_so_far > best.get(current, math.inf):
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path, expanded
        expanded += 1
        if expanded > expansion_limit:
            raise ValueError("Route search exceeded its bounded grid-cell expansion limit")
        row, col = current
        for drow, dcol in NEIGHBORS:
            next_row, next_col = row + drow, col + dcol
            if not (0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col]):
                continue
            if drow and dcol and (not mask[row, next_col] or not mask[next_row, col]):
                continue
            distance_m = math.hypot(dcol * pixel_x_m, drow * pixel_y_m)
            midpoint_sic = (sic[row, col] + sic[next_row, next_col]) / 2.0
            edge_cost = distance_m / 1000.0 * (1.0 + concentration_weight * midpoint_sic ** 2)
            candidate = cost_so_far + edge_cost
            neighbor = (next_row, next_col)
            if candidate < best.get(neighbor, math.inf):
                best[neighbor] = candidate
                parent[neighbor] = current
                serial += 1
                heappush(frontier, (candidate + heuristic(next_row, next_col), serial,
                                    candidate, neighbor))
    raise ValueError("No continuous valid-data path connects the requested endpoints")


def compress_collinear(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep endpoints and turns while preserving the exact raster path geometry."""
    if len(path) <= 2:
        return path.copy()
    result = [path[0]]
    previous_step = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for index in range(1, len(path) - 1):
        step = (path[index + 1][0] - path[index][0],
                path[index + 1][1] - path[index][1])
        if step != previous_step:
            result.append(path[index])
        previous_step = step
    result.append(path[-1])
    return result


def route_metrics(field: np.ndarray, path: list[tuple[int, int]],
                  pixel_x_m: float, pixel_y_m: float,
                  concentration_weight: float, expanded_cells: int) -> dict[str, Any]:
    """Summarize exactly the cells/edges used by the optimizer."""
    sic = np.asarray(field, dtype=np.float64)
    distances_km = []
    midpoint_sic = []
    objective = 0.0
    for first, second in zip(path, path[1:]):
        distance = math.hypot((second[1] - first[1]) * pixel_x_m,
                              (second[0] - first[0]) * pixel_y_m) / 1000.0
        concentration = (sic[first] + sic[second]) / 2.0
        distances_km.append(distance)
        midpoint_sic.append(concentration)
        objective += distance * (1.0 + concentration_weight * concentration ** 2)
    route_distance = float(sum(distances_km))
    if distances_km:
        mean_sic = float(np.average(midpoint_sic, weights=distances_km))
        max_sic = float(np.max(sic[[p[0] for p in path], [p[1] for p in path]]))
        p95_sic = float(np.percentile(sic[[p[0] for p in path], [p[1] for p in path]], 95))
    else:
        mean_sic = max_sic = p95_sic = float(sic[path[0]])
    return {
        "grid_distance_km": route_distance,
        "distance_weighted_mean_sic_fraction": mean_sic,
        "route_cell_p95_sic_fraction": p95_sic,
        "route_cell_maximum_sic_fraction": max_sic,
        "sic_distance_integral_fraction_km": mean_sic * route_distance,
        "objective": {"projected_distance_equivalent_km": objective,
                      "concentration_weight": concentration_weight,
                      "formula": "sum(edge projected km * (1 + weight * midpoint SIC fraction squared))"},
        "search": {"expanded_grid_cells": expanded_cells,
                   "raster_path_cell_count": len(path),
                   "display_vertex_count": len(compress_collinear(path))},
    }


def route_candidates(field: np.ndarray, valid: np.ndarray,
                     origin: tuple[int, int], destination: tuple[int, int],
                     pixel_x_m: float, pixel_y_m: float) -> list[dict[str, Any]]:
    """Return three transparent objectives; their weights are not vessel calibrated."""
    objectives = (
        ("shortest_grid_distance", 0.0,
         "Minimizes projected grid distance; observed SIC is reported but not penalized."),
        ("distance_ice_tradeoff", 4.0,
         "Trades projected distance against squared observed SIC using the displayed weight 4."),
        ("lower_observed_ice", 12.0,
         "Penalizes squared observed SIC more strongly using the displayed weight 12."),
    )
    reports = []
    previous_paths: dict[tuple[tuple[int, int], ...], str] = {}
    for name, weight, explanation in objectives:
        path, expanded = astar_path(field, valid, origin, destination,
                                    pixel_x_m, pixel_y_m, weight)
        key = tuple(path)
        reports.append({"candidate_id": name, "explanation": explanation,
                        "distinct_from": previous_paths.get(key),
                        "metrics": route_metrics(field, path, pixel_x_m, pixel_y_m,
                                                 weight, expanded),
                        "path_cells": path,
                        "display_cells": compress_collinear(path)})
        previous_paths.setdefault(key, name)
    return reports
