import numpy as np
import pytest

from service.route_planner import astar_path, compress_collinear, route_candidates, research_transit_sensitivity


def test_research_transit_sensitivity_is_hypothetical_and_abstains_on_high_ice():
    field = np.array([[0.0, 0.0, 0.0]], dtype=float)
    estimate = research_transit_sensitivity(field, [(0, 0), (0, 1), (0, 2)], 1852, 1852, 10, 3)
    assert estimate == {"status": "hypothetical", "central_hours": 0.2,
                        "fast_hours": 0.17, "slow_hours": 0.25}
    field[0, 1] = 1.0
    field[0, 2] = 1.0
    assert research_transit_sensitivity(field, [(0, 0), (0, 1), (0, 2)], 1852, 1852, 10, 3)["status"] == "unavailable"


def test_low_ice_objective_can_choose_longer_path_around_dense_observed_ice():
    field = np.full((7, 7), 0.1, dtype=np.float32)
    field[1:7, 3] = 0.9
    valid = np.ones_like(field, dtype=bool)
    shortest, tradeoff, low_ice = route_candidates(field, valid, (3, 0), (3, 6), 1000, 1000)
    shortest_path = shortest["path_cells"]
    low_path = low_ice["path_cells"]
    assert any(col == 3 and row >= 1 for row, col in shortest_path)
    assert all(row == 0 or col != 3 for row, col in low_path)
    assert low_ice["metrics"]["grid_distance_km"] > shortest["metrics"]["grid_distance_km"]
    assert low_ice["metrics"]["distance_weighted_mean_sic_fraction"] < shortest["metrics"]["distance_weighted_mean_sic_fraction"]
    assert [candidate["metrics"]["objective"]["concentration_weight"] for candidate in (shortest, tradeoff, low_ice)] == [0, 4, 12]


def test_invalid_cells_are_impassable_and_diagonal_cannot_cut_missing_corner():
    field = np.full((2, 2), 0.2, dtype=np.float32)
    valid = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError, match="No continuous valid-data path"):
        astar_path(field, valid, (0, 0), (1, 1), 1000, 1000, 0)
    with pytest.raises(ValueError, match="without valid observed SIC"):
        astar_path(field, np.array([[True, False], [True, True]]), (0, 1), (1, 1), 1000, 1000, 0)


def test_search_is_bounded_and_reports_clear_failure():
    field = np.full((9, 9), 0.1, dtype=np.float32)
    with pytest.raises(ValueError, match="expansion limit"):
        astar_path(field, np.ones_like(field, dtype=bool), (0, 0), (8, 8),
                   1000, 1000, 0, expansion_limit=1)


def test_collinear_compression_preserves_turns_and_single_cell_path():
    path = [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 1)]
    assert compress_collinear(path) == [(0, 0), (2, 2), (4, 2), (5, 1)]
    assert compress_collinear([(2, 2)]) == [(2, 2)]
