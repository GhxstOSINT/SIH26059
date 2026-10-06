import numpy as np
import pytest

from ml.evaluate_route_corridors import aggregate, score_day, validate_routes


def test_scores_model_and_persistence_on_identical_valid_points():
    observed = np.array([[.2, .8], [.5, .7]])
    baseline = np.array([[.3, .8], [.5, .7]])
    model = np.array([[.2, np.nan], [.5, .7]])
    day = score_day(model, baseline, observed, np.array([0, 0]), np.array([0, 1]), np.array([True, True]))
    assert day["common_valid_points"] == 1
    assert day["coverage_fraction"] == .5
    summary = aggregate([day])
    assert summary["model_mae"] == 0
    assert summary["persistence_mae"] == pytest.approx(.1)


def test_route_file_must_be_frozen_research_scope():
    with pytest.raises(ValueError, match="research_only"):
        validate_routes({"schema_version": 1, "scope": "operational", "routes": []})
    with pytest.raises(ValueError, match="unique"):
        validate_routes({"schema_version": 1, "scope": "research_only", "routes": [
            {"id": "x", "coordinates": [[-60, -66], [-59, -66]]},
            {"id": "x", "coordinates": [[-60, -66], [-59, -66]]}]})
