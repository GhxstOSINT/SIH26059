from datetime import date
import json

import numpy as np
import pytest

from ml.research_transect import load_research_transect, sample_transect


def _grid():
    lat = np.arange(-64.0, -60.9, 0.1)
    lon = np.arange(-53.0, -35.9, 0.1)
    shape = (len(lat), len(lon))
    return lat, lon, np.full(shape, 0.5), np.full(shape, 0.4), np.full(shape, 0.1), np.zeros(shape)


def test_research_transect_requires_post_freeze_bulletin():
    route, digest = load_research_transect()
    lat, lon, forecast, truth, uncertainty, flags = _grid()
    earlier = sample_transect(forecast, truth, uncertainty, flags, lat, lon, route, digest,
                              bulletin_date=date(2026, 10, 3))
    assert earlier["evaluation_status"] == "not_eligible_pre_freeze"
    assert "forecast_mae_fraction" not in earlier
    later = sample_transect(forecast, truth, uncertainty, flags, lat, lon, route, digest,
                            bulletin_date=date(2026, 10, 4), persistence=np.full(truth.shape, 0.3))
    assert later["evaluation_status"] == "single_bulletin_research_sample"
    assert later["forecast_mae_fraction"] == pytest.approx(0.1)
    assert later["persistence_mae_fraction"] == pytest.approx(0.1)
    assert later["common_valid_cells"] == later["unique_observation_cells"]
    assert later["navigation_clearance"] is False


def test_non_nominal_cells_reduce_coverage_and_no_extrapolation():
    route, digest = load_research_transect()
    lat, lon, forecast, truth, uncertainty, flags = _grid()
    flags[:, :80] = 64
    result = sample_transect(forecast, truth, uncertainty, flags, lat, lon, route, digest,
                             bulletin_date=date(2026, 10, 4))
    assert 0 < result["nominal_cell_coverage_fraction"] < 1
    assert result["common_valid_cells"] < result["unique_observation_cells"]
    with pytest.raises(ValueError, match="outside"):
        sample_transect(forecast[:, 30:], truth[:, 30:], uncertainty[:, 30:], flags[:, 30:],
                         lat, lon[30:], route, digest, bulletin_date=date(2026, 10, 4))


def test_scored_route_cells_are_preserved_for_metric_recomputation():
    route, digest = load_research_transect()
    lat, lon, forecast, truth, uncertainty, flags = _grid()
    saved = {}
    result = sample_transect(forecast, truth, uncertainty, flags, lat, lon, route, digest,
                             bulletin_date=date(2026, 10, 4), sample_arrays=saved)
    assert len(saved["transect_forecast_sic_fraction"]) == result["common_valid_cells"]
    assert np.mean(saved["transect_forecast_sic_fraction"] - saved["transect_observed_sic_fraction"]) == pytest.approx(result["forecast_bias_fraction"])
    assert result["ice_edge_disagreement_fraction"] == 0


def test_route_config_cannot_claim_pre_freeze_bulletin(tmp_path):
    route, _ = load_research_transect()
    route["eligible_bulletins_from"] = "2026-10-03"
    path = tmp_path / "route.json"
    path.write_text(json.dumps(route), encoding="utf-8")
    with pytest.raises(ValueError, match="post-freeze"):
        load_research_transect(path)
