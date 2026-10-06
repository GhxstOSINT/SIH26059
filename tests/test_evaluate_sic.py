import json

import numpy as np
import xarray as xr

from ml.evaluate_sic import (add_ice_edge_metrics, blank_ice_edge_totals,
                             coarsen_grid, g02202_cell_geometry, main,
                             summarize_ice_edge)


def test_coarsen_grid_block_averages_finite_values():
    values = np.array([[1, 3, 5], [5, np.nan, 7], [9, 11, 13]], dtype=np.float32)
    result = coarsen_grid(values, 2)
    assert result.shape == (1, 1)
    assert result[0, 0] == 3.0


def test_g02202_cell_geometry_uses_source_projection(tmp_path):
    from pyproj import CRS

    crs = CRS.from_epsg(3412)
    dataset = xr.Dataset(
        {"crs": xr.DataArray(0, attrs={"crs_wkt": crs.to_wkt()})},
        coords={"x": [-12500.0, 12500.0], "y": [12500.0, -12500.0]},
    )
    dataset.to_netcdf(tmp_path / "grid.nc")
    area, latitude = g02202_cell_geometry(tmp_path, (2, 2))
    assert area.shape == latitude.shape == (2, 2)
    assert np.all(np.isfinite(area)) and np.all(area > 0)
    assert np.all(latitude < -80)


def test_ice_edge_metrics_measure_disagreement_and_sie_error_on_shared_valid_area():
    prediction = np.array([[0.2, 0.0, 0.9]])
    persistence = np.array([[0.0, 0.2, 0.1]])
    target = np.array([[0.2, 0.2, 0.0]])
    valid = np.array([[True, True, False]])
    cell_area = np.array([[2.0, 3.0, 100.0]])
    totals = blank_ice_edge_totals()

    add_ice_edge_metrics(totals, prediction, persistence, target, valid, cell_area)
    result = summarize_ice_edge(totals)

    assert result["valid_daily_fields"] == 1
    assert result["threshold_sic_fraction"] == 0.15
    assert result["mean_daily_iiee_million_km2"] == 0.000003
    assert result["persistence_mean_daily_iiee_million_km2"] == 0.000002
    assert result["mean_daily_absolute_sie_error_million_km2"] == 0.000003
    assert result["persistence_mean_daily_absolute_sie_error_million_km2"] == 0.000002


def test_independent_evaluation_writes_overall_and_month_metrics(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "netcdf"
    data_dir.mkdir()
    dates = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-05-01"))
    values = np.broadcast_to(np.linspace(0.1, 0.8, len(dates), dtype=np.float32)[:, None, None],
                             (len(dates), 2, 3)).copy()
    xr.Dataset({"IceConc": (("time", "y", "x"), values)}, coords={"time": dates}).to_netcdf(data_dir / "sample.nc")
    (data_dir / "manifest.json").write_text(json.dumps({"dataset_id": "TEST", "files": []}), encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"model_id": "test-model", "coefficients": [0, 1, 0, 0, 0]}), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr("sys.argv", ["evaluate_sic", str(data_dir), "--model", str(model), "--output", str(output)])

    assert main() == 0
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["period"] == {"first_target_date": "2010-01-03", "last_target_date": "2010-04-30"}
    assert report["overall"]["grid_cell_samples"] == (len(dates) - 2) * 6
    assert report["source_manifest"]["dataset_id"] == "TEST"
    assert len(report["model_fingerprint"]) == 64
    assert set(report["by_target_month"]) == {"01", "02", "03", "04"}
    assert set(report["by_target_year"]) == {"2010"}
    assert report["metric_weighting"].startswith("Cell-pooled metrics")
