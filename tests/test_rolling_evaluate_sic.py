import datetime as dt

import numpy as np
import pytest
import xarray as xr

from ml import rolling_evaluate_sic
from ml.rolling_evaluate_sic import build_folds, evaluate


def test_build_folds_expands_training_and_keeps_validation_windows_disjoint():
    folds = build_folds(2000, 2011, train_years=8, validation_years=2, step_years=2)
    assert len(folds) == 2
    assert folds[0]["train_start"] == dt.date(2000, 1, 1)
    assert folds[0]["train_end"] == dt.date(2007, 12, 31)
    assert folds[0]["validation_start"] == dt.date(2008, 1, 1)
    assert folds[1]["train_end"] == dt.date(2009, 12, 31)
    assert folds[1]["validation_start"] == dt.date(2010, 1, 1)


def test_build_folds_rejects_archive_shorter_than_window():
    with pytest.raises(ValueError, match="too short"):
        build_folds(2000, 2008, train_years=8, validation_years=2, step_years=2)


def test_expanding_evaluation_uses_later_calendar_blocks(tmp_path):
    data_dir = tmp_path / "archive"
    data_dir.mkdir()
    for year in range(2000, 2012):
        dates = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01"))
        phase = np.arange(len(dates), dtype=np.float32) / 365
        values = np.broadcast_to((0.4 + 0.2 * np.sin(2 * np.pi * phase))[:, None, None],
                                 (len(dates), 2, 2)).copy()
        xr.Dataset({"IceConc": (("time", "y", "x"), values)}, coords={"time": dates}).to_netcdf(
            data_dir / f"sample_{year}.nc")
    report = evaluate(data_dir, train_years=8, validation_years=2, step_years=2)
    assert report["fold_count"] == 2
    assert report["folds"][0]["validation_period"] == {"start": "2008-01-01", "end": "2009-12-31"}
    assert report["folds"][1]["validation_period"] == {"start": "2010-01-01", "end": "2011-12-31"}
    assert report["overall"]["grid_cell_samples"] > 5_000
    assert report["overall"]["rmse"] >= 0
    assert report["monthly_climatology"]["grid_cell_samples"] > 0
    assert report["monthly_climatology"]["rmse"] >= 0
    assert all(fold["monthly_climatology"]["grid_cell_samples"] > 0 for fold in report["folds"])
    assert report["by_target_month_pooled"]
    assert "route-scale" in report["warning"]


def test_area_weighted_metrics_use_ground_area_and_include_season_and_latitude(tmp_path, monkeypatch):
    data_dir = tmp_path / "g02202"
    data_dir.mkdir()
    for year in range(2000, 2012):
        dates = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01"))
        phase = np.arange(len(dates), dtype=np.float32) / 365
        base = 0.4 + 0.2 * np.sin(2 * np.pi * phase)
        values = np.broadcast_to(base[:, None, None], (len(dates), 2, 2)).copy()
        xr.Dataset({"IceConc": (("time", "y", "x"), values)}, coords={"time": dates}).to_netcdf(
            data_dir / f"sample_{year}.nc")
    (data_dir / "manifest.json").write_text('{"dataset_id":"G02202"}', encoding="utf-8")
    monkeypatch.setattr(rolling_evaluate_sic, "verify_g02202_manifest", lambda *_args: None)
    monkeypatch.setattr(rolling_evaluate_sic, "g02202_cell_geometry", lambda *_args: (
        np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[-85.0, -75.0], [-65.0, -55.0]])))

    report = evaluate(data_dir, train_years=8, validation_years=2, step_years=2)

    area = report["area_weighted"]
    assert area["overall"]["evaluated_area_km2_samples"] > 0
    assert area["monthly_climatology"]["evaluated_area_km2_samples"] > 0
    assert area["monthly_climatology"]["rmse"] >= 0
    assert area["by_target_month"]["01"]["rmse"] is not None
    assert set(area["by_latitude_band"]) == {"south_of_80S", "80S_to_70S", "70S_to_60S", "north_of_60S"}
    assert all("area_weighted" in fold for fold in report["folds"])


def test_area_weighted_summary_respects_cell_ground_area():
    metrics = rolling_evaluate_sic._new_area_metrics()
    rolling_evaluate_sic._accumulate_area(metrics, np.array([0.0, 1.0]), np.array([0.0, 0.0]),
                                          np.array([1.0, 1.0]), np.array([9.0, 1.0]))
    summary = rolling_evaluate_sic._summarize_area(metrics)
    assert summary["rmse"] == pytest.approx(np.sqrt(0.1))
    assert summary["persistence_rmse"] == pytest.approx(1.0)
    assert summary["evaluated_area_km2_samples"] == pytest.approx(10.0)
