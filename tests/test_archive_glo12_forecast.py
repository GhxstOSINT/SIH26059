from datetime import datetime, timezone
from types import SimpleNamespace
import json

import numpy as np
import pytest
import xarray as xr

from ml.archive_glo12_forecast import inventory_forecast
from ml.inventory_copernicus_sic import inventory as inventory_observation
from ml.fetch_glo12_forecast import expected_names, fetch_run
from ml.verify_glo12_forecast import match_run, write_match
from ml.assess_glo12_verification import assess_reports
from service.glo12_status import forecast_archive_status


NAME = "glo12_rg_1d-m_20261003-20261003_2D_fcst_R20261002.nc"


def _file(tmp_path, *, filename=NAME, valid_time="2026-10-03T12:00:00", value=0.25):
    path = tmp_path / filename
    ds = xr.Dataset(
        {"siconc": (("time", "latitude", "longitude"),
                     np.full((1, 2, 2), value, dtype=np.float32), {"units": "1"})},
        coords={"time": np.array([valid_time], dtype="datetime64[s]"),
                "latitude": [-67.0, -66.0], "longitude": [-61.0, -60.0]},
    )
    ds.to_netcdf(path)
    return path


def test_original_forecast_inventory_binds_run_valid_time_and_hash(tmp_path):
    path = _file(tmp_path)
    report = inventory_forecast(
        path, captured_at=datetime(2026, 10, 2, 10, 30, tzinfo=timezone.utc),
        provider_last_modified_at=datetime(2026, 10, 2, 2, 30, tzinfo=timezone.utc),
    )
    assert report["bulletin_date"] == "2026-10-02"
    assert report["valid_date"] == "2026-10-03"
    assert report["lead_days"] == 1
    assert len(report["source_sha256"]) == 64
    assert report["grid"]["valid_southern_sic_cells"] == 4
    assert report["availability_before_capture_verified"] is False
    assert report["uncertainty_calibrated"] is False
    assert report["navigation_clearance"] is False


@pytest.mark.parametrize("filename", [
    "glo12_rg_1d-m_20261003-20261003_2D_hcst_R20261002.nc",
    "glo12_rg_1d-m_20261003-20261003_2D_fcst_R20261004.nc",
    "glo12_rg_1d-m_20261013-20261013_2D_fcst_R20261002.nc",
    "subset_20261003.nc",
])
def test_refuses_analysis_invalid_lead_and_subset_filename(tmp_path, filename):
    with pytest.raises(ValueError):
        inventory_forecast(_file(tmp_path, filename=filename))


def test_refuses_mismatched_time_and_out_of_range_sic(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        inventory_forecast(_file(tmp_path, valid_time="2026-10-04T12:00:00"))
    with pytest.raises(ValueError, match="outside"):
        inventory_forecast(_file(tmp_path, value=1.2))


def test_refuses_future_provider_modification_time(tmp_path):
    with pytest.raises(ValueError, match="conflicts"):
        inventory_forecast(
            _file(tmp_path), captured_at=datetime(2026, 10, 2, 10, tzinfo=timezone.utc),
            provider_last_modified_at=datetime(2026, 10, 2, 11, tzinfo=timezone.utc),
        )


def test_capture_after_valid_date_is_not_prospective(tmp_path):
    report = inventory_forecast(
        _file(tmp_path), captured_at=datetime(2026, 10, 4, tzinfo=timezone.utc))
    assert report["eligible_for_prospective_verification"] is False


def test_expected_forecast_names_are_exact_and_bounded():
    names = expected_names(datetime(2026, 10, 2).date(), (1, 3, 5, 7))
    assert len(names) == 4
    assert NAME in names
    assert "glo12_rg_1d-m_20261009-20261009_2D_fcst_R20261002.nc" in names
    with pytest.raises(ValueError):
        expected_names(datetime(2026, 10, 2).date(), (1, 1))
    with pytest.raises(ValueError):
        expected_names(datetime(2026, 10, 2).date(), (11,))


def test_fetch_keeps_original_manifest_and_refuses_tampering(tmp_path, monkeypatch):
    import copernicusmarine

    path = _file(tmp_path)
    calls = []

    def fake_get(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(files=[SimpleNamespace(
            filename=NAME, file_path=path,
            last_modified_datetime="2026-10-02T02:30:00+00:00")])

    monkeypatch.setattr(copernicusmarine, "get", fake_get)
    first = fetch_run(datetime(2026, 10, 2).date(), tmp_path, (1,))
    second = fetch_run(datetime(2026, 10, 2).date(), tmp_path, (1,))
    assert first == second
    assert len(calls) == 1
    assert calls[0]["no_directories"] is True
    assert calls[0]["skip_existing"] is True
    assert calls[0]["dataset_version"] == "202406"
    manifest = tmp_path / f"{path.stem}.manifest.json"
    content = manifest.read_text(encoding="utf-8").replace(
        '"uncertainty_calibrated": false', '"uncertainty_calibrated": true')
    manifest.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        fetch_run(datetime(2026, 10, 2).date(), tmp_path, (1,))


def test_verification_matches_only_nominal_observed_pixels(tmp_path):
    forecast = _file(tmp_path)
    manifest = tmp_path / f"{forecast.stem}.manifest.json"
    report = inventory_forecast(
        forecast, captured_at=datetime(2026, 10, 2, 10, tzinfo=timezone.utc))
    manifest.write_text(json.dumps(report), encoding="utf-8")
    observation = tmp_path / "observation.nc"
    xr.Dataset(
        {"ice_conc": (("time", "latitude", "longitude"),
                       np.array([[[25, 30], [40, 50]]], dtype=np.float32), {"units": "%"}),
         "total_uncertainty": (("time", "latitude", "longitude"),
                               np.full((1, 2, 2), 10, dtype=np.float32)),
         "status_flag": (("time", "latitude", "longitude"),
                         np.array([[[0, 0], [64, 0]]], dtype=np.int16))},
        coords={"time": np.array(["2026-10-03"], dtype="datetime64[D]"),
                "latitude": [-67.0, -66.0], "longitude": [-61.0, -60.0]},
    ).to_netcdf(observation)
    score, samples = match_run(manifest, observation)
    assert score["lead_days"] == 1
    assert score["matched_cells"] == 3
    assert score["matched_fraction"] == 0.75
    assert score["uncertainty_calibrated"] is False
    assert len(samples["forecast_sic_fraction"]) == 3
    assert score["mae_fraction"] == pytest.approx(0.10, abs=1e-6)


def test_persistence_input_must_have_been_captured_before_issue(tmp_path):
    forecast = _file(tmp_path)
    manifest = tmp_path / f"{forecast.stem}.manifest.json"
    manifest.write_text(json.dumps(inventory_forecast(
        forecast, captured_at=datetime(2026, 10, 2, 10, tzinfo=timezone.utc))), encoding="utf-8")
    def observed(path, day, value):
        xr.Dataset(
            {"ice_conc": (("time", "latitude", "longitude"),
                           np.full((1, 2, 2), value, dtype=np.float32), {"units": "%"}),
             "total_uncertainty": (("time", "latitude", "longitude"),
                                   np.full((1, 2, 2), 5, dtype=np.float32)),
             "status_flag": (("time", "latitude", "longitude"),
                             np.zeros((1, 2, 2), dtype=np.int16))},
            coords={"time": np.array([day], dtype="datetime64[D]"),
                    "latitude": [-67.0, -66.0], "longitude": [-61.0, -60.0]},
        ).to_netcdf(path)
    target = tmp_path / "target.nc"
    observed(target, "2026-10-03", 30)
    baseline = tmp_path / "baseline.nc"
    observed(baseline, "2026-10-01", 20)
    baseline_report = inventory_observation(baseline, datetime(2026, 10, 1).date(),
                                            datetime(2026, 10, 1).date())
    baseline_manifest = tmp_path / "baseline.manifest.json"
    baseline_report["captured_at_utc"] = "2026-10-02T01:00:00+00:00"
    baseline_manifest.write_text(json.dumps(baseline_report), encoding="utf-8")
    with pytest.raises(ValueError, match="after the forecast bulletin"):
        match_run(manifest, target, baseline_manifest)
    baseline_report["captured_at_utc"] = "2026-10-01T12:00:00+00:00"
    baseline_manifest.write_text(json.dumps(baseline_report), encoding="utf-8")
    score, samples = match_run(manifest, target, baseline_manifest)
    assert score["as_issued_persistence_comparison"]["persistence_mae_fraction"] == pytest.approx(0.1)
    assert len(samples["persistence_sic_fraction"]) == score["matched_cells"]


def test_forecast_status_exposes_only_integrity_verified_runs(tmp_path):
    forecast = _file(tmp_path)
    manifest = tmp_path / f"{forecast.stem}.manifest.json"
    record = inventory_forecast(
        forecast, captured_at=datetime(2026, 10, 2, 10, tzinfo=timezone.utc))
    manifest.write_text(json.dumps(record), encoding="utf-8")
    status = forecast_archive_status(tmp_path)
    assert status["archive_state"] == "verified_original_files"
    assert status["verified_leads"] == [1]
    assert status["operator_decision_ready"] is False
    _file(tmp_path, value=0.5)
    changed = forecast_archive_status(tmp_path)
    assert changed["archive_state"] == "unavailable"
    assert changed["rejected_files"] == 1


def test_calibration_gate_counts_bulletins_not_pixels(tmp_path):
    reports = []
    for lead in (1, 3, 5, 7):
        report = tmp_path / f"lead-{lead}.json"
        record = {
            "scope": "single_run_research_verification_sample",
            "bulletin_date": "2026-10-02", "valid_date": f"2026-10-{2 + lead:02d}",
            "lead_days": lead, "matched_cells": 200000,
            "forecast_source_sha256": "a" * 64,
            "observation_source_sha256": "b" * 64,
            "uncertainty_calibrated": False, "navigation_clearance": False,
        }
        sample = {"forecast_sic_fraction": np.full(2, 0.2, dtype=np.float32),
                  "observed_sic_fraction": np.full(2, 0.3, dtype=np.float32),
                  "observation_uncertainty_fraction": np.full(2, 0.1, dtype=np.float32)}
        record["matched_cells"] = 2
        write_match(record, sample, report)
        reports.append(report)
    gate = assess_reports(reports)
    assert gate["research_calibration_sample_gate_passed"] is False
    assert all(item["independent_bulletin_dates"] == 1 for item in gate["leads"])
    assert gate["uncertainty_calibrated"] is False
    with reports[0].with_suffix(".npz").open("ab") as target:
        target.write(b"altered")
    rejected = assess_reports(reports)
    assert rejected["rejected_reports"] == 1
