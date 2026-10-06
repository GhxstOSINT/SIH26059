from datetime import date
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from ml.fetch_osi_sic_observation import fetch_observation, load_pilot_region
from ml.run_daily_forecast_cycle import run_cycle


def _observation(path):
    xr.Dataset(
        {"ice_conc": (("time", "latitude", "longitude"),
                       np.full((1, 2, 2), 20, dtype=np.float32), {"units": "%"}),
         "total_uncertainty": (("time", "latitude", "longitude"),
                               np.full((1, 2, 2), 5, dtype=np.float32)),
         "status_flag": (("time", "latitude", "longitude"),
                         np.zeros((1, 2, 2), dtype=np.int16))},
        coords={"time": np.array(["2026-10-01"], dtype="datetime64[D]"),
                "latitude": [-67.0, -66.0], "longitude": [-61.0, -60.0]},
    ).to_netcdf(path)


def test_observation_capture_is_idempotent_and_rejects_tampering(tmp_path, monkeypatch):
    import copernicusmarine

    calls = []

    def fake_subset(**kwargs):
        calls.append(kwargs)
        source = tmp_path / kwargs["output_filename"]
        _observation(source)
        return SimpleNamespace(file_path=source)

    monkeypatch.setattr(copernicusmarine, "subset", fake_subset)
    first = fetch_observation(date(2026, 10, 1), tmp_path)
    second = fetch_observation(date(2026, 10, 1), tmp_path)
    assert first == second
    assert len(calls) == 1
    assert calls[0]["variables"] == ["ice_conc", "status_flag", "total_uncertainty"]
    assert first["navigation_clearance"] is False
    source = tmp_path / first["source_file"]
    _observation(source)
    with source.open("ab") as target:
        target.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fetch_observation(date(2026, 10, 1), tmp_path)


def test_daily_cycle_reports_future_valid_dates_as_not_yet_due(tmp_path):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    (forecast_dir / "run.manifest.json").write_text(json.dumps({
        "valid_date": "2026-10-03", "bulletin_date": "2026-10-02",
        "lead_days": 1, "eligible_for_prospective_verification": True,
    }), encoding="utf-8")
    status = run_cycle(date(2026, 10, 2), forecast_dir, tmp_path / "obs",
                       tmp_path / "verification", fetch_forecast=False)
    assert status["completed_verifications"] == []
    assert status["pending_observations"] == []
    assert status["observation_region_id"] == "peninsula-weddell"
    assert status["observation_bbox_west_east_south_north"] == [-73.0, -20.0, -78.0, -58.0]
    assert status["calibration"]["uncertainty_calibrated"] is False


def test_frozen_pilot_envelope_contains_reference_sites():
    region_id, (west, east, south, north), digest = load_pilot_region()
    assert region_id == "peninsula-weddell"
    assert west < -68.1248 < east and south < -67.568889 < north
    assert west < -25.50852 < east and south < -75.56821 < north
    assert len(digest) == 64
