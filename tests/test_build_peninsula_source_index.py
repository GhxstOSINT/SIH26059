from pathlib import Path

import pytest

from ml import build_peninsula_source_index as module


def _source(platform: str, grid_hash: str = "same") -> dict:
    return {
        "dataset_id": next(key for key, value in module.ALLOWED_IDS.items() if value == platform),
        "platform": platform,
        "manifest_sha256": "manifest",
        "archive_accessed_at_utc": "2026-10-02T00:00:00Z",
        "selection": {
            "start_date": "2024-06-01",
            "end_date": "2024-06-02",
            "center_lon_lat": [-60, -66],
            "half_width_km": 100,
            "projected_bounds_m": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
        },
        "grid_shape": [2, 2],
        "grid_coordinate_sha256": grid_hash,
        "scenes": {
            "2024-06-01": {
                "observed_at_utc": "2024-06-01T12:00:00Z",
                "valid_grid_fraction": 0.75,
                "source_file": "sample.nc",
                "source_sha256": "file",
                "published_at_utc": None,
                "asof_verified": False,
            }
        },
    }


def test_index_preserves_missing_days_and_unknown_publication(monkeypatch):
    sources = {"a": _source("S-NPP"), "b": _source("NOAA-21")}
    monkeypatch.setattr(module, "read_source", lambda path: sources[path.name])
    report = module.build_index([Path("a"), Path("b")])
    assert report["summary"]["scheduled_days"] == 2
    assert report["summary"]["days_observed_by_all_platforms"] == 1
    assert report["summary"]["forecast_origins_with_verified_availability"] == 0
    assert report["days"][1]["platforms"]["S-NPP"]["status"] == "missing_observation"
    assert report["days"][0]["platforms"]["NOAA-21"]["published_at_utc"] is None


def test_index_rejects_different_grid_coordinates(monkeypatch):
    sources = {"a": _source("S-NPP"), "b": _source("NOAA-21", "different")}
    monkeypatch.setattr(module, "read_source", lambda path: sources[path.name])
    with pytest.raises(ValueError, match="same period and grid"):
        module.build_index([Path("a"), Path("b")])


def test_index_rejects_future_observation(monkeypatch):
    source = _source("S-NPP")
    source["scenes"]["2024-06-01"]["observed_at_utc"] = "2024-06-02T00:00:00Z"
    monkeypatch.setattr(module, "read_source", lambda _: source)
    with pytest.raises(ValueError, match="not earlier"):
        module.build_index([Path("a")])
