import datetime as dt
import hashlib
import json

import numpy as np
import pytest
import xarray as xr

from ml.refresh_viirs import active_status, refresh, validate_policy


NOW = dt.datetime(2026, 10, 1, 12, tzinfo=dt.timezone.utc)
SOURCE = "noaacwVIIRSn21iceconcSP06Daily"


def policy():
    return {"schema_version": 1, "scope": "research_only", "dataset_priority": [SOURCE],
            "max_source_age_hours": 72, "max_local_age_hours": 72, "min_valid_fraction": .25,
            "window_days": 3, "center_lon_lat": [-60.0, -66.0], "half_width_km": 100.0}


def fake_download(target, dataset_id, start, end, config):
    name = "scene.nc"
    path = target / name
    ds = xr.Dataset({"IceConc": (("time", "rows", "cols"), np.full((1, 2, 2), .3, dtype=np.float32))},
                    coords={"time": [np.datetime64("2026-10-01T06:00:00")],
                            "rows": [1000.0, 0.0], "cols": [0.0, 1000.0]})
    ds.to_netcdf(path)
    manifest = {"complete": True, "dataset_id": dataset_id,
                "selection": {"center_lon_lat": config["center_lon_lat"],
                              "half_width_km": config["half_width_km"], "crs": "EPSG:3976"},
                "files": [{"file": name, "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_refresh_promotes_only_quality_checked_staging_and_expires(tmp_path):
    config = validate_policy(policy())
    report = refresh(tmp_path, config, now=NOW,
                     fetcher=lambda _: "2026-10-01T06:00:00Z", downloader=fake_download)
    assert report["status"] == "passed"
    assert active_status(tmp_path, config, now=NOW)["available_for_research"] is True
    assert active_status(tmp_path, config, now=NOW + dt.timedelta(days=5))["available_for_research"] is False


def test_failed_refresh_retains_previous_pointer_and_detects_tampering(tmp_path):
    config = policy()
    refresh(tmp_path, config, now=NOW,
            fetcher=lambda _: "2026-10-01T06:00:00Z", downloader=fake_download)
    pointer = (tmp_path / "active.json").read_bytes()

    def broken(*_args):
        raise IOError("source download failed")

    with pytest.raises(IOError, match="download failed"):
        refresh(tmp_path, config, now=NOW, fetcher=lambda _: "2026-10-01T06:00:00Z", downloader=broken)
    assert (tmp_path / "active.json").read_bytes() == pointer
    active = json.loads(pointer)
    path = tmp_path / "runs" / active["run_id"] / "scene.nc"
    with path.open("ab") as stream:
        stream.write(b"altered")
    assert active_status(tmp_path, config, now=NOW)["available_for_research"] is False


def test_policy_rejects_unknown_source_and_operational_scope():
    config = policy()
    config["dataset_priority"] = ["attacker"]
    with pytest.raises(ValueError, match="allowlisted"):
        validate_policy(config)
    config = policy()
    config["scope"] = "operational"
    with pytest.raises(ValueError, match="research_only"):
        validate_policy(config)


def test_existing_lock_refuses_concurrent_refresh(tmp_path):
    (tmp_path / ".refresh.lock").write_text("other writer", encoding="utf-8")
    with pytest.raises(FileExistsError):
        refresh(tmp_path, policy(), now=NOW,
                fetcher=lambda _: "2026-10-01T06:00:00Z", downloader=fake_download)
    assert not (tmp_path / "active.json").exists()
