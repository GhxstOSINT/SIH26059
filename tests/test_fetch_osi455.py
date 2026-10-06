from datetime import date
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from ml import fetch_osi455
from ml.fetch_osi455 import (CATALOG_NS, DATASET_ID, cached_file_record, discover_files, requested_dates,
                             select_files, validate_file)


def catalog(entries):
    root = ET.Element(f"{CATALOG_NS}catalog")
    for name, path in entries:
        ET.SubElement(root, f"{CATALOG_NS}dataset", {"name": name, "urlPath": path})
    return ET.tostring(root)


def test_select_files_keeps_only_exact_daily_southern_merged_osi455_entries():
    date1 = "202009141200"
    date2 = "202009151200"
    sh1 = f"ice_drift_sh_ease2-750_cdr-v1p0_24h-{date1}.nc"
    sh2 = f"ice_drift_sh_ease2-750_cdr-v1p0_24h-{date2}.nc"
    nh = f"ice_drift_nh_ease2-750_cdr-v1p0_24h-{date2}.nc"
    old_version = f"ice_drift_sh_ease2-750_cdr-v0p9_24h-{date2}.nc"
    prefix = "osisaf/met.no/reprocessed/ice/drift_455m_files/merged/2020/09/"
    xml = catalog([(sh1, prefix + sh1), (sh2, prefix + sh2), (nh, prefix + nh),
                   (old_version, prefix + old_version), (sh2, prefix + "other-file.nc")])

    selected = select_files(xml, date(2020, 9, 15), date(2020, 9, 15))

    assert len(selected) == 1
    assert selected[0]["file"] == sh2
    assert selected[0]["date"] == "2020-09-15"
    assert selected[0]["url"].startswith("https://thredds.met.no/thredds/fileServer/")


def test_select_files_rejects_catalog_path_traversal():
    name = "ice_drift_sh_ease2-750_cdr-v1p0_24h-202009151200.nc"
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        select_files(catalog([(name, "../" + name)]), date(2020, 9, 15), date(2020, 9, 15))


def test_discovery_requires_a_complete_daily_catalog_coverage():
    day1 = "ice_drift_sh_ease2-750_cdr-v1p0_24h-202009141200.nc"
    path = "osisaf/met.no/reprocessed/ice/drift_455m_files/merged/2020/09/" + day1

    class FakeResponse:
        content = catalog([(day1, path)])

        def raise_for_status(self):
            return None

    class FakeSession:
        def get(self, url, timeout):
            assert "2020/09/catalog.xml" in url
            return FakeResponse()

    with pytest.raises(ValueError, match="no daily file.*2020-09-15"):
        discover_files(FakeSession(), date(2020, 9, 14), date(2020, 9, 15))


def test_requested_dates_are_inclusive_and_bounded():
    assert requested_dates(date(2020, 12, 31), date(2021, 1, 1), max_days=2) == [
        date(2020, 12, 31), date(2021, 1, 1)]
    with pytest.raises(ValueError, match="exceeds the bounded limit"):
        requested_dates(date(2020, 1, 1), date(2020, 2, 1), max_days=31)


def test_validate_osi455_netcdf_preserves_quality_flags_uncertainty_and_time(tmp_path):
    path = tmp_path / "osi455.nc.part"
    flags = np.array([[[30, 22], [1, 2]]], dtype=np.int8)
    dims = ("time", "yc", "xc")
    ds = xr.Dataset(
        {
            "Lambert_Azimuthal_Equal_Area": xr.DataArray(0, attrs={
                "grid_mapping_name": "lambert_azimuthal_equal_area",
                "latitude_of_projection_origin": -90.0,
                "proj4_string": "+proj=laea +lat_0=-90 +lon_0=0 +units=m",
            }),
            "time_bnds": (("time", "nv"), np.array([["2020-09-14T12", "2020-09-15T12"]], dtype="datetime64[h]")),
            "dX": (dims, np.array([[[1.0, 2.0], [np.nan, np.nan]]], dtype=np.float32), {"units": "km"}),
            "dY": (dims, np.array([[[3.0, 4.0], [np.nan, np.nan]]], dtype=np.float32), {"units": "km"}),
            "status_flag": (dims, flags, {"flag_meanings": "missing_input_data over_land interpolated nominal_quality"}),
            "uncert_dX_and_dY": (dims, np.array([[[2.0, 3.0], [np.nan, np.nan]]], dtype=np.float32), {"units": "km"}),
        },
        coords={"time": [np.datetime64("2020-09-15T12")], "xc": [0.0, 75.0], "yc": [75.0, 0.0]},
    )
    ds.to_netcdf(path)

    info = validate_file(path, "2020-09-15")

    assert info["time_start_utc"] == "2020-09-14T12:00:00Z"
    assert info["time_end_utc"] == "2020-09-15T12:00:00Z"
    assert info["grid_spacing_km"] == {"xc": 75.0, "yc": 75.0}
    assert info["status_flag_counts"] == {"1": 1, "2": 1, "22": 1, "30": 1}
    assert info["cells_with_success_flags_20_to_30"] == 2
    assert info["grid_cell_count"] == 4
    assert info["variables"]["dX"]["units"] == "km"
    assert info["uncertainty_km_median"] == 2.5


def test_dataset_identifier_is_specific_to_osi455():
    assert DATASET_ID == "EUMETSAT-OSI-455"


def test_incremental_fetch_cache_requires_exact_url_and_verified_hash(tmp_path):
    filename = "ice_drift_sh_ease2-750_cdr-v1p0_24h-202009151200.nc"
    payload = b"previously acquired immutable sample"
    (tmp_path / filename).write_bytes(payload)
    record = {"file": filename, "time": "2020-09-15", "url": "https://thredds.met.no/sample",
              "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    item = {"file": filename, "date": "2020-09-15", "url": record["url"]}

    assert cached_file_record(tmp_path, item, {filename: record}) == record
    assert cached_file_record(tmp_path, {**item, "url": "https://thredds.met.no/revised"},
                              {filename: record}) is None

    (tmp_path / filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="failed integrity verification"):
        cached_file_record(tmp_path, item, {filename: record})


def test_fetch_resumes_only_cached_files_with_a_matching_manifest_and_checksum(tmp_path, monkeypatch):
    days = ["2020-09-14", "2020-09-15"]
    records = [{"date": day, "file": f"ice_drift_sh_ease2-750_cdr-v1p0_24h-{day.replace('-', '')}1200.nc",
                "url": f"https://thredds.met.no/{day}.nc", "url_path": f"sample/{day}.nc"} for day in days]
    cached_payload = b"verified prior acquisition"
    cached_path = tmp_path / records[0]["file"]
    cached_path.write_bytes(cached_payload)
    cached_record = {"file": records[0]["file"], "time": days[0], "url": records[0]["url"],
                     "bytes": len(cached_payload), "sha256": hashlib.sha256(cached_payload).hexdigest(),
                     "validation": {"time": days[0]}}
    (tmp_path / "manifest.json").write_text(
        json.dumps({"dataset_id": DATASET_ID, "complete": False,
                    "requested_period": {"start": days[0], "end": days[1]},
                    "files": [cached_record]}), encoding="utf-8")
    monkeypatch.setattr(fetch_osi455, "discover_files", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(fetch_osi455, "validate_file", lambda _path, day=None: {"time": day})
    requested = []

    class FakeResponse:
        headers = {"Content-Length": "8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            yield b"new data"

    class FakeDownloadSession:
        def get(self, url, **_kwargs):
            requested.append(url)
            return FakeResponse()

    monkeypatch.setattr(fetch_osi455, "make_session", FakeDownloadSession)
    manifest = fetch_osi455.fetch(object(), tmp_path, date(2020, 9, 14), date(2020, 9, 15),
                                  max_days=2, workers=2)

    assert requested == [records[1]["url"]]
    assert manifest["complete"] is True
    assert len(manifest["files"]) == 2
    assert manifest["files"][0]["sha256"] == cached_record["sha256"]
    assert (tmp_path / "manifest.json").exists()


def test_atomic_manifest_write_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    replace = fetch_osi455.os.replace
    attempts = 0

    def transiently_locked(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporary file lock")
        return replace(source, destination)

    monkeypatch.setattr(fetch_osi455.os, "replace", transiently_locked)
    monkeypatch.setattr(fetch_osi455.time, "sleep", lambda _seconds: None)
    fetch_osi455.write_manifest_atomic(tmp_path, {"complete": False, "files": []})

    assert attempts == 2
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {
        "complete": False, "files": []}
