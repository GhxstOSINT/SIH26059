from datetime import date
import json
import numpy as np
import xarray as xr
import pytest

from ml.fetch_nsidc0116 import (ARCHIVE_ROOT, DATASET_ID, archive_files,
                               existing_complete_manifest, select_overlapping_files,
                               sha256_file, validate_motion_file)


class FakeResponse:
    def __init__(self, url, text):
        self.url = url
        self.text = text

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, pages, redirects=None):
        self.pages = pages
        self.redirects = redirects or {}
        self.visited = []

    def get(self, url, **kwargs):
        self.visited.append(url)
        return FakeResponse(self.redirects.get(url, url), self.pages[url])


def test_archive_inventory_is_bounded_to_overlapping_years_and_safe_host():
    year2020 = ARCHIVE_ROOT + "2020/"
    year2021 = ARCHIVE_ROOT + "2021/"
    pages = {
        ARCHIVE_ROOT: '<a href="2020/">2020/</a><a href="2021/">2021/</a>'
                      '<a href="https://example.org/2020/">external</a>',
        year2020: '<a href="icemotion_daily_sh_25km_20200101_20200131_v4.1.nc">file</a>'
                  '<a href="https://evil.example/icemotion_daily_sh_25km_20200101_20200131_v4.1.nc">bad</a>',
    }
    session = FakeSession(pages)
    files = archive_files(session, date(2020, 1, 12), date(2020, 1, 20))
    assert files == [("icemotion_daily_sh_25km_20200101_20200131_v4.1.nc",
                      year2020 + "icemotion_daily_sh_25km_20200101_20200131_v4.1.nc")]
    assert year2021 not in session.visited
    assert select_overlapping_files(files, date(2020, 1, 12), date(2020, 1, 20)) == files


def test_archive_inventory_rejects_out_of_root_links():
    session = FakeSession({
        ARCHIVE_ROOT: '<a href="../../other/">escape</a><a href="2020/">2020/</a>',
        ARCHIVE_ROOT + "2020/": '<a href="../../../../icemotion_daily_sh_25km_20200101_20200131_v4.1.nc">escape</a>',
    })
    assert archive_files(session, date(2020, 1, 1), date(2020, 1, 31)) == []


def test_archive_inventory_rejects_redirect_outside_archive():
    session = FakeSession(
        {ARCHIVE_ROOT: ""},
        redirects={ARCHIVE_ROOT: "https://urs.earthdata.nasa.gov/login"},
    )
    with pytest.raises(ValueError, match="redirected outside"):
        archive_files(session, date(2020, 1, 1), date(2020, 1, 31))


def test_motion_file_validation_requires_two_components_and_valid_time(tmp_path):
    path = tmp_path / "motion.nc"
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    dataset = xr.Dataset(
        {"u": (("time", "y", "x"), np.ones((2, 2, 3))),
         "v": (("time", "y", "x"), np.ones((2, 2, 3))),
         "icemotion_error_estimate": (("time", "y", "x"), np.ones((2, 2, 3))),
         "crs": xr.DataArray(0, attrs={"grid_mapping_name": "lambert_azimuthal_equal_area", "spatial_ref": "EPSG:3409"})},
        coords={"time": time, "x": [0.0, 25000.0, 50000.0], "y": [50000.0, 25000.0]},
    )
    dataset.to_netcdf(path)
    checked = validate_motion_file(path)
    assert checked["time_start"] == "2020-01-01"
    assert checked["time_end"] == "2020-01-02"
    assert checked["time_count"] == 2
    assert checked["crs_attributes"]["spatial_ref"] == "EPSG:3409"
    assert checked["grid_axes"]["x"]["median_spacing"] == 25000.0

    dataset.drop_vars("v").to_netcdf(path, mode="w")
    try:
        validate_motion_file(path)
    except ValueError as exc:
        assert "u/v" in str(exc)
    else:
        raise AssertionError("A file without both vector components must be rejected")


def test_existing_download_is_reused_only_when_manifest_and_checksums_match(tmp_path):
    name = "icemotion_daily_sh_25km_20200101_20200131_v4.1.nc"
    path = tmp_path / name
    time = np.array(["2020-01-01"], dtype="datetime64[ns]")
    xr.Dataset(
        {"u": (("time", "y", "x"), np.ones((1, 2, 2))),
         "v": (("time", "y", "x"), np.ones((1, 2, 2))),
         "crs": xr.DataArray(0, attrs={"grid_mapping_name": "lambert_azimuthal_equal_area"})},
        coords={"time": time, "x": [0.0, 25000.0], "y": [25000.0, 0.0]},
    ).to_netcdf(path)
    url = f"https://daacdata.apps.nsidc.org/pub/{name}"
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset_id": DATASET_ID, "complete": True,
        "requested_period": {"start": "2020-01-01", "end": "2020-01-31"},
        "files": [{"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}],
    }), encoding="utf-8")
    selected = [(name, url)]
    assert existing_complete_manifest(tmp_path, date(2020, 1, 1), date(2020, 1, 31), selected)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity verification"):
        existing_complete_manifest(tmp_path, date(2020, 1, 1), date(2020, 1, 31), selected)
