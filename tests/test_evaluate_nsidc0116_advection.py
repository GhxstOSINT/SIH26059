from datetime import date
import hashlib
import json
import sys

import numpy as np
import pytest
import xarray as xr

from ml.align_nsidc0116_archive import align_archive
from ml.evaluate_nsidc0116_advection import (backtrace_sic, motion_archive_entry,
                                               read_sic, score)


def test_backtrace_uses_motion_direction_and_respects_support():
    pytest.importorskip("scipy")
    base = np.tile(np.arange(5, dtype=np.float32), (3, 1))
    dx = np.full_like(base, 1000.0)
    dy = np.zeros_like(base)
    support = np.ones_like(base)
    advected, valid = backtrace_sic(base, dx, dy, support, 1000.0, 1000.0)

    assert not valid[:, 0].any()  # Eastward ice movement samples the western source; left edge exits the grid.
    np.testing.assert_allclose(advected[:, 1:], np.tile(np.arange(4, dtype=np.float32), (3, 1)))

    support[:, 2] = 0.5
    _, valid_with_gap = backtrace_sic(base, dx, dy, support, 1000.0, 1000.0)
    assert not valid_with_gap[:, 2].any()


def test_manifest_date_lookup_requires_a_unique_covering_file():
    entries = {
        "motion-a.nc": {"file": "motion-a.nc", "time_start": "2020-01-01", "time_end": "2020-01-31"},
        "motion-b.nc": {"file": "motion-b.nc", "time_start": "2020-02-01", "time_end": "2020-02-29"},
    }
    assert motion_archive_entry(entries, date(2020, 1, 15))["file"] == "motion-a.nc"
    with pytest.raises(ValueError, match="found 0"):
        motion_archive_entry(entries, date(2020, 3, 1))
    entries["duplicate.nc"] = {"file": "duplicate.nc", "time_start": "2020-01-15", "time_end": "2020-01-15"}
    with pytest.raises(ValueError, match="found 2"):
        motion_archive_entry(entries, date(2020, 1, 15))


def test_sic_mask_excludes_interpolated_and_invalid_quality_cells():
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset(
        {
            "cdr_seaice_conc": (("time", "y", "x"), np.array([[[0.2, 0.3, 0.4]]], dtype=np.float32)),
            "cdr_seaice_conc_qa_flag": (("time", "y", "x"), np.array([[[0, 32, 8]]], dtype=np.uint8)),
            "cdr_seaice_conc_interp_temporal_flag": (("time", "y", "x"), np.array([[[0, 0, 0]]], dtype=np.uint8)),
        }, coords={"time": np.array(["2020-01-01"], dtype="datetime64[D]"), "y": [0], "x": [0, 1, 2]},
    )
    values, valid = read_sic(ds, date(2020, 1, 1))
    np.testing.assert_array_equal(valid, [[True, False, False]])
    assert np.isnan(values[0, 1:]).all()
    with pytest.raises(ValueError, match="exactly one field"):
        read_sic(ds, date(2020, 1, 2))


def test_score_uses_only_common_valid_cells_and_reports_persistence_baseline():
    base = np.array([[0.2, 0.4]], dtype=np.float32)
    truth = np.array([[0.4, 0.4]], dtype=np.float32)
    advected = np.array([[0.4, 0.0]], dtype=np.float32)
    valid = np.array([[True, False]])
    report = score([(date(2020, 1, 1), base, truth, advected, valid)], 1.0)
    assert report["n_cell_days"] == 1
    assert report["persistence_mae"] == pytest.approx(0.2)
    assert report["blend_mae"] == pytest.approx(0.0)
    assert report["mae_reduction_percent"] == pytest.approx(100.0)


def test_coefficient_report_rejects_overlap_before_loading_archives(tmp_path, monkeypatch):
    from ml.evaluate_nsidc0116_advection import main
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({
        "experiment": "one-day lagged NSIDC-0116 motion-advection SIC hindcast",
        "date_range": {"end_issue": "2024-01-10"}, "selected_lambda": 0.7,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate_nsidc0116_advection.py",
                                   "--motion-dir", str(tmp_path / "missing-motion"),
                                   "--aligned-dir", str(tmp_path / "missing-aligned"),
                                   "--sic-file", str(tmp_path / "missing-sic.nc"),
                                   "--start", "2024-01-02", "--end", "2024-01-10",
                                   "--lambda-report", str(prior),
                                   "--output", str(tmp_path / "out.json")])
    with pytest.raises(ValueError, match="must end before"):
        main()


def test_aligned_archive_hindcast_is_manifest_bound_and_chronological(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    pyproj = pytest.importorskip("pyproj")
    from pyproj import CRS, Transformer
    from ml.evaluate_nsidc0116_advection import main

    source_crs = CRS.from_epsg(3409)
    target_crs = CRS.from_epsg(3412)
    source_x = np.arange(-37500.0, 50000.0, 25000.0)
    source_y = np.arange(-962500.0, -1050000.0, -25000.0)
    target_center_x, target_center_y = Transformer.from_crs(
        source_crs, target_crs, always_xy=True).transform(0.0, -1000000.0)
    target_x = target_center_x + np.arange(-37500.0, 50000.0, 25000.0)
    target_y = target_center_y + np.arange(-37500.0, 50000.0, 25000.0)[::-1]

    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    motion_path = motion_dir / "motion-20191231-20200101.nc"
    times = np.array(["2019-12-31", "2020-01-01"], dtype="datetime64[ns]")
    motion = xr.Dataset(
        {
            "u": (("time", "y", "x"), np.zeros((2, 4, 4))),
            "v": (("time", "y", "x"), np.zeros((2, 4, 4))),
            "icemotion_error_estimate": (("time", "y", "x"), np.full((2, 4, 4), 10.0)),
            "crs": xr.DataArray(0),
        }, coords={"time": times, "x": source_x, "y": source_y},
    )
    motion["crs"].attrs["spatial_ref"] = source_crs.to_wkt()
    motion.to_netcdf(motion_path)
    motion_hash = hashlib.sha256(motion_path.read_bytes()).hexdigest()
    source_manifest = {
        "dataset_id": "NSIDC-0116", "dataset_version": "4", "doi": "10.5067/INAWUWO7QH7B",
        "complete": True, "requested_period": {"start": "2019-12-31", "end": "2020-01-01"},
        "files": [{"file": motion_path.name, "bytes": motion_path.stat().st_size,
                   "sha256": motion_hash, "time_start": "2019-12-31", "time_end": "2020-01-01"}],
    }
    (motion_dir / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")

    sic_path = tmp_path / "sic.nc"
    sic_values = np.stack([np.full((4, 4), value, dtype=np.float32)
                           for value in (0.2, 0.21, 0.22)])
    sic = xr.Dataset(
        {
            "cdr_seaice_conc": (("time", "y", "x"), sic_values, {"grid_mapping": "crs"}),
            "cdr_seaice_conc_qa_flag": (("time", "y", "x"), np.zeros((3, 4, 4), dtype=np.uint8)),
            "cdr_seaice_conc_interp_temporal_flag": (("time", "y", "x"), np.zeros((3, 4, 4), dtype=np.uint8)),
            "crs": xr.DataArray(0),
        }, coords={"time": np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]"),
                   "x": target_x, "y": target_y},
    )
    sic["crs"].attrs["spatial_ref"] = target_crs.to_wkt()
    sic_path.parent.mkdir(parents=True, exist_ok=True)
    sic.to_netcdf(sic_path)

    aligned_dir = tmp_path / "aligned"
    result = align_archive(motion_dir, sic_path, aligned_dir,
                           date(2019, 12, 31), date(2020, 1, 1))
    assert result["date_count"] == 2
    output_path = tmp_path / "hindcast.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_nsidc0116_advection.py",
                                      "--motion-dir", str(motion_dir),
                                      "--aligned-dir", str(aligned_dir),
                                      "--sic-file", str(sic_path),
                                      "--start", "2020-01-01", "--end", "2020-01-02",
                                      "--fit-through", "2020-01-01",
                                      "--output", str(output_path)])
    assert main() == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["decision_status"] == "research_only_not_navigation_or_operational_forecast"
    assert report["selected_lambda"] == 0.0
    assert report["fit_metrics"]["n_cell_days"] > 0
    assert report["holdout_metrics"]["n_cell_days"] > 0
    assert report["holdout_metrics"]["evaluated_days"] == 1
