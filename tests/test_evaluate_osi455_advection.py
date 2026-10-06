from datetime import date

import numpy as np
import xarray as xr

from ml.evaluate_osi455_advection import (backtrace_coordinates, day_metrics,
                                          geographic_group_masks, latest_causal_motion_date,
                                          manifest_covers_experiment, read_sic, stratified_metrics)


def test_read_sic_excludes_missing_interpolated_and_bad_values():
    ds = xr.Dataset(
        {
            "cdr_seaice_conc": (("time", "y", "x"), np.array([[[0.1, 0.2, 0.3, 0.4, 1.2]]], dtype=np.float32)),
            "cdr_seaice_conc_qa_flag": (("time", "y", "x"), np.array([[[0, 8, 32, 64, 0]]], dtype=np.uint8)),
            "cdr_seaice_conc_interp_temporal_flag": (("time", "y", "x"), np.array([[[0, 0, 0, 0, 0]]], dtype=np.uint8)),
        },
        coords={"time": [np.datetime64("2020-09-01")], "y": [0], "x": np.arange(5)},
    )
    values, valid = read_sic(ds, date(2020, 9, 1))
    assert valid.tolist() == [[True, False, False, False, False]]
    assert values[0, 0] == np.float32(0.1)
    assert np.isnan(values[0, 1:]).all()


def test_day_metrics_reports_pooled_mae_and_rmse():
    persistence_error = np.array([[0.0, 0.2]], dtype=np.float32)
    advection_error = np.array([[0.0, 0.1]], dtype=np.float32)
    mask = np.array([[True, True]])
    metrics = day_metrics([(persistence_error, advection_error, mask)])
    assert metrics["n_cells"] == 2
    assert np.isclose(metrics["persistence_mae"], 0.1)
    assert np.isclose(metrics["advection_mae"], 0.05)
    assert np.isclose(metrics["persistence_rmse"], np.sqrt(0.02))


def test_backtrace_accounts_for_southward_increasing_row_index():
    dx = np.full((2, 2), 75.0, dtype=np.float32)
    dy = np.full((2, 2), 75.0, dtype=np.float32)
    rows, cols = backtrace_coordinates(dx, dy)
    assert rows[0, 0] == 1.0  # northward drift: previous source lies one row south
    assert cols[0, 0] == -1.0  # eastward drift: previous source lies one column west


def test_motion_file_is_latest_completed_vector_before_sic_issue():
    assert latest_causal_motion_date(date(2020, 9, 1)) == date(2020, 8, 31)


def test_manifest_may_be_a_larger_complete_archive_but_must_cover_causal_days():
    manifest = {"complete": True,
                "requested_period": {"start": "2016-12-31", "end": "2020-12-31"}}
    assert manifest_covers_experiment(manifest, date(2017, 1, 1), date(2017, 12, 31))
    assert manifest_covers_experiment(manifest, date(2020, 1, 1), date(2020, 12, 31))
    assert not manifest_covers_experiment(manifest, date(2016, 12, 31), date(2017, 12, 31))
    manifest["complete"] = False
    assert not manifest_covers_experiment(manifest, date(2017, 1, 1), date(2017, 12, 31))


def test_geographic_groups_normalize_longitude_and_separate_latitude_bands():
    groups = geographic_group_masks(np.array([45, 135, -135, -45]),
                                    np.array([-55, -65, -75, -85]))
    for name in ("longitude_000_090E", "longitude_090_180E", "longitude_180_270E", "longitude_270_360E",
                 "latitude_50S_60S", "latitude_60S_70S", "latitude_70S_80S", "latitude_80S_90S"):
        assert groups[name].sum() == 1


def test_stratified_metrics_scores_only_cells_in_each_group():
    longitude = np.array([[45, 135, -135, -45]])
    latitude = np.array([[-55, -65, -75, -85]])
    groups = geographic_group_masks(longitude, latitude)
    base = np.zeros((1, 4), dtype=np.float32)
    truth = np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32)
    advected = np.array([[0.0, 0.2, 0.1, 0.3]], dtype=np.float32)
    valid = np.ones((1, 4), dtype=bool)
    result = stratified_metrics([(date(2020, 1, 1), base, truth, advected, valid)], 1.0, groups)
    assert result["longitude_000_090E"]["n_cells"] == 1
    assert np.isclose(result["longitude_000_090E"]["persistence_mae"], 0.1)
    assert np.isclose(result["longitude_000_090E"]["advection_mae"], 0.1)
    assert result["longitude_270_360E"]["n_cells"] == 1
    assert np.isclose(result["longitude_270_360E"]["advection_mae"], 0.1)
    assert result["latitude_80S_90S"]["n_cells"] == 1
