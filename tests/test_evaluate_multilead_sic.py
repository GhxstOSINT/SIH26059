import datetime as dt

import numpy as np
import xarray as xr

from ml.evaluate_multilead_sic import evaluate, recursive_step


def test_recursive_step_uses_both_lags_clips_and_preserves_missing_cells():
    previous = np.array([[0.4, 0.8]], dtype=np.float32)
    older = np.array([[0.2, 0.6]], dtype=np.float32)
    valid = np.array([[True, False]])
    coefficients = np.array([0.1, 1.2, 0.5, 0, 0], dtype=np.float64)

    result = recursive_step(previous, older, dt.date(2020, 1, 2), coefficients, valid)

    assert np.isclose(result[0, 0], 0.68)
    assert np.isnan(result[0, 1])


def test_frozen_recursive_holdout_reports_each_requested_lead(tmp_path):
    start = dt.date(2020, 1, 1)
    for offset in range(10):
        day = start + dt.timedelta(days=offset)
        value = np.float32(0.2 + 0.03 * offset)
        dataset = xr.Dataset({"sic": (("y", "x"), np.full((2, 3), value, dtype=np.float32))})
        dataset.to_netcdf(tmp_path / f"sic_{day:%Y%m%d}.nc")

    model = {
        "model_id": "test-lag",
        "coefficients": [0.1, 0.5, 0.2, 0, 0],
        "training": {"selected_period_end": "2020-01-02"},
    }
    report = evaluate(tmp_path, model, (1, 3, 5, 7))

    assert report["status"] == "frozen_recursive_holdout_diagnostic"
    assert report["evaluation_method"]["target_scope"].startswith("chronological targets strictly after")
    assert report["by_lead"]["1"]["period"]["first_target_date"] == "2020-01-03"
    assert report["by_lead"]["1"]["valid_cell_samples"] == 6 * 8
    assert report["by_lead"]["7"]["target_days"] == 2
    assert report["by_lead"]["7"]["valid_cell_samples"] == 6 * 2
