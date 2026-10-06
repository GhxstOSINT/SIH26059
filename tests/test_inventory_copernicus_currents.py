from datetime import date

import numpy as np
import pytest
import xarray as xr

from ml.inventory_copernicus_currents import inventory


def _subset(path, *, dates=("2024-06-01", "2024-06-02"), v_mask_mismatch=False):
    shape = (len(dates), 1, 2, 2)
    u = np.full(shape, 0.1, dtype=np.float32)
    v = np.full(shape, -0.2, dtype=np.float32)
    if v_mask_mismatch:
        v[0, 0, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "uo": (("time", "depth", "latitude", "longitude"), u, {"units": "m s-1"}),
            "vo": (("time", "depth", "latitude", "longitude"), v, {"units": "m s-1"}),
        },
        coords={"time": np.array(dates, dtype="datetime64[ns]"), "depth": [0.494025],
                "latitude": [-67.0, -66.0], "longitude": [-61.0, -60.0]},
    )
    ds.to_netcdf(path)


def test_inventory_accepts_retrospective_surface_currents(tmp_path):
    source = tmp_path / "currents.nc"
    _subset(source)
    report = inventory(source, date(2024, 6, 1), date(2024, 6, 2))
    assert report["period"]["daily_fields"] == 2
    assert report["summary"]["paired_valid_fraction"] == 1.0
    assert report["asof_verified"] is False


def test_inventory_rejects_different_component_masks(tmp_path):
    source = tmp_path / "currents.nc"
    _subset(source, v_mask_mismatch=True)
    with pytest.raises(ValueError, match="different masks"):
        inventory(source, date(2024, 6, 1), date(2024, 6, 2))
