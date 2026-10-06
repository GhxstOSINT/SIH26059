from datetime import date

import numpy as np
import pytest
import xarray as xr

from ml.inventory_copernicus_sic import inventory


def _subset(path, *, dates=("2024-06-01", "2024-06-02"), uncertainty=5.0):
    shape = (len(dates), 2, 2)
    ds = xr.Dataset(
        {
            "ice_conc": (("time", "latitude", "longitude"), np.full(shape, 25.0), {"units": "%"}),
            "status_flag": (("time", "latitude", "longitude"), np.zeros(shape, dtype=np.int16)),
            "total_uncertainty": (("time", "latitude", "longitude"), np.full(shape, uncertainty)),
        },
        coords={"time": np.array(dates, dtype="datetime64[ns]"),
                "latitude": [-67.0, -66.0], "longitude": [-61.0, -60.0]},
    )
    ds.to_netcdf(path)


def test_inventory_records_daily_coverage_but_not_asof(tmp_path):
    source = tmp_path / "south.nc"
    _subset(source)
    report = inventory(source, date(2024, 6, 1), date(2024, 6, 2))
    assert report["period"]["daily_fields"] == 2
    assert report["summary"]["valid_fraction"] == 1.0
    assert report["published_at_utc"] is None
    assert report["asof_verified"] is False


def test_inventory_rejects_missing_day(tmp_path):
    source = tmp_path / "south.nc"
    _subset(source, dates=("2024-06-01", "2024-06-03"))
    with pytest.raises(ValueError, match="every requested daily field"):
        inventory(source, date(2024, 6, 1), date(2024, 6, 3))


def test_inventory_rejects_bad_uncertainty(tmp_path):
    source = tmp_path / "south.nc"
    _subset(source, uncertainty=-1.0)
    with pytest.raises(ValueError, match="Invalid uncertainty"):
        inventory(source, date(2024, 6, 1), date(2024, 6, 2))
