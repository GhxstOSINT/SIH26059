from datetime import date

import numpy as np
import pytest
import xarray as xr

from ml.fetch_bootstrap_v4 import select_daily_granules, validate_daily_file


class Granule:
    def __init__(self, name):
        self.meta = {"native-id": name}
        self.name = name

    def __getitem__(self, key):
        return self.meta if key == "meta" else None

    def data_links(self):
        return ["https://data.nsidc.earthdatacloud.nasa.gov/path/" + self.name]


def test_select_daily_south_only_and_require_full_date_inventory():
    south_1 = "NSIDC0079_SEAICE_PS_S25km_20240101_v4.0.nc"
    south_2 = "NSIDC0079_SEAICE_PS_S25km_20240102_v4.0.nc"
    north = "NSIDC0079_SEAICE_PS_N25km_20240101_v4.0.nc"
    chosen = select_daily_granules([Granule(north), Granule(south_2), Granule(south_1)],
                                   date(2024, 1, 1), date(2024, 1, 2))
    assert [item[1] for item in chosen] == [south_1, south_2]
    with pytest.raises(ValueError, match="no Southern"):
        select_daily_granules([Granule(south_1)], date(2024, 1, 1), date(2024, 1, 2))


def test_validate_daily_file_masks_documented_sentinels(tmp_path):
    path = tmp_path / "NSIDC0079_SEAICE_PS_S25km_20240101_v4.0.nc"
    values = np.full((1, 332, 316), 0.5, dtype=np.float32)
    values[0, 0, 0] = 1.1
    values[0, 0, 1] = 1.2
    xr.Dataset({"F17_ICECON": (("t", "y", "x"), values),
                "crs": ((), 0, {"crs_wkt": "test-crs"})},
               coords={"t": [np.datetime64("2024-01-01")],
                       "x": np.arange(316), "y": np.arange(332)}).to_netcdf(path)
    report = validate_daily_file(path, date(2024, 1, 1))
    assert report["valid_cell_count"] == 332 * 316 - 2
    with pytest.raises(ValueError, match="time does not match"):
        validate_daily_file(path, date(2024, 1, 2))
