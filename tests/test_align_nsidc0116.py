from datetime import date

import numpy as np
import pytest
import xarray as xr

from ml.align_nsidc0116 import align_nsidc0116_file


def _write_test_inputs(tmp_path):
    pyproj = pytest.importorskip("pyproj")
    pytest.importorskip("rasterio")
    from pyproj import CRS, Transformer

    source_crs = CRS.from_epsg(3409)
    target_crs = CRS.from_epsg(3412)
    source_x = np.arange(-37500.0, 50000.0, 25000.0)
    source_y = np.arange(-962500.0, -1050000.0, -25000.0)
    center_x, center_y = Transformer.from_crs(source_crs, target_crs, always_xy=True).transform(0.0, -1000000.0)
    target_x = center_x + np.arange(-37500.0, 50000.0, 25000.0)
    target_y = center_y + np.arange(-37500.0, 50000.0, 25000.0)[::-1]

    motion = xr.Dataset(
        {
            "u": (("time", "y", "x"), np.broadcast_to(
                np.array([60.0, 80.0, 100.0, 120.0])[None, :, None], (1, 4, 4)).copy()),
            "v": (("time", "y", "x"), np.zeros((1, 4, 4))),
            "icemotion_error_estimate": (("time", "y", "x"), np.full((1, 4, 4), 10.0)),
            "crs": xr.DataArray(0),
        },
        coords={"time": np.array(["2020-01-02"], dtype="datetime64[ns]"),
                "x": source_x, "y": source_y},
    )
    motion["u"].attrs["units"] = "cm/s"
    motion["v"].attrs["units"] = "cm/s"
    motion["icemotion_error_estimate"].attrs["units"] = "cm/s"
    motion["crs"].attrs.update({"spatial_ref": source_crs.to_wkt(),
                                "grid_mapping_name": "lambert_azimuthal_equal_area"})
    reference = xr.Dataset(
        {"cdr_seaice_conc": (("y", "x"), np.zeros((4, 4)), {"grid_mapping": "crs"}),
         "crs": xr.DataArray(0)},
        coords={"x": target_x, "y": target_y},
    )
    reference["crs"].attrs.update({"spatial_ref": target_crs.to_wkt(),
                                   "grid_mapping_name": "polar_stereographic"})
    motion_path = tmp_path / "motion.nc"
    reference_path = tmp_path / "sic-reference.nc"
    motion.to_netcdf(motion_path)
    reference.to_netcdf(reference_path)
    return motion_path, reference_path


def test_file_alignment_reprojects_geodesic_displacement_and_records_provenance(tmp_path):
    motion, reference = _write_test_inputs(tmp_path)
    result = align_nsidc0116_file(motion, reference, date(2020, 1, 2))

    assert result.sizes == {"y": 4, "x": 4}
    assert result.attrs["source_crs"].startswith("PROJCRS[")
    assert result.attrs["target_crs"].startswith("PROJCRS[")
    assert result.crs.attrs["spatial_ref"] == result.attrs["target_crs"]
    assert result.ice_motion_dx_24h.attrs["grid_mapping"] == "crs"
    assert len(result.attrs["source_motion_sha256"]) == 64
    assert len(result.attrs["sic_reference_sha256"]) == 64
    assert result.attrs["decision_status"] == "research_preprocessing_only"
    valid = np.isfinite(result.ice_motion_dx_24h.values)
    assert valid.any()
    assert np.nanmedian(result.ice_motion_dx_24h.values) > 50000.0
    support = result.ice_motion_support_fraction.values
    assert np.all((support >= 0.0) & (support <= 1.0))


def test_file_alignment_requires_exact_date_and_error_quality_field(tmp_path):
    motion, reference = _write_test_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly one NSIDC-0116 record"):
        align_nsidc0116_file(motion, reference, date(2020, 1, 3))

    with xr.open_dataset(motion) as source:
        invalid_motion = source.drop_vars("icemotion_error_estimate").load()
    invalid_path = tmp_path / "motion-no-error.nc"
    invalid_motion.to_netcdf(invalid_path)
    with pytest.raises(ValueError, match="icemotion_error_estimate"):
        align_nsidc0116_file(invalid_path, reference, date(2020, 1, 2))


def test_file_alignment_accepts_native_ascending_source_y_without_changing_vectors(tmp_path):
    motion, reference = _write_test_inputs(tmp_path)
    baseline = align_nsidc0116_file(motion, reference, date(2020, 1, 2))
    with xr.open_dataset(motion) as source:
        ascending = source.sortby("y").load()
    ascending_path = tmp_path / "motion-ascending-y.nc"
    ascending.to_netcdf(ascending_path)

    result = align_nsidc0116_file(ascending_path, reference, date(2020, 1, 2))
    assert result.attrs["source_y_reversed_for_north_up_raster"] == 1
    np.testing.assert_allclose(result.ice_motion_dx_24h.values,
                               baseline.ice_motion_dx_24h.values, equal_nan=True)
    np.testing.assert_allclose(result.ice_motion_dy_24h.values,
                               baseline.ice_motion_dy_24h.values, equal_nan=True)
