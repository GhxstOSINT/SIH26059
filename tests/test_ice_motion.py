import numpy as np
import pytest

from ml.ice_motion import (align_nsidc0116_daily_displacement,
                           southern_grid_motion_to_earth)


def test_southern_rotation_matches_documented_cardinal_directions():
    u = np.array([100.0, 100.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 100.0, 100.0])
    lon = np.array([0.0, 90.0, 0.0, 90.0])

    east, north, valid = southern_grid_motion_to_earth(u, v, lon)

    np.testing.assert_allclose(east, [1.0, 0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(north, [0.0, 1.0, 1.0, 0.0], atol=1e-12)
    assert valid.all()


def test_missing_and_documented_low_quality_motion_vectors_are_masked():
    east, north, valid = southern_grid_motion_to_earth(
        np.array([100.0, -9999.0, 100.0, 100.0, 100.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(5),
        np.array([20.0, 20.0, -5.0, 1000.0, -9999.0]),
    )

    np.testing.assert_array_equal(valid, [True, False, False, False, False])
    assert east[0] == 1.0 and north[0] == 0.0
    assert np.isnan(east[1:]).all() and np.isnan(north[1:]).all()


def test_motion_transform_broadcasts_grid_longitudes():
    east, north, valid = southern_grid_motion_to_earth(
        np.ones((2, 1)) * 100.0,
        np.zeros((1, 2)),
        np.array([[0.0, 90.0]]),
    )

    assert east.shape == north.shape == valid.shape == (2, 2)
    np.testing.assert_allclose(east, [[1.0, 0.0], [1.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(north, [[0.0, 1.0], [0.0, 1.0]], atol=1e-12)


def test_nsidc_displacement_alignment_preserves_daily_units_and_masks_low_support():
    pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from affine import Affine

    # One 25-km NSIDC pixel centred on the EPSG:3409 central meridian.
    transform = Affine(25000.0, 0.0, -12500.0, 0.0, -25000.0, -987500.0)
    dx, dy, support = align_nsidc0116_daily_displacement(
        np.array([[100.0]]), np.array([[0.0]]), np.array([[10.0]]),
        transform, transform, (1, 1), source_crs="EPSG:3409",
        target_crs="EPSG:3409", minimum_support=0.75,
    )
    assert support[0, 0] == 1.0
    assert np.isfinite(dx[0, 0]) and np.isfinite(dy[0, 0])
    assert dx[0, 0] > 80000.0  # 1 m/s for 24 hours, eastward near lon=0.
    assert abs(dy[0, 0]) < 100.0

    invalid_x, invalid_y, invalid_support = align_nsidc0116_daily_displacement(
        np.array([[-9999.0]]), np.array([[0.0]]), np.array([[10.0]]),
        transform, transform, (1, 1), source_crs="EPSG:3409",
        target_crs="EPSG:3409", minimum_support=0.75,
    )
    assert invalid_support[0, 0] == 0.0
    assert np.isnan(invalid_x[0, 0]) and np.isnan(invalid_y[0, 0])
