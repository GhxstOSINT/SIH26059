"""Small, explicit transforms for NSIDC-0116 Southern Hemisphere vectors.

These helpers operate on arrays already decoded from the native NSIDC-0116
EPSG:3409 grid. They do not reproject, interpolate, or align vectors to SIC.
"""
from __future__ import annotations

import numpy as np


NSIDC0116_FILL_VALUE = -9999.0


def southern_grid_motion_to_earth(
    u_cm_s: np.ndarray,
    v_cm_s: np.ndarray,
    longitude_degrees_e: np.ndarray,
    error_estimate_cm_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert NSIDC-0116 SH along-grid components to east/north m/s.

    NSIDC's Southern Hemisphere convention is E = u*cos(lon) - v*sin(lon),
    N = u*sin(lon) + v*cos(lon). Input components are cm/s. Missing component
    values (-9999) are masked. If the daily error estimate is supplied, its
    negative coastline values and >=1000 values (nearest input >1250 km) are
    also masked. This is conservative screening, not a quality uncertainty
    model; the returned mask must be preserved by downstream processing.
    """
    u, v, lon = np.broadcast_arrays(
        np.asarray(u_cm_s, dtype=np.float64),
        np.asarray(v_cm_s, dtype=np.float64),
        np.asarray(longitude_degrees_e, dtype=np.float64),
    )
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(lon)
    valid &= (u != NSIDC0116_FILL_VALUE) & (v != NSIDC0116_FILL_VALUE)
    if error_estimate_cm_s is not None:
        error = np.broadcast_to(np.asarray(error_estimate_cm_s, dtype=np.float64), u.shape)
        valid &= np.isfinite(error) & (error != NSIDC0116_FILL_VALUE)
        valid &= (error >= 0.0) & (error < 1000.0)

    angle = np.deg2rad(lon)
    east = (u * np.cos(angle) - v * np.sin(angle)) / 100.0
    north = (u * np.sin(angle) + v * np.cos(angle)) / 100.0
    east = np.where(valid, east, np.nan)
    north = np.where(valid, north, np.nan)
    return east, north, valid


def align_nsidc0116_daily_displacement(
    u_cm_s: np.ndarray,
    v_cm_s: np.ndarray,
    error_estimate_cm_s: np.ndarray,
    source_transform,
    target_transform,
    target_shape: tuple[int, int],
    *,
    source_crs: str = "EPSG:3409",
    target_crs: str = "EPSG:3412",
    minimum_support: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align NSIDC-0116 vectors as target-grid x/y displacements per day.

    Vectors are first rotated from NSIDC's Southern Hemisphere grid convention
    to earth-relative east/north. A geodesic 24-hour endpoint is then projected
    to the target CRS; this is important because simply warping east/north
    components as scalar rasters would ignore the target projection's local
    orientation. The resulting displacement components are reprojected using
    validity-weighted bilinear interpolation. Cells with less than
    ``minimum_support`` valid interpolation support are returned as NaN.

    This is a preprocessing transform, not a forecast or a quality uncertainty
    model. The daily NSIDC motion vectors are historical/research inputs; users
    must preserve the returned support mask and product provenance.
    """
    if not 0.0 < minimum_support <= 1.0:
        raise ValueError("minimum_support must be in (0, 1].")
    u, v = np.broadcast_arrays(np.asarray(u_cm_s, dtype=np.float64),
                               np.asarray(v_cm_s, dtype=np.float64))
    error = np.broadcast_to(np.asarray(error_estimate_cm_s, dtype=np.float64), u.shape)
    if u.ndim != 2 or min(u.shape) < 1:
        raise ValueError("NSIDC-0116 components must be non-empty 2-D grids.")
    try:
        from affine import Affine
        from pyproj import Geod, Transformer
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional geospatial dependencies before aligning NSIDC-0116 motion."
        ) from exc

    source_transform = Affine(*source_transform[:6])
    target_transform = Affine(*target_transform[:6])
    rows, cols = np.indices(u.shape, dtype=np.float64)
    source_x = (source_transform.c + (cols + 0.5) * source_transform.a
                + (rows + 0.5) * source_transform.b)
    source_y = (source_transform.f + (cols + 0.5) * source_transform.d
                + (rows + 0.5) * source_transform.e)
    to_geographic = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_geographic.transform(source_x, source_y)
    east, north, valid = southern_grid_motion_to_earth(u, v, lon, error)
    valid &= np.isfinite(lat) & (lat >= -90.0) & (lat <= 90.0)

    speed = np.hypot(east, north)
    azimuth = np.rad2deg(np.arctan2(east, north))
    geod = Geod(ellps="WGS84")
    end_lon, end_lat, _ = geod.fwd(lon, lat, azimuth, speed * 86400.0)
    to_target = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    origin_x, origin_y = to_target.transform(lon, lat)
    end_x, end_y = to_target.transform(end_lon, end_lat)
    displacement_x = np.where(valid, end_x - origin_x, 0.0).astype(np.float32)
    displacement_y = np.where(valid, end_y - origin_y, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    out_shape = tuple(int(size) for size in target_shape)
    if len(out_shape) != 2 or min(out_shape) < 1:
        raise ValueError("target_shape must contain two positive dimensions.")
    out_x_num = np.zeros(out_shape, dtype=np.float32)
    out_y_num = np.zeros(out_shape, dtype=np.float32)
    support = np.zeros(out_shape, dtype=np.float32)
    options = {"src_transform": source_transform, "src_crs": source_crs,
               "dst_transform": target_transform, "dst_crs": target_crs,
               "resampling": Resampling.bilinear, "init_dest_nodata": True}
    reproject(displacement_x * weights, out_x_num, **options)
    reproject(displacement_y * weights, out_y_num, **options)
    reproject(weights, support, **options)
    accepted = np.isfinite(support) & (support >= minimum_support)
    out_x = np.full(out_shape, np.nan, dtype=np.float32)
    out_y = np.full(out_shape, np.nan, dtype=np.float32)
    np.divide(out_x_num, support, out=out_x, where=accepted)
    np.divide(out_y_num, support, out=out_y, where=accepted)
    return out_x, out_y, np.where(np.isfinite(support), support, 0.0)
