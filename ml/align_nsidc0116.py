"""Align one NSIDC-0116 daily motion field to a SIC reference grid.

This is research preprocessing only. It does not create a forecast, establish
operational data availability, or make sea-ice motion into iceberg motion.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

if __package__:
    from .ice_motion import align_nsidc0116_daily_displacement
else:  # Support the documented ``python ml/align_nsidc0116.py ...`` invocation.
    from ice_motion import align_nsidc0116_daily_displacement


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid_transform(dataset: xr.Dataset, *, allow_ascending_y: bool = False):
    from affine import Affine

    if "x" not in dataset.coords or "y" not in dataset.coords:
        raise ValueError("Both NetCDF grids must provide one-dimensional x/y coordinates.")
    x = np.asarray(dataset.x.values, dtype=np.float64)
    y = np.asarray(dataset.y.values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("Each grid axis must contain at least two one-dimensional cell centres.")
    dx = np.diff(x)
    dy = np.diff(y)
    if not np.allclose(dx, dx[0], rtol=1e-7, atol=1e-5) or not np.allclose(dy, dy[0], rtol=1e-7, atol=1e-5):
        raise ValueError("Rotated, irregular, or nonuniform grids are not supported by this affine workflow.")
    if dx[0] <= 0 or dy[0] == 0 or (dy[0] > 0 and not allow_ascending_y):
        raise ValueError("Grid axes must be ordered x increasing and y decreasing unless source y reversal is explicitly allowed.")
    reverse_y = bool(dy[0] > 0)
    if reverse_y:
        y = y[::-1]
        dy = np.diff(y)
    transform = Affine(float(dx[0]), 0.0, float(x[0] - dx[0] / 2.0),
                       0.0, float(dy[0]), float(y[0] - dy[0] / 2.0))
    return transform, (int(y.size), int(x.size)), reverse_y


def _crs(dataset: xr.Dataset) -> str:
    for variable in dataset.variables.values():
        attrs = variable.attrs
        for name in ("spatial_ref", "crs_wkt", "epsg_code"):
            if attrs.get(name):
                return str(attrs[name])
    raise ValueError("NetCDF grid is missing a CRS definition (spatial_ref/crs_wkt/epsg_code).")


def align_nsidc0116_file(motion_path: Path, reference_path: Path,
                         motion_date: date, *, motion_sha256: str | None = None,
                         sic_reference_sha256: str | None = None) -> xr.Dataset:
    """Return displacement components/support on the SIC reference x/y grid."""
    from pyproj import CRS

    motion_path = Path(motion_path)
    reference_path = Path(reference_path)
    motion_sha256 = motion_sha256 or sha256_file(motion_path)
    sic_reference_sha256 = sic_reference_sha256 or sha256_file(reference_path)
    if (len(motion_sha256) != 64 or len(sic_reference_sha256) != 64
            or any(character not in "0123456789abcdef" for character in motion_sha256.lower())
            or any(character not in "0123456789abcdef" for character in sic_reference_sha256.lower())):
        raise ValueError("Input SHA-256 values must be 64 hexadecimal characters.")
    with xr.open_dataset(motion_path, decode_cf=True) as motion_source, \
            xr.open_dataset(reference_path, decode_cf=True) as reference_source:
        for required in ("u", "v", "icemotion_error_estimate"):
            if required not in motion_source:
                raise ValueError(f"NSIDC-0116 file lacks required quality-screened field {required}.")
        if "time" not in motion_source.coords:
            raise ValueError("NSIDC-0116 file has no time coordinate.")
        matching = [index for index, value in enumerate(motion_source.time.values)
                    if str(value)[:10] == motion_date.isoformat()]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one NSIDC-0116 record for {motion_date}; found {len(matching)}.")
        source_crs = _crs(motion_source)
        target_crs = _crs(reference_source)
        # Parse definitions up front; downstream GDAL/PROJ failures then carry a clear error.
        source_crs = CRS.from_user_input(source_crs).to_wkt()
        target_crs = CRS.from_user_input(target_crs).to_wkt()
        source_transform, _, reverse_source_y = _grid_transform(motion_source, allow_ascending_y=True)
        target_transform, target_shape, _ = _grid_transform(reference_source)
        index = matching[0]
        def field(name):
            value = motion_source[name]
            if "time" in value.dims:
                value = value.isel(time=index)
            value = value.squeeze(drop=True)
            if value.ndim != 2 or set(value.dims) != {"y", "x"}:
                raise ValueError(f"NSIDC-0116 {name} must resolve to a 2-D y/x field.")
            array = np.asarray(value.transpose("y", "x").values, dtype=np.float64)
            return array[::-1, :] if reverse_source_y else array

        if "cdr_seaice_conc" not in reference_source:
            raise ValueError("SIC reference must contain cdr_seaice_conc to verify its y/x grid.")
        sic = reference_source["cdr_seaice_conc"]
        if not {"y", "x"}.issubset(sic.dims):
            raise ValueError("SIC reference concentration does not use the x/y grid coordinates.")
        dx, dy, support = align_nsidc0116_daily_displacement(
            field("u"), field("v"), field("icemotion_error_estimate"),
            source_transform, target_transform, target_shape,
            source_crs=source_crs, target_crs=target_crs,
        )
        result = xr.Dataset(
            {
                "ice_motion_dx_24h": (("y", "x"), dx),
                "ice_motion_dy_24h": (("y", "x"), dy),
                "ice_motion_support_fraction": (("y", "x"), support.astype(np.float32)),
                "crs": xr.DataArray(0, attrs={"spatial_ref": target_crs,
                                               "crs_wkt": target_crs}),
            },
            coords={"x": reference_source.x.values, "y": reference_source.y.values},
            attrs={
                "title": "NSIDC-0116 daily ice-motion displacement aligned to SIC reference grid",
                "decision_status": "research_preprocessing_only",
                "motion_date": motion_date.isoformat(),
                "motion_dataset": "NSIDC-0116 v4",
                "motion_doi": "10.5067/INAWUWO7QH7B",
                "source_crs": source_crs,
                "source_y_reversed_for_north_up_raster": int(reverse_source_y),
                "target_crs": target_crs,
                "vector_transform": "Southern Hemisphere along-grid cm/s -> earth-relative -> WGS84 geodesic endpoint after 86400 s -> target-grid x/y metres per day",
                "resampling": "validity-weighted bilinear; support fraction retained; cells below 0.75 support are NaN",
                "quality_screen": "exclude non-finite/fill u/v; require finite icemotion_error_estimate in [0,1000) cm/s",
                "source_motion_file": motion_path.name,
                "source_motion_sha256": motion_sha256,
                "sic_reference_file": reference_path.name,
                "sic_reference_sha256": sic_reference_sha256,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "warning": "Historical/research motion preprocessing only; not a forecast, iceberg drift estimate, route-safety assessment, or operationally available input guarantee.",
            },
        )
        result["ice_motion_dx_24h"].attrs.update({"long_name": "24-hour sea-ice displacement in target-grid x direction", "units": "m day-1"})
        result["ice_motion_dy_24h"].attrs.update({"long_name": "24-hour sea-ice displacement in target-grid y direction", "units": "m day-1"})
        result["ice_motion_support_fraction"].attrs.update({"long_name": "bilinear valid-source support fraction", "units": "1", "valid_min": 0.0, "valid_max": 1.0})
        for variable_name in ("ice_motion_dx_24h", "ice_motion_dy_24h", "ice_motion_support_fraction"):
            result[variable_name].attrs["grid_mapping"] = "crs"
        result.x.attrs.update({"standard_name": "projection_x_coordinate", "units": "m"})
        result.y.attrs.update({"standard_name": "projection_y_coordinate", "units": "m"})
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_file", type=Path)
    parser.add_argument("sic_reference_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--date", type=date.fromisoformat, required=True,
                        help="Exact daily NSIDC time record to align (YYYY-MM-DD).")
    args = parser.parse_args()
    output = args.output_file.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    try:
        aligned = align_nsidc0116_file(args.motion_file, args.sic_reference_file, args.date)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                             suffix=".tmp.nc", delete=False) as handle:
                temporary = Path(handle.name)
            aligned.to_netcdf(temporary)
            os.replace(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        print(json.dumps({"complete": True, "output": str(output),
                          "motion_date": args.date.isoformat(),
                          "valid_cells": int(np.isfinite(aligned.ice_motion_dx_24h.values).sum()),
                          "total_cells": int(aligned.ice_motion_dx_24h.size),
                          "source_motion_sha256": aligned.attrs["source_motion_sha256"],
                          "sic_reference_sha256": aligned.attrs["sic_reference_sha256"]}, indent=2))
    except Exception as exc:
        parser.exit(1, f"NSIDC-0116 alignment failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
