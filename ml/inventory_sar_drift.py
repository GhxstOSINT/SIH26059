"""Validate and checksum a bounded Copernicus Antarctic SAR ice-drift subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


DATASET_ID = "cmems_sat-si_glo_drift_nrt_south_d"
PRODUCT_ID = "SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS_011_006"
PRODUCT_URL = f"https://data.marine.copernicus.eu/product/{PRODUCT_ID}/description"


def inventory(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Need a regular SAR ice-drift NetCDF file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with xr.open_dataset(path, decode_cf=True, engine="netcdf4") as ds:
        required = {"dX_mean", "dY_mean", "dX_std", "dY_std", "status_flag",
                    "qos_mean_dX", "qos_mean_dY", "qos_stddev_dX", "qos_stddev_dY"}
        missing = sorted(required - set(ds.variables))
        if missing:
            raise ValueError("Missing SAR ice-drift fields: " + ", ".join(missing))
        if ds.attrs.get("title") != "Antarctic Sea Ice Drift, L4 SAR, 10km daily":
            raise ValueError("Unexpected SAR ice-drift product title")
        if ds.dX_mean.dims != ("time", "latitude", "longitude") or ds.dY_mean.shape != ds.dX_mean.shape:
            raise ValueError("Unexpected drift grid dimensions")
        if ds.dX_mean.attrs.get("units") != "km" or ds.dY_mean.attrs.get("units") != "km":
            raise ValueError("Drift components must be expressed in kilometres")
        times = np.asarray(ds.time.values)
        if times.size < 1 or np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "s")):
            raise ValueError("Drift timestamps are missing or unordered")
        dx = np.asarray(ds.dX_mean.values, dtype=np.float32)
        dy = np.asarray(ds.dY_mean.values, dtype=np.float32)
        valid = np.isfinite(dx) & np.isfinite(dy)
        if np.any(valid & ((np.abs(dx) > 250) | (np.abs(dy) > 250))):
            raise ValueError("Drift displacement exceeds documented valid range")
        lat = np.asarray(ds.latitude.values, dtype=np.float64)
        lon = np.asarray(ds.longitude.values, dtype=np.float64)
        if not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
            raise ValueError("Geographic coordinates are not ascending")
        timestamp_values = [np.datetime_as_string(value, unit="s") + "Z" for value in times]
        return {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "product_id": PRODUCT_ID,
            "product_url": PRODUCT_URL,
            "dataset_title": ds.attrs["title"],
            "provider": "Copernicus Marine / DTU Space",
            "source_file": path.name,
            "source_bytes": path.stat().st_size,
            "source_sha256": digest,
            "inventory_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "timestamps_utc": timestamp_values,
            "grid": {"crs": "EPSG:4326", "shape": [len(times), len(lat), len(lon)],
                     "bbox_west_east_south_north": [float(lon[0]), float(lon[-1]),
                                                       float(lat[0]), float(lat[-1])],
                     "nominal_resolution_km": 10},
            "valid_vector_cells": int(valid.sum()),
            "valid_vector_fraction": float(valid.mean()),
            "displacement_km": {
                "median": float(np.nanmedian(np.hypot(dx[valid], dy[valid]))) if np.any(valid) else None,
                "p95": float(np.nanquantile(np.hypot(dx[valid], dy[valid]), 0.95)) if np.any(valid) else None,
            },
            "quality_fields": sorted(required - {"dX_mean", "dY_mean"}),
            "catalog_product_status": ds.attrs.get("product_status"),
            "decision_status": "research_motion_evidence_only",
            "limitations": [
                "This is sea-ice displacement, not iceberg displacement or vessel drift.",
                "The provider labels the file metadata preoperational; it is not an approved navigation product.",
                "Spatially correlated grid cells are not independent route outcomes.",
                "The file's global start/stop attributes can lag the aggregated coordinate range; verified time coordinates govern this subset.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "timestamps_utc": report["timestamps_utc"],
                      "valid_vector_cells": report["valid_vector_cells"],
                      "decision_status": report["decision_status"]}, indent=2))


if __name__ == "__main__":
    main()
