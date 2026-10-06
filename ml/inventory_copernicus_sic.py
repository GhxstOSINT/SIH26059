"""Verify a Copernicus South SIC subset before any training alignment.

The subset's observation date is not its historical publication timestamp.
Nothing produced here establishes as-of availability or navigation readiness.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


DATASET_ID = "osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m"
DATASET_VERSION = "202304"
PRODUCT_URL = "https://data.marine.copernicus.eu/product/SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS_011_001/services"


def inventory(path: Path, start: date, end: date) -> dict:
    if not path.is_file() or path.is_symlink() or end < start:
        raise ValueError("Need a regular subset file and valid period")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with xr.open_dataset(path) as ds:
        if not all(name in ds for name in ("ice_conc", "status_flag", "total_uncertainty", "time", "latitude", "longitude")):
            raise ValueError("Missing required South SIC variables or coordinates")
        if ds.ice_conc.attrs.get("units") != "%":
            raise ValueError("Unexpected SIC units")
        if ds.ice_conc.dims != ("time", "latitude", "longitude"):
            raise ValueError("Unexpected SIC dimensions")
        if ds.status_flag.shape != ds.ice_conc.shape or ds.total_uncertainty.shape != ds.ice_conc.shape:
            raise ValueError("Flags and uncertainty must share the SIC grid")
        expected = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
        observed = [date.fromisoformat(np.datetime_as_string(value, unit="D")) for value in ds.time.values]
        if observed != expected:
            raise ValueError("Subset does not contain every requested daily field in order")
        sic = np.asarray(ds.ice_conc.values, dtype=np.float32)
        uncertainty = np.asarray(ds.total_uncertainty.values, dtype=np.float32)
        flags = np.asarray(ds.status_flag.values)
        valid = np.isfinite(sic) & (sic >= 0) & (sic <= 100)
        if np.any(np.isfinite(sic) & ~valid):
            raise ValueError("Out-of-range SIC values")
        if np.any(valid & (~np.isfinite(uncertainty) | (uncertainty < 0) | (uncertainty > 100))):
            raise ValueError("Invalid uncertainty on SIC cells")
        counts = {}
        for value, count in zip(*np.unique(flags[np.isfinite(flags)].astype(np.int64), return_counts=True)):
            counts[str(int(value))] = int(count)
        daily = [{"date": day.isoformat(), "valid_fraction": float(mask.mean()),
                  "valid_cells": int(mask.sum())} for day, mask in zip(observed, valid)]
        lat = np.asarray(ds.latitude.values)
        lon = np.asarray(ds.longitude.values)
        if lat.ndim != 1 or lon.ndim != 1 or not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
            raise ValueError("Unexpected geographic coordinate order")
        return {
            "schema_version": 1,
            "scope": "research_source_inventory",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "product_url": PRODUCT_URL,
            "source_file": path.name,
            "source_bytes": path.stat().st_size,
            "source_sha256": digest.hexdigest(),
            "inventory_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "period": {"start_date": start.isoformat(), "end_date": end.isoformat(), "daily_fields": len(daily)},
            "subset_grid": {"crs": "EPSG:4326", "shape": [len(lat), len(lon)],
                            "latitude_range": [float(lat[0]), float(lat[-1])],
                            "longitude_range": [float(lon[0]), float(lon[-1])]},
            "variables": {"ice_conc": "%", "status_flag": "bit field", "total_uncertainty": "%"},
            "summary": {"valid_sic_cells": int(valid.sum()), "total_grid_cells": int(valid.size),
                        "valid_fraction": float(valid.mean()), "status_flag_counts": counts},
            "daily": daily,
            "published_at_utc": None,
            "asof_verified": False,
            "limitations": [
                "Historical source publication times are not in this subset; observation dates cannot certify forecast-origin availability.",
                "The downloaded geographic subset is not yet reprojected onto the VIIRS EPSG:3976 grid.",
                "Raw finite coverage is not an approved source-specific quality mask or route-corridor coverage measure.",
                "Not a trained forecast, vessel-safety label, or navigation clearance.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.source, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "summary": result["summary"],
                      "asof_verified": result["asof_verified"]}, indent=2))


if __name__ == "__main__":
    main()
