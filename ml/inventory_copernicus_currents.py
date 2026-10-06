"""Verify a retrospective Copernicus GLORYS surface-current subset."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
DATASET_VERSION = "202311"
PRODUCT_URL = "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/services"


def inventory(path: Path, start: date, end: date) -> dict:
    if not path.is_file() or path.is_symlink() or end < start:
        raise ValueError("Need a regular current subset file and valid period")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with xr.open_dataset(path) as ds:
        if not all(name in ds for name in ("uo", "vo", "time", "depth", "latitude", "longitude")):
            raise ValueError("Missing current fields or coordinates")
        if ds.uo.dims != ("time", "depth", "latitude", "longitude") or ds.vo.shape != ds.uo.shape:
            raise ValueError("Unexpected current dimensions")
        if ds.sizes["depth"] != 1 or not 0 <= float(ds.depth.values[0]) <= 1:
            raise ValueError("Expected one near-surface depth")
        if ds.uo.attrs.get("units") != "m s-1" or ds.vo.attrs.get("units") != "m s-1":
            raise ValueError("Unexpected current units")
        expected = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
        observed = [date.fromisoformat(np.datetime_as_string(value, unit="D")) for value in ds.time.values]
        if observed != expected:
            raise ValueError("Current subset has missing or unordered dates")
        u = np.asarray(ds.uo.values, dtype=np.float32)
        v = np.asarray(ds.vo.values, dtype=np.float32)
        paired = np.isfinite(u) & np.isfinite(v)
        if np.any(np.isfinite(u) ^ np.isfinite(v)):
            raise ValueError("Current components have different masks")
        if np.any(paired & ((np.abs(u) > 5) | (np.abs(v) > 5))):
            raise ValueError("Implausible current values")
        lat = np.asarray(ds.latitude.values)
        lon = np.asarray(ds.longitude.values)
        if lat.ndim != 1 or lon.ndim != 1 or not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
            raise ValueError("Unexpected coordinate order")
        daily = [{"date": day.isoformat(), "paired_valid_fraction": float(mask.mean()),
                  "paired_valid_cells": int(mask.sum())} for day, mask in zip(observed, paired[:, 0])]
        return {
            "schema_version": 1,
            "scope": "retrospective_reanalysis_inventory",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "product_url": PRODUCT_URL,
            "source_file": path.name,
            "source_bytes": path.stat().st_size,
            "source_sha256": digest.hexdigest(),
            "inventory_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "period": {"start_date": start.isoformat(), "end_date": end.isoformat(), "daily_fields": len(daily)},
            "subset_grid": {"crs": "EPSG:4326", "shape": [len(lat), len(lon)],
                            "surface_depth_m": float(ds.depth.values[0]),
                            "latitude_range": [float(lat[0]), float(lat[-1])],
                            "longitude_range": [float(lon[0]), float(lon[-1])]},
            "variables": {"uo": "m s-1", "vo": "m s-1"},
            "summary": {"paired_valid_cells": int(paired.sum()), "total_grid_cells": int(paired.size),
                        "paired_valid_fraction": float(paired.mean())},
            "daily": daily,
            "published_at_utc": None,
            "asof_verified": False,
            "limitations": [
                "GLORYS is a retrospective model reanalysis, not a historically available operational forecast field.",
                "Target-day currents may assimilate later observations; no forecast-origin feature use is authorized.",
                "This geographic subset has not been aligned to the VIIRS or Copernicus SIC grids.",
                "Ocean currents alone do not establish iceberg drift or safe navigation.",
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
