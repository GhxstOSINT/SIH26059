"""Inventory an original GLO12 forecast file without confusing it with later analysis.

Use with ``copernicusmarine get`` original files, not the Subset service. The
original filename carries the valid date, forecast mode and bulletin date.
This records when *we* captured the file; it does not infer an earlier public
availability time or claim that a single run has calibrated forecast skill.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import xarray as xr


DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
DATASET_VERSION = "202406"
PRODUCT_ID = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
PRODUCT_URL = f"https://data.marine.copernicus.eu/product/{PRODUCT_ID}/description"
NATIVE_NAME = re.compile(
    r"^glo12_rg_1d-m_(?P<valid>\d{8})-(?P=valid)_2D_fcst_R(?P<run>\d{8})\.nc$"
)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a UTC offset")
    return value.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_forecast(
    path: Path, *, captured_at: datetime | None = None,
    provider_last_modified_at: datetime | None = None,
) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("An original regular NetCDF forecast file is required")
    match = NATIVE_NAME.fullmatch(path.name)
    if not match:
        raise ValueError("Filename must contain a GLO12 daily forecast valid date and bulletin date")
    valid_date = datetime.strptime(match["valid"], "%Y%m%d").date()
    run_date = datetime.strptime(match["run"], "%Y%m%d").date()
    lead_days = (valid_date - run_date).days
    if not 1 <= lead_days <= 10:
        raise ValueError("Forecast valid date must be 1–10 days after the bulletin date")
    captured = _utc(captured_at or datetime.now(timezone.utc))
    if captured < datetime.combine(run_date, time.min, tzinfo=timezone.utc):
        raise ValueError("Capture time precedes the forecast bulletin date")
    captured_before_valid = captured < datetime.combine(valid_date, time.min, tzinfo=timezone.utc)
    modified = _utc(provider_last_modified_at) if provider_last_modified_at else None
    if modified and (modified > captured or modified.date() < run_date):
        raise ValueError("Provider modification time conflicts with bulletin or capture time")

    with xr.open_dataset(path) as ds:
        if "siconc" not in ds or "time" not in ds.coords:
            raise ValueError("GLO12 sea-ice concentration or time coordinate is missing")
        sic = ds["siconc"]
        if sic.dims != ("time", "latitude", "longitude") or ds.sizes["time"] != 1:
            raise ValueError("Expected one daily sea-ice concentration grid")
        if sic.attrs.get("units") != "1":
            raise ValueError("GLO12 sea-ice fraction must use unit 1")
        observed_day = date.fromisoformat(np.datetime_as_string(ds.time.values[0], unit="D"))
        if observed_day != valid_date:
            raise ValueError("NetCDF valid time disagrees with the original filename")
        latitude = np.asarray(ds.latitude.values)
        longitude = np.asarray(ds.longitude.values)
        if (latitude.ndim != 1 or longitude.ndim != 1 or
                not np.all(np.diff(latitude) > 0) or not np.all(np.diff(longitude) > 0)):
            raise ValueError("Unexpected geographic grid coordinates")
        values = np.asarray(sic.values[0], dtype=np.float32)
        finite = np.isfinite(values)
        if np.any(finite & ((values < 0) | (values > 1))):
            raise ValueError("Sea-ice concentration is outside the 0–1 fraction range")
        valid_cells = int(finite.sum())
        if not valid_cells:
            raise ValueError("Forecast contains no valid sea-ice concentration cells")
        southern = latitude < 0
        southern_count = int(np.isfinite(values[southern]).sum())
        if not southern_count:
            raise ValueError("Forecast contains no Southern Hemisphere sea-ice data")

    return {
        "schema_version": 1,
        "scope": "research_forecast_run_archive",
        "product_id": PRODUCT_ID,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "product_url": PRODUCT_URL,
        "source_file": path.name,
        "source_bytes": path.stat().st_size,
        "source_sha256": _sha256(path),
        "bulletin_date": run_date.isoformat(),
        "forecast_base_time_utc": datetime.combine(run_date, time.min, tzinfo=timezone.utc).isoformat(),
        "valid_date": valid_date.isoformat(),
        "lead_days": lead_days,
        "field_kind": "original_forecast_not_reprocessed_analysis",
        "captured_at_utc": captured.isoformat(),
        "provider_last_modified_at_utc": modified.isoformat() if modified else None,
        "available_no_later_than_utc": captured.isoformat(),
        "availability_before_capture_verified": False,
        "captured_before_valid_date": captured_before_valid,
        "eligible_for_prospective_verification": captured_before_valid,
        "grid": {
            "crs": "EPSG:4326",
            "shape": [int(len(latitude)), int(len(longitude))],
            "latitude_range": [float(latitude[0]), float(latitude[-1])],
            "longitude_range": [float(longitude[0]), float(longitude[-1])],
            "valid_sic_cells": valid_cells,
            "valid_southern_sic_cells": southern_count,
        },
        "navigation_clearance": False,
        "uncertainty_calibrated": False,
        "limitations": [
            "The original forecast filename identifies a bulletin date; it does not prove an earlier publication timestamp.",
            "Capture proves possession no later than captured_at_utc, not historical as-of availability.",
            "One forecast run cannot establish predictive skill, calibrated uncertainty, or route safety.",
            "The global grid ends at 80 degrees south and cannot resolve individual icebergs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--provider-last-modified-at", type=datetime.fromisoformat)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory_forecast(args.source, provider_last_modified_at=args.provider_last_modified_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "bulletin_date": result["bulletin_date"],
                      "valid_date": result["valid_date"], "lead_days": result["lead_days"],
                      "uncertainty_calibrated": False}, indent=2))


if __name__ == "__main__":
    main()
