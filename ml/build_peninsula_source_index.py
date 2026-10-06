"""Build a checksum-verified, forecast-origin audit of Peninsula VIIRS scenes.

The index records observation time and grid coverage, but deliberately does not
claim that a historical scene was *published* by any hypothetical forecast
origin. A separate availability record is required before causal training.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


ALLOWED_IDS = {
    "noaacwVIIRSnppiceconcSP06Daily": "S-NPP",
    "noaacwVIIRSn21iceconcSP06Daily": "NOAA-21",
    "noaacwVIIRSn20iceconcSP06Daily": "NOAA-20",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_source(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    dataset_id = manifest.get("dataset_id")
    selection = manifest.get("selection") or {}
    if dataset_id not in ALLOWED_IDS or manifest.get("complete") is not True:
        raise ValueError("Only complete, allowlisted South VIIRS archives are supported")
    if selection.get("crs") != "EPSG:3976" or not manifest.get("files"):
        raise ValueError("Missing supported Antarctic grid or file inventory")
    start = date.fromisoformat(selection["start_date"])
    end = date.fromisoformat(selection["end_date"])
    if end < start:
        raise ValueError("Invalid source period")
    scenes = {}
    grid = None
    grid_sha256 = None
    for item in manifest["files"]:
        name = item.get("file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".nc"):
            raise ValueError("Unsafe source file name")
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Source file escapes archive")
        if not path.is_file() or path.stat().st_size != item.get("bytes") or digest(path) != item.get("sha256"):
            raise ValueError(f"Missing or altered source file: {name}")
        with xr.open_dataset(path) as ds:
            if "IceConc" not in ds or not all(key in ds for key in ("time", "rows", "cols")):
                raise ValueError("Missing required South VIIRS variables")
            x = np.asarray(ds.cols.values)
            y = np.asarray(ds.rows.values)
            if x.ndim != 1 or y.ndim != 1 or len(x) == 0 or len(y) == 0:
                raise ValueError("Invalid grid coordinates")
            this_grid = (x, y)
            if grid is not None and (not np.array_equal(grid[0], x) or not np.array_equal(grid[1], y)):
                raise ValueError("Chunks do not share a grid")
            grid = this_grid
            coordinates_digest = hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()
                                                + np.asarray(y, dtype="<f8").tobytes()).hexdigest()
            grid_sha256 = coordinates_digest
            for offset, timestamp in enumerate(ds.time.values):
                observed_at = np.datetime_as_string(timestamp, unit="s") + "Z"
                day = date.fromisoformat(observed_at[:10])
                if not start <= day <= end or day in scenes:
                    raise ValueError("Duplicate or out-of-period observation date")
                field = np.asarray(ds.IceConc.isel(time=offset).squeeze().values)
                if field.shape != (len(y), len(x)):
                    raise ValueError("Ice concentration/grid shape mismatch")
                valid = np.isfinite(field) & (field >= 0) & (field <= 1)
                scenes[day.isoformat()] = {
                    "observed_at_utc": observed_at,
                    "valid_grid_fraction": float(valid.mean()),
                    "source_file": name,
                    "source_sha256": item["sha256"],
                    "published_at_utc": None,
                    "asof_verified": False,
                }
    return {
        "dataset_id": dataset_id,
        "platform": ALLOWED_IDS[dataset_id],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_accessed_at_utc": manifest.get("accessed_at_utc"),
        "selection": selection,
        "grid_shape": [len(grid[1]), len(grid[0])],
        "grid_coordinate_sha256": grid_sha256,
        "scenes": scenes,
    }


def build_index(directories: list[Path]) -> dict:
    if not directories:
        raise ValueError("At least one source is required")
    sources = [read_source(directory) for directory in directories]
    if len({source["dataset_id"] for source in sources}) != len(sources):
        raise ValueError("Duplicate platform")
    first = sources[0]
    period = (first["selection"]["start_date"], first["selection"]["end_date"])
    region = (first["selection"].get("center_lon_lat"), first["selection"].get("half_width_km"),
              first["selection"].get("projected_bounds_m"), first["grid_shape"],
              first["grid_coordinate_sha256"])
    for source in sources[1:]:
        subset = source["selection"]
        if (subset["start_date"], subset["end_date"]) != period or (
            subset.get("center_lon_lat"), subset.get("half_width_km"),
            subset.get("projected_bounds_m"), source["grid_shape"],
            source["grid_coordinate_sha256"]
        ) != region:
            raise ValueError("Sources must cover the same period and grid")
    start, end = (date.fromisoformat(value) for value in period)
    days = []
    for delta in range((end - start).days + 1):
        day = start + timedelta(days=delta)
        day_text = day.isoformat()
        origin = datetime.combine(day + timedelta(days=1), time.min, tzinfo=timezone.utc)
        row = {"observation_date": day_text, "forecast_origin_utc": origin.isoformat().replace("+00:00", "Z"),
               "platforms": {}}
        for source in sources:
            scene = source["scenes"].get(day_text)
            if scene is None:
                row["platforms"][source["platform"]] = {"status": "missing_observation"}
            else:
                if datetime.fromisoformat(scene["observed_at_utc"].replace("Z", "+00:00")) >= origin:
                    raise ValueError("Observation is not earlier than forecast origin")
                row["platforms"][source["platform"]] = {"status": "observed_availability_unverified", **scene}
        days.append(row)
    return {
        "schema_version": 1,
        "scope": "research_source_inventory",
        "period": {"start_date": period[0], "end_date": period[1]},
        "region": {"center_lon_lat": region[0], "half_width_km": region[1],
                   "crs": "EPSG:3976", "grid_shape": region[3],
                   "grid_coordinate_sha256": region[4]},
        "sources": [{key: value for key, value in source.items() if key != "scenes"} for source in sources],
        "summary": {
            "scheduled_days": len(days),
            "observed_days_by_platform": {source["platform"]: len(source["scenes"]) for source in sources},
            "days_observed_by_all_platforms": sum(all(source["scenes"].get(row["observation_date"])
                                                   for source in sources) for row in days),
            "forecast_origins_with_verified_availability": 0,
        },
        "days": days,
        "next_gate": "Obtain source publication/availability timestamps and source-specific QA policy before causal training; archive access dates are not historical publication times.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_index(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
