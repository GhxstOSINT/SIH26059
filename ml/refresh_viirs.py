"""Stage, validate, and atomically promote a regional NOAA VIIRS research feed.

This is a data-engineering gate, not an approval for operational navigation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

try:
    from .check_polarwatch_viirs import fetch_latest_timestamp, write_report_atomic
    from .fetch_polarwatch_viirs import DATASETS, query_url
except ImportError:
    from check_polarwatch_viirs import fetch_latest_timestamp, write_report_atomic
    from fetch_polarwatch_viirs import DATASETS, query_url


def utc_timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp requires an explicit timezone")
    return parsed.astimezone(dt.timezone.utc)


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Require an explicit local policy; never infer an operational threshold."""
    if policy.get("schema_version") != 1 or policy.get("scope") != "research_only":
        raise ValueError("A schema-v1 research_only policy is required")
    ids = policy.get("dataset_priority")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(x not in DATASETS for x in ids):
        raise ValueError("dataset_priority must contain distinct allowlisted NOAA VIIRS IDs")
    for key, low, high in (("max_source_age_hours", 1, 720), ("max_local_age_hours", 1, 720),
                           ("min_valid_fraction", 0, 1), ("half_width_km", 10, 250)):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be finite and between {low} and {high}")
    if (not isinstance(policy.get("window_days"), int) or isinstance(policy["window_days"], bool)
            or not 3 <= policy["window_days"] <= 31):
        raise ValueError("window_days must be an integer from 3 to 31")
    center = policy.get("center_lon_lat")
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError("center_lon_lat must contain longitude and latitude")
    query_url(dt.date(2024, 1, 1), dt.date(2024, 1, 3), center[0], center[1], policy["half_width_km"], ids[0])
    return policy


def choose_source(policy: dict[str, Any], *, now: dt.datetime,
                  fetcher: Callable[[str], str] = fetch_latest_timestamp) -> tuple[str, dt.datetime, list[dict[str, Any]]]:
    statuses = []
    selected = None
    for dataset_id in policy["dataset_priority"]:
        try:
            latest = utc_timestamp(fetcher(dataset_id))
            age = (now - latest).total_seconds() / 3600
            status = "eligible" if -1 <= age <= policy["max_source_age_hours"] else "stale_or_future"
            statuses.append({"dataset_id": dataset_id, "latest_timestamp": latest.isoformat(),
                             "age_hours": round(age, 2), "status": status})
            if selected is None and status == "eligible":
                selected = (dataset_id, latest)
        except (OSError, ValueError, TimeoutError, json.JSONDecodeError) as exc:
            statuses.append({"dataset_id": dataset_id, "status": "unavailable", "error": type(exc).__name__})
    if selected is None:
        raise ValueError(f"No allowlisted NOAA VIIRS source met the research freshness policy: {statuses}")
    return selected[0], selected[1], statuses


def inspect_staged(directory: Path, policy: dict[str, Any], dataset_id: str,
                   *, now: dt.datetime) -> dict[str, Any]:
    """Hash-check every declared file and inspect actual SIC/coordinates/timestamps."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("dataset_id") != dataset_id:
        raise ValueError("Staged manifest is incomplete or identifies the wrong source")
    selection = manifest.get("selection", {})
    if (selection.get("center_lon_lat") != policy["center_lon_lat"]
            or selection.get("half_width_km") != policy["half_width_km"]
            or selection.get("crs") != "EPSG:3976"):
        raise ValueError("Staged spatial selection does not match policy")
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Staged manifest has no files")
    seen_files, seen_times, scenes = set(), set(), []
    grid_reference = None
    for item in inventory:
        name, expected = item.get("file"), item.get("sha256")
        if (not isinstance(name, str) or Path(name).name != name or not name.endswith(".nc")
                or name in seen_files or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("Unsafe or duplicate staged inventory record")
        seen_files.add(name)
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Staged file must remain inside its run directory")
        if not path.is_file() or path.stat().st_size != item.get("bytes"):
            raise ValueError(f"Missing or size-mismatched staged file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.lower():
            raise ValueError(f"SHA-256 mismatch in staged file: {name}")
        with xr.open_dataset(path) as ds:
            if not {"IceConc", "time", "rows", "cols"}.issubset(ds.variables):
                raise ValueError("Staged NetCDF lacks required field or coordinates")
            x, y = np.asarray(ds.cols.values), np.asarray(ds.rows.values)
            grid = (x, y)
            if (len(x) < 2 or len(y) < 2 or not np.all(np.diff(x) > 0)
                    or not np.all(np.diff(y) < 0)):
                raise ValueError("Staged coordinates are not a regular oriented grid")
            if grid_reference is not None and (not np.array_equal(x, grid_reference[0])
                                               or not np.array_equal(y, grid_reference[1])):
                raise ValueError("Staged NetCDF chunks use different grids")
            grid_reference = grid
            for index, value in enumerate(ds.time.values):
                timestamp = utc_timestamp(np.datetime_as_string(value, unit="s") + "Z")
                if timestamp in seen_times or timestamp > now + dt.timedelta(hours=1):
                    raise ValueError("Duplicate or future staged observation timestamp")
                seen_times.add(timestamp)
                field = np.asarray(ds.IceConc.isel(time=index).squeeze().values, dtype=np.float32)
                if field.shape != (len(y), len(x)):
                    raise ValueError("Staged SIC dimensions do not match coordinates")
                valid = np.isfinite(field) & (field >= 0) & (field <= 1)
                fraction = float(np.count_nonzero(valid) / field.size)
                scenes.append({"timestamp": timestamp.isoformat(), "valid_fraction": round(fraction, 5)})
    if not scenes:
        raise ValueError("Staged archive has no dated observations")
    latest = max(utc_timestamp(scene["timestamp"]) for scene in scenes)
    age_hours = (now - latest).total_seconds() / 3600
    if age_hours > policy["max_local_age_hours"]:
        raise ValueError(f"Latest local scene is stale ({age_hours:.1f} hours)")
    if any(scene["valid_fraction"] < policy["min_valid_fraction"] for scene in scenes):
        raise ValueError("At least one scene fails the minimum valid-pixel fraction")
    return {"schema_version": 1, "status": "passed", "scope": "research_only",
            "dataset_id": dataset_id, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "latest_timestamp": latest.isoformat(), "latest_age_hours": round(age_hours, 2),
            "scene_count": len(scenes), "scenes": scenes,
            "warning": "Grid completeness is not navigation fitness; no vessel, hazard, or agency approval is implied."}


def active_status(root: Path, policy: dict[str, Any], *, now: dt.datetime) -> dict[str, Any]:
    """Recheck age and integrity; a previous successful promotion can become stale."""
    try:
        pointer = json.loads((root / "active.json").read_text(encoding="utf-8"))
        run_id = pointer["run_id"]
        if not isinstance(run_id, str) or not run_id.isalnum():
            raise ValueError("Invalid active run identifier")
        directory = root / "runs" / run_id
        report = inspect_staged(directory, policy, pointer["dataset_id"], now=now)
        if report["manifest_sha256"] != pointer["manifest_sha256"]:
            raise ValueError("Active manifest changed since promotion")
        return {"available_for_research": True, "operational_decision_ready": False,
                "run_id": run_id, **report}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"available_for_research": False, "operational_decision_ready": False,
                "status": "unavailable", "reason": str(exc)}


def refresh(root: Path, policy: dict[str, Any], *, now: dt.datetime,
            fetcher: Callable[[str], str] = fetch_latest_timestamp,
            downloader: Callable[[Path, str, dt.date, dt.date, dict[str, Any]], None] | None = None) -> dict[str, Any]:
    validate_policy(policy)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_refresh(root):
        return _refresh_locked(root, policy, now=now, fetcher=fetcher, downloader=downloader)


@contextmanager
def exclusive_refresh(root: Path):
    """Prevent concurrent writers; an abandoned lock requires operator inspection."""
    lock = root / ".refresh.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()} at={dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def _refresh_locked(root: Path, policy: dict[str, Any], *, now: dt.datetime,
                    fetcher: Callable[[str], str],
                    downloader: Callable[[Path, str, dt.date, dt.date, dict[str, Any]], None] | None) -> dict[str, Any]:
    dataset_id, upstream_time, sources = choose_source(policy, now=now, fetcher=fetcher)
    run_id = uuid.uuid4().hex
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    end = upstream_time.date()
    start = end - dt.timedelta(days=policy["window_days"] - 1)
    if downloader is None:
        def downloader(target: Path, selected: str, lower: dt.date, upper: dt.date, config: dict[str, Any]) -> None:
            subprocess.run([sys.executable, str(Path(__file__).with_name("fetch_polarwatch_viirs.py")), str(target),
                            "--start", lower.isoformat(), "--end", upper.isoformat(), "--dataset-id", selected,
                            "--lon", str(config["center_lon_lat"][0]), "--lat", str(config["center_lon_lat"][1]),
                            "--half-width-km", str(config["half_width_km"]), "--chunk-days", "7"], check=True)
    try:
        downloader(directory, dataset_id, start, end, policy)
        report = inspect_staged(directory, policy, dataset_id, now=now)
        report.update({"run_id": run_id, "checked_at": now.isoformat(), "source_statuses": sources,
                       "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()})
        write_report_atomic(directory / "quality.json", report)
        write_report_atomic(root / "active.json", {"schema_version": 1, "run_id": run_id,
                                                  "dataset_id": dataset_id,
                                                  "manifest_sha256": report["manifest_sha256"],
                                                  "promoted_at": now.isoformat()})
        return report
    except Exception as exc:
        write_report_atomic(directory / "quality.json", {"schema_version": 1, "status": "rejected",
                                                      "run_id": run_id, "reason": str(exc), "checked_at": now.isoformat()})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("work/active-viirs"))
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    policy = validate_policy(json.loads(args.policy.read_text(encoding="utf-8")))
    now = dt.datetime.now(dt.timezone.utc)
    result = active_status(args.root, policy, now=now) if args.status_only else refresh(args.root, policy, now=now)
    print(json.dumps(result, indent=2))
    return 0 if result.get("available_for_research", result.get("status") == "passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
