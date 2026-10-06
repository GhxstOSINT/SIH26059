"""Check NOAA PolarWatch ERDDAP metadata without downloading raster data."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

try:
    from .fetch_polarwatch_viirs import DATASETS
except ImportError:  # Support direct execution as `python ml/check_polarwatch_viirs.py`.
    from fetch_polarwatch_viirs import DATASETS

MAX_METADATA_BYTES = 2 * 1024 * 1024
USER_AGENT = "SouthernPassage-source-monitor/0.1"


def parse_coverage_end(payload: dict[str, Any]) -> str:
    """Extract and validate the dataset's declared latest observation timestamp."""
    rows = payload.get("table", {}).get("rows", [])
    for row in rows:
        if (isinstance(row, list) and len(row) >= 5 and row[0] == "attribute"
                and row[1] == "NC_GLOBAL" and row[2] == "time_coverage_end"):
            value = row[4]
            if not isinstance(value, str):
                break
            try:
                parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("NOAA time_coverage_end is not a valid ISO-8601 timestamp") from exc
            if parsed.tzinfo is None:
                raise ValueError("NOAA time_coverage_end must include a timezone")
            return parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    raise ValueError("NOAA metadata did not contain a valid global time_coverage_end")


def fetch_latest_timestamp(dataset_id: str, *, timeout: float = 20.0,
                           opener: Callable[..., Any] | None = None) -> str:
    if dataset_id not in DATASETS:
        raise ValueError("Dataset ID is not an allowlisted Antarctic VIIRS product")
    url = f"https://polarwatch.noaa.gov/erddap/info/{dataset_id}/index.json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    open_url = opener or urllib.request.urlopen
    with open_url(request, timeout=timeout) as response:
        body = response.read(MAX_METADATA_BYTES + 1)
    if len(body) > MAX_METADATA_BYTES:
        raise ValueError("NOAA metadata response exceeded the 2 MiB safety limit")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("NOAA metadata response is not a JSON object")
    return parse_coverage_end(payload)


def check_sources(*, now: dt.datetime | None = None, recent_hours: float = 48,
                  delayed_hours: float = 168,
                  fetcher: Callable[[str], str] = fetch_latest_timestamp) -> dict[str, Any]:
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if recent_hours <= 0 or delayed_hours <= recent_hours:
        raise ValueError("Freshness thresholds must satisfy 0 < recent_hours < delayed_hours")
    products = []
    for dataset_id, info in DATASETS.items():
        try:
            latest = fetcher(dataset_id)
            observed_at = dt.datetime.fromisoformat(latest.replace("Z", "+00:00"))
            if observed_at.tzinfo is None:
                raise ValueError("Source timestamp must include a timezone")
            age_hours = (current - observed_at.astimezone(dt.timezone.utc)).total_seconds() / 3600
            if age_hours < -6:
                status = "future_timestamp"
            elif age_hours <= recent_hours:
                status = "recent"
            elif age_hours <= delayed_hours:
                status = "delayed"
            else:
                status = "stale"
            products.append({"dataset_id": dataset_id, "platform": info["platform"],
                             "status": status, "latest_source_timestamp": latest,
                             "age_hours": round(max(0.0, age_hours), 1)})
        except Exception as exc:  # Per-feed failure must not hide the other upstream statuses.
            products.append({"dataset_id": dataset_id, "platform": info["platform"],
                             "status": "unavailable", "latest_source_timestamp": None,
                             "age_hours": None, "error": type(exc).__name__})
    statuses = [product["status"] for product in products]
    if all(status == "unavailable" for status in statuses):
        status = "unavailable"
    elif any(status in {"unavailable", "future_timestamp"} for status in statuses):
        status = "partial_or_invalid"
    elif all(value == "recent" for value in statuses):
        status = "recent"
    elif any(value == "stale" for value in statuses):
        status = "stale"
    else:
        status = "delayed"
    return {"schema_version": 1, "checked_at": current.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": status, "classification_basis": "demonstration_thresholds_not_approved_operational_limits",
            "thresholds_hours": {"recent_max": recent_hours, "delayed_max": delayed_hours},
            "products": products,
            "warning": "Upstream catalog metadata only. This check does not verify source-file quality, local archive integrity, spatial coverage, or operational fitness."}


def write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("work/monitoring/polarwatch-source-status.json"))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--recent-hours", type=float, default=48.0,
                        help="Demonstration threshold only; obtain operational approval before use")
    parser.add_argument("--delayed-hours", type=float, default=168.0,
                        help="Demonstration threshold only; obtain operational approval before use")
    args = parser.parse_args()
    report = check_sources(recent_hours=args.recent_hours, delayed_hours=args.delayed_hours,
                           fetcher=lambda dataset_id: fetch_latest_timestamp(dataset_id, timeout=args.timeout))
    write_report_atomic(args.output, report)
    print(json.dumps({"status": report["status"], "checked_at": report["checked_at"],
                      "output": str(args.output), "products": report["products"]}, indent=2))
    return 0 if report["status"] in {"recent", "delayed", "stale"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
