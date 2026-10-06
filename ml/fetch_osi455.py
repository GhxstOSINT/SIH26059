"""Fetch a bounded sample of daily Southern Hemisphere OSI-455 drift files."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import numpy as np
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CATALOG_BASE = "https://thredds.met.no/thredds/catalog/osisaf/met.no/reprocessed/ice/drift_455m_files/merged"
FILESERVER_BASE = "https://thredds.met.no/thredds/fileServer/"
ALLOWED_HOST = "thredds.met.no"
DATASET_ID = "EUMETSAT-OSI-455"
DOI = "10.15770/EUM_SAF_OSI_0012"
CATALOG_NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"
FILE_RE = re.compile(r"^ice_drift_sh_ease2-750_cdr-v1p0_24h-(?P<time>\d{12})\.nc$")
MAX_DAYS_DEFAULT = 31


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "SouthernPassageResearch/1.0 (OSI-455 reproducibility sample)"})
    return session


def monthly_catalog_url(year: int, month: int) -> str:
    return f"{CATALOG_BASE}/{year:04d}/{month:02d}/catalog.xml"


def select_files(catalog_xml: bytes, start: date, end: date) -> list[dict[str, str]]:
    """Select only the exact daily SH merged OSI-455 files within a date range."""
    try:
        root = ET.fromstring(catalog_xml)
    except ET.ParseError as exc:
        raise ValueError("MET Norway returned malformed THREDDS catalog XML") from exc
    selected: dict[date, dict[str, str]] = {}
    for dataset in root.iter(f"{CATALOG_NS}dataset"):
        name = dataset.attrib.get("name", "")
        url_path = dataset.attrib.get("urlPath", "")
        match = FILE_RE.fullmatch(name)
        if not match or not url_path or Path(url_path).name != name:
            continue
        valid_at = datetime.strptime(match["time"], "%Y%m%d%H%M").date()
        if valid_at < start or valid_at > end:
            continue
        path_parts = url_path.split("/")
        if (path_parts[-1] != name or not path_parts[-2:][0].isdigit()
                or any(part in {"", ".", ".."} for part in path_parts)):
            raise ValueError(f"Unsafe or unexpected THREDDS path for {name}")
        download_url = urljoin(FILESERVER_BASE, url_path)
        parsed = urlparse(download_url)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            raise ValueError(f"Refusing non-HTTPS or non-MET Norway download URL for {name}")
        record = {"date": valid_at.isoformat(), "file": name, "url_path": url_path, "url": download_url}
        if valid_at in selected and selected[valid_at]["url_path"] != url_path:
            raise ValueError(f"Conflicting OSI-455 merged files for {valid_at}")
        selected[valid_at] = record
    return [selected[day] for day in sorted(selected)]


def requested_dates(start: date, end: date, max_days: int = MAX_DAYS_DEFAULT) -> list[date]:
    if end < start:
        raise ValueError("end date must be on or after start date")
    count = (end - start).days + 1
    if count > max_days:
        raise ValueError(f"Requested {count} days exceeds the bounded limit of {max_days}; split the period or raise --max-days deliberately")
    return [date.fromordinal(start.toordinal() + offset) for offset in range(count)]


def discover_files(session: requests.Session, start: date, end: date,
                   max_days: int = MAX_DAYS_DEFAULT, catalog_workers: int = 8) -> list[dict[str, str]]:
    if not 1 <= catalog_workers <= 16:
        raise ValueError("catalog_workers must be between 1 and 16")
    days = requested_dates(start, end, max_days)
    months = sorted({(day.year, day.month) for day in days})
    records: dict[date, dict[str, str]] = {}
    def read_month(month: tuple[int, int]) -> tuple[int, int, bytes]:
        year, month_number = month
        response = session.get(monthly_catalog_url(year, month_number), timeout=(15, 60))
        response.raise_for_status()
        return year, month_number, response.content

    with ThreadPoolExecutor(max_workers=min(catalog_workers, len(months)),
                            thread_name_prefix="osi455-catalog") as executor:
        catalog_results = list(executor.map(read_month, months))
    for year, month, content in catalog_results:
        for item in select_files(content, start, end):
            day = date.fromisoformat(item["date"])
            if (day.year, day.month) == (year, month):
                records[day] = item
    missing = [day.isoformat() for day in days if day not in records]
    if missing:
        raise ValueError("OSI-455 merged SH catalog has no daily file(s) for: " + ", ".join(missing))
    return [records[day] for day in days]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected_date: str | None = None) -> dict:
    """Validate date, vectors, status flags and projection metadata; retain raw flags."""
    with xr.open_dataset(path, decode_cf=True, engine="netcdf4") as ds:
        required = {"dX", "dY", "status_flag", "uncert_dX_and_dY", "time_bnds",
                    "Lambert_Azimuthal_Equal_Area"}
        missing = sorted(required - set(ds.variables))
        if missing:
            raise ValueError(f"{path.name} is missing OSI-455 fields: {', '.join(missing)}")
        if ds.sizes.get("time") != 1 or not {"xc", "yc"}.issubset(ds.coords):
            raise ValueError(f"{path.name} must contain one dated field on xc/yc coordinates")
        observed_date = np.datetime_as_string(ds.time.values[0], unit="D")
        if expected_date is not None and observed_date != expected_date:
            raise ValueError(f"{path.name} time {observed_date} does not match catalog date {expected_date}")
        bounds = np.asarray(ds["time_bnds"].values).reshape(-1)
        if bounds.size != 2 or np.isnat(bounds).any() or (bounds[1] - bounds[0]) != np.timedelta64(24, "h"):
            raise ValueError(f"{path.name} does not describe a valid 24-hour displacement interval")
        for variable in ("dX", "dY", "status_flag", "uncert_dX_and_dY"):
            if ds[variable].dims != ("time", "yc", "xc"):
                raise ValueError(f"{path.name} has unexpected dimensions for {variable}: {ds[variable].dims}")
        if ds["dX"].attrs.get("units") != "km" or ds["dY"].attrs.get("units") != "km":
            raise ValueError(f"{path.name} displacement units are not documented as kilometres")
        flags = np.asarray(ds["status_flag"].values, dtype=np.int16)
        dx = np.asarray(ds["dX"].values, dtype=np.float64)
        dy = np.asarray(ds["dY"].values, dtype=np.float64)
        uncertainty = np.asarray(ds["uncert_dX_and_dY"].values, dtype=np.float64)
        # The archived variable is named `uncert_dX_and_dY`; keep its exact
        # source spelling above and report only descriptive summaries here.
        values, counts = np.unique(flags, return_counts=True)
        accepted = np.isin(flags, (20, 21, 22, 23, 24, 25, 30)) & np.isfinite(dx) & np.isfinite(dy)
        magnitude = np.hypot(dx[accepted], dy[accepted])
        finite_uncertainty = uncertainty[accepted & np.isfinite(uncertainty)]
        xc = np.asarray(ds.xc.values, dtype=np.float64)
        yc = np.asarray(ds.yc.values, dtype=np.float64)
        projection = ds["Lambert_Azimuthal_Equal_Area"].attrs
        flag_var = ds["status_flag"]
        return {
            "time": observed_date,
            "time_start_utc": np.datetime_as_string(bounds[0], unit="s") + "Z",
            "time_end_utc": np.datetime_as_string(bounds[1], unit="s") + "Z",
            "dimensions": {name: int(size) for name, size in ds.sizes.items()},
            "grid_spacing_km": {
                "xc": float(np.nanmedian(np.abs(np.diff(xc)))) if xc.size > 1 else None,
                "yc": float(np.nanmedian(np.abs(np.diff(yc)))) if yc.size > 1 else None,
            },
            "projection": {key: str(value) for key, value in projection.items()},
            "variables": {name: {"units": ds[name].attrs.get("units"), "standard_name": ds[name].attrs.get("standard_name")}
                          for name in ("dX", "dY", "uncert_dX_and_dY")},
            "status_flag_meanings": flag_var.attrs.get("flag_meanings", ""),
            "status_flag_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
            "cells_with_success_flags_20_to_30": int(accepted.sum()),
            "grid_cell_count": int(flags.size),
            "success_flag_cell_fraction": float(accepted.sum() / flags.size),
            "displacement_magnitude_km_quantiles": {
                label: float(value) for label, value in zip(("p50", "p90", "p95", "p99"),
                    np.nanquantile(magnitude, (0.5, 0.9, 0.95, 0.99)))
            } if magnitude.size else {},
            "uncertainty_km_median": float(np.nanmedian(finite_uncertainty)) if finite_uncertainty.size else None,
        }


def existing_manifest_matches(output_dir: Path, start: date, end: date,
                               records: list[dict[str, str]]) -> bool:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("dataset_id") != DATASET_ID or manifest.get("complete") is not True
                or manifest.get("requested_period") != {"start": start.isoformat(), "end": end.isoformat()}
                or [item.get("file") for item in manifest.get("files", [])] != [r["file"] for r in records]):
            return False
        for item in manifest["files"]:
            local = output_dir / item["file"]
            if (not local.is_file() or local.stat().st_size != item["bytes"]
                    or sha256_file(local) != item["sha256"]):
                raise ValueError(f"Existing OSI-455 file failed integrity verification: {local.name}")
            validate_file(local, item["time"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Existing OSI-455 manifest is invalid") from exc
    return True


def cached_file_record(output_dir: Path, item: dict[str, str],
                       prior_records: dict[str, dict]) -> dict | None:
    """Reuse a previously manifested immutable file without redownloading it."""
    record = prior_records.get(item["file"])
    local = output_dir / item["file"]
    if record is None or record.get("url") != item["url"] or record.get("time") != item["date"]:
        return None
    if not local.is_file():
        return None
    if (local.stat().st_size != record.get("bytes")
            or sha256_file(local) != record.get("sha256")):
        raise ValueError(f"Previously manifested OSI-455 file failed integrity verification: {local.name}")
    return record


def write_manifest_atomic(output_dir: Path, manifest: dict) -> None:
    """Persist a resumable checkpoint without exposing a partial JSON file."""
    manifest_path = output_dir / "manifest.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir,
                                     prefix=".manifest.", suffix=".tmp", delete=False) as temp:
        manifest_temp = Path(temp.name)
        try:
            json.dump(manifest, temp, indent=2, sort_keys=True)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        except BaseException:
            manifest_temp.unlink(missing_ok=True)
            raise
    for attempt in range(8):
        try:
            os.replace(manifest_temp, manifest_path)
            return
        except PermissionError:
            if attempt == 7:
                manifest_temp.unlink(missing_ok=True)
                raise
            # Antivirus/indexing and concurrent readers can briefly hold a
            # Windows handle without delete sharing; retry the atomic replace.
            time.sleep(min(0.1 * (2 ** attempt), 1.0))


def fetch(session: requests.Session, output_dir: Path, start: date, end: date,
          max_days: int = MAX_DAYS_DEFAULT, workers: int = 6) -> dict:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    records = discover_files(session, start, end, max_days)
    output_dir.mkdir(parents=True, exist_ok=True)
    if existing_manifest_matches(output_dir, start, end, records):
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    prior_records: dict[str, dict] = {}
    prior_manifest_path = output_dir / "manifest.json"
    if prior_manifest_path.is_file():
        try:
            prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Existing OSI-455 manifest is invalid; refusing incremental reuse") from exc
        if prior_manifest.get("dataset_id") != DATASET_ID or not isinstance(prior_manifest.get("files"), list):
            raise ValueError("Existing manifest is not a reusable OSI-455 file manifest")
        prior_records = {entry["file"]: entry for entry in prior_manifest["files"]
                         if isinstance(entry, dict) and isinstance(entry.get("file"), str)}

    thread_state = threading.local()

    def acquire(item: dict[str, str]) -> dict:
        destination = output_dir / item["file"]
        cached = cached_file_record(output_dir, item, prior_records)
        if cached is not None:
            metadata = cached.get("validation")
            if not isinstance(metadata, dict):
                metadata = validate_file(destination, item["date"])
            return {**cached, "validation": metadata}
        if not hasattr(thread_state, "session"):
            thread_state.session = make_session()
        response = thread_state.session.get(item["url"], stream=True, timeout=(15, 120))
        with response:
            response.raise_for_status()
            declared_size = response.headers.get("Content-Length")
            digest = hashlib.sha256()
            byte_count = 0
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{destination.name}.", suffix=".part", delete=False) as temp:
                temp_path = Path(temp.name)
                try:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        temp.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                    temp.flush()
                    os.fsync(temp.fileno())
                except BaseException:
                    temp_path.unlink(missing_ok=True)
                    raise
        if declared_size is not None and byte_count != int(declared_size):
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"Incomplete transfer for {destination.name}: expected {declared_size}, got {byte_count}")
        try:
            metadata = validate_file(temp_path, item["date"])
            if destination.exists():
                # Never overwrite an unmanifested local file; require byte identity
                # with the current remote source before accepting it.
                if sha256_file(destination) != digest.hexdigest():
                    raise ValueError(f"Refusing to overwrite a non-matching local file: {destination.name}")
                temp_path.unlink(missing_ok=True)
            else:
                os.replace(temp_path, destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return {"file": destination.name, "time": item["date"], "url": item["url"],
                "bytes": byte_count, "sha256": digest.hexdigest(), "validation": metadata}

    file_records: dict[str, dict] = {}
    checkpoint_lock = threading.Lock()

    def persist_checkpoint() -> None:
        with checkpoint_lock:
            partial_manifest = build_manifest(list(file_records.values()), complete=False)
            write_manifest_atomic(output_dir, partial_manifest)

    def build_manifest(items: list[dict], complete: bool) -> dict:
        return {
            "dataset_id": DATASET_ID,
            "dataset_title": "EUMETSAT OSI SAF Global Low Resolution Sea Ice Drift CDR v1 (OSI-455)",
            "doi": DOI,
            "source_catalog": CATALOG_BASE,
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requested_period": {"start": start.isoformat(), "end": end.isoformat()},
            "complete": complete,
            "selection": {"hemisphere": "Southern", "product": "merged multi-sensor CDR", "displacement_interval_hours": 24},
            "files": sorted(items, key=lambda entry: entry["file"]),
            "limitations": [
                "Sea-ice motion is not iceberg motion.",
                "The source CDR has 75 km cells; selected status flags include interpolated and wind-filled values, which must be stratified in evaluation.",
                "Local SHA-256 establishes acquisition integrity, not provider-published authenticity or fitness for navigation.",
            ],
        }

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="osi455") as executor:
        futures = {executor.submit(acquire, item): item for item in records}
        try:
            for future in as_completed(futures):
                item = futures[future]
                file_records[item["file"]] = future.result()
                if len(file_records) % 10 == 0:
                    persist_checkpoint()
        except BaseException:
            for future, item in futures.items():
                if future.done() and not future.cancelled():
                    try:
                        file_records[item["file"]] = future.result()
                    except BaseException:
                        pass
            if file_records:
                persist_checkpoint()
            for future in futures:
                future.cancel()
            raise
    manifest = build_manifest(list(file_records.values()),
                              complete=len(file_records) == (end - start).days + 1)
    write_manifest_atomic(output_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--max-days", type=int, default=MAX_DAYS_DEFAULT,
                        help="Maximum inclusive number of dates (default: 31)")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=6,
                        help="Parallel downloads (1-16; default: 6)")
    parser.add_argument("--list-only", action="store_true", help="List selected archive files without downloading")
    args = parser.parse_args()
    session = make_session()
    records = discover_files(session, args.start, args.end, args.max_days)
    if args.list_only:
        print(json.dumps(records, indent=2))
        return
    print(json.dumps(fetch(session, args.output_dir, args.start, args.end, args.max_days, args.workers), indent=2))


if __name__ == "__main__":
    main()
