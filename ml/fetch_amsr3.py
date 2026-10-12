"""Acquire immutable operational OSI SAF AMSR3 Antarctic SIC observations.

The files are the provider's original Southern Hemisphere OSI-408-g NetCDFs.
They retain concentration, retrieval uncertainty, screening flags, coordinates,
and projection metadata.  A SHA-256 manifest records the exact local evidence.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xarray as xr


CATALOG_BASE = "https://thredds.met.no/thredds/catalog/osisaf/met.no/ice/amsr2_conc"
FILESERVER_BASE = "https://thredds.met.no/thredds/fileServer/"
PRODUCT_URL = "https://osi-saf.eumetsat.int/products/osi-408-g"
DATASET_ID = "EUMETSAT-OSI-408-g"
ALLOWED_HOST = "thredds.met.no"
CATALOG_NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"
FILE_RE = re.compile(r"^ice_conc_sh_polstere-100_amsr3_(?P<time>\d{12})\.nc$")


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "SouthernPassageResearch/1.0 (OSI-408-g archive)"})
    return session


def monthly_catalog_url(year: int, month: int) -> str:
    return f"{CATALOG_BASE}/{year:04d}/{month:02d}/catalog.xml"


def select_files(catalog_xml: bytes) -> list[dict[str, str]]:
    """Return standard (non test/temporal-uncertainty) SH AMSR3 files."""
    try:
        root = ET.fromstring(catalog_xml)
    except ET.ParseError as exc:
        raise ValueError("MET Norway returned malformed THREDDS catalog XML") from exc
    selected: dict[str, dict[str, str]] = {}
    for dataset in root.iter(f"{CATALOG_NS}dataset"):
        name = dataset.attrib.get("name", "")
        url_path = dataset.attrib.get("urlPath", "")
        match = FILE_RE.fullmatch(name)
        if not match or not url_path or Path(url_path).name != name:
            continue
        parts = url_path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Unsafe THREDDS path for {name}")
        url = urljoin(FILESERVER_BASE, url_path)
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            raise ValueError(f"Refusing untrusted download URL for {name}")
        timestamp = datetime.strptime(match["time"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        selected[name] = {"file": name, "timestamp": timestamp.isoformat(), "url": url}
    return sorted(selected.values(), key=lambda item: item["timestamp"])


def discover_latest(session: requests.Session, now: datetime, days: int) -> list[dict[str, str]]:
    if not 1 <= days <= 14:
        raise ValueError("days must be between 1 and 14")
    months = {(now.year, now.month), ((now - timedelta(days=31)).year, (now - timedelta(days=31)).month)}
    records: dict[str, dict[str, str]] = {}
    for year, month in sorted(months):
        response = session.get(monthly_catalog_url(year, month), timeout=(15, 60))
        response.raise_for_status()
        for item in select_files(response.content):
            records[item["file"]] = item
    if len(records) < days:
        raise ValueError(f"Only {len(records)} operational AMSR3 files were discoverable; need {days}")
    return sorted(records.values(), key=lambda item: item["timestamp"])[-days:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected_timestamp: str | None = None) -> dict:
    with xr.open_dataset(path, decode_cf=True, engine="netcdf4") as ds:
        required = {"ice_conc", "status_flag", "total_uncertainty",
                    "smearing_uncertainty", "algorithm_uncertainty",
                    "Polar_Stereographic_Grid", "lat", "lon", "time"}
        missing = sorted(required - set(ds.variables))
        if missing:
            raise ValueError(f"{path.name} lacks OSI-408-g fields: {', '.join(missing)}")
        if (ds.attrs.get("product_id") != "OSI-408-g" or ds.attrs.get("instrument") != "AMSR3"
                or ds.attrs.get("platform") != "GOSAT-GW"
                or ds.attrs.get("product_status") != "operational"):
            raise ValueError(f"{path.name} is not an operational AMSR3 OSI-408-g product")
        if ds.sizes.get("time") != 1 or ds["ice_conc"].dims != ("time", "yc", "xc"):
            raise ValueError(f"{path.name} has unexpected time or grid dimensions")
        observed = np.datetime_as_string(ds.time.values[0], unit="s") + "Z"
        if expected_timestamp and observed != expected_timestamp.replace("+00:00", "Z"):
            raise ValueError(f"{path.name} time does not match its catalog identity")
        sic = np.asarray(ds.ice_conc.values, dtype=np.float32)
        uncertainty = np.asarray(ds.total_uncertainty.values, dtype=np.float32)
        flags = np.asarray(ds.status_flag.values, dtype=np.float32)
        valid = np.isfinite(sic) & (sic >= 0) & (sic <= 100)
        nominal = valid & np.isfinite(flags) & (flags == 0)
        if not np.any(valid):
            raise ValueError(f"{path.name} contains no valid SIC cells")
        if np.any(np.isfinite(uncertainty) & ((uncertainty < 0) | (uncertainty > 100))):
            raise ValueError(f"{path.name} contains invalid uncertainty values")
        xc = np.asarray(ds.xc.values, dtype=np.float64)
        yc = np.asarray(ds.yc.values, dtype=np.float64)
        if (xc.size < 2 or yc.size < 2 or not np.all(np.diff(xc) > 0)
                or not np.all(np.diff(yc) < 0)):
            raise ValueError(f"{path.name} coordinates are not regularly oriented")
        projection = ds.Polar_Stereographic_Grid.attrs
        if projection.get("grid_mapping_name") != "polar_stereographic":
            raise ValueError(f"{path.name} projection metadata is missing")
        nominal_uncertainty = uncertainty[nominal & np.isfinite(uncertainty)]
        return {
            "timestamp_utc": observed,
            "shape": [int(ds.sizes["yc"]), int(ds.sizes["xc"])],
            "grid_spacing_km": [float(np.median(np.diff(xc))), float(np.median(np.abs(np.diff(yc))))],
            "valid_sic_cells": int(valid.sum()),
            "valid_sic_fraction": float(valid.mean()),
            "nominal_cells": int(nominal.sum()),
            "nominal_fraction": float(nominal.mean()),
            "sic_percent_range": [float(np.nanmin(sic[valid])), float(np.nanmax(sic[valid]))],
            "nominal_total_uncertainty_percent": {
                "median": float(np.nanmedian(nominal_uncertainty)) if nominal_uncertainty.size else None,
                "p90": float(np.nanquantile(nominal_uncertainty, 0.9)) if nominal_uncertainty.size else None,
            },
            "product_version": str(ds.attrs.get("product_version")),
            "software_version": str(ds.attrs.get("software_version")),
            "projection": {key: str(value) for key, value in projection.items()},
        }


def write_json_atomic(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".manifest.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fetch_latest(output_dir: Path, *, days: int = 3, now: datetime | None = None,
                 session: requests.Session | None = None) -> dict:
    if output_dir.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    session = session or make_session()
    checked_at = now or datetime.now(timezone.utc)
    records = discover_latest(session, checked_at, days)
    files = []
    for item in records:
        destination = output_dir / item["file"]
        if destination.is_symlink():
            raise ValueError("Dataset files may not be symlinks")
        if not destination.exists():
            response = session.get(item["url"], stream=True, timeout=(15, 180))
            with response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{destination.name}.",
                                                 suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                try:
                    validate_file(temporary, item["timestamp"])
                    os.replace(temporary, destination)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
        validation = validate_file(destination, item["timestamp"])
        files.append({**item, "bytes": destination.stat().st_size,
                      "sha256": sha256_file(destination), "validation": validation})
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_title": "Operational Southern Hemisphere AMSR3 Sea Ice Concentration (OSI-408-g)",
        "provider": "EUMETSAT OSI SAF / MET Norway",
        "product_url": PRODUCT_URL,
        "source_catalog": CATALOG_BASE,
        "accessed_at_utc": checked_at.isoformat(),
        "complete": True,
        "selection": {"hemisphere": "Southern", "latest_files": days,
                      "grid_spacing_km": 10, "instrument": "AMSR3", "platform": "GOSAT-GW"},
        "files": files,
        "decision_status": "observational_evidence_only",
        "limitations": [
            "Retrieval uncertainty is observation uncertainty, not forecast uncertainty or passage-safety probability.",
            "Ten-kilometre passive-microwave cells cannot resolve individual icebergs or narrow leads.",
            "Status flags must be retained and screened before model training or scoring.",
            "SHA-256 proves local byte integrity, not operational fitness or ministry approval.",
        ],
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("work/datasets/osisaf-amsr3-operational-latest"))
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    result = fetch_latest(args.output_dir, days=args.days)
    latest = result["files"][-1]
    print(json.dumps({"dataset_id": result["dataset_id"], "files": len(result["files"]),
                      "latest_timestamp_utc": latest["timestamp"],
                      "latest_sha256": latest["sha256"],
                      "decision_status": result["decision_status"]}, indent=2))


if __name__ == "__main__":
    main()
