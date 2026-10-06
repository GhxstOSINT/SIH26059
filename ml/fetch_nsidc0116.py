"""Acquire a bounded NSIDC-0116 v4 Southern Hemisphere daily-motion sample.

Requires the optional `earthaccess` package and an Earthdata Login authorized for
NSIDC DAAC downloads. It inventories the authenticated NSIDC HTTPS archive
because this legacy collection currently has no matching granule records in CMR.
Credentials are read by earthaccess from its supported credential stores; they
are never accepted as CLI arguments or written here.
Downloaded-file completeness is distinct from daily vector coverage, because
the product can contain missing vectors over parts of the Southern Hemisphere.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import sys
from urllib.parse import urljoin, urlparse

import numpy as np
import xarray as xr

DATASET_ID = "NSIDC-0116"
DATASET_VERSION = "4"
DOI = "10.5067/INAWUWO7QH7B"
FILENAME_RE = re.compile(
    r"^icemotion_daily_sh_25km_(?P<start>\d{8})_(?P<end>\d{8})_v(?P<version>4(?:\.\d+)?)\.nc$"
)
ALLOWED_HOSTS = {"daacdata.apps.nsidc.org"}
ARCHIVE_ROOT = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/"
    "nsidc0116_icemotion_vectors_v4/south/daily/"
)


class _DirectoryLinks(HTMLParser):
    """Collect ordinary links from the NSIDC HTTPS directory index."""

    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def archive_files(session, start: date, end: date) -> list[tuple[str, str]]:
    """Inventory date-overlapping daily SH files in NSIDC's HTTPS archive."""
    root = urlparse(ARCHIVE_ROOT)
    wanted_years = set(range(start.year, end.year + 1))
    pending = [(ARCHIVE_ROOT, 0)]
    visited = set()
    files: dict[str, str] = {}
    while pending:
        directory, depth = pending.pop()
        if directory in visited:
            continue
        visited.add(directory)
        response = session.get(directory, timeout=60)
        response.raise_for_status()
        final_url = urlparse(response.url)
        if (final_url.scheme != "https" or final_url.hostname not in ALLOWED_HOSTS
                or not final_url.path.startswith(root.path)):
            raise ValueError("NSIDC archive request redirected outside the expected HTTPS directory.")
        links = _DirectoryLinks()
        links.feed(response.text)
        for href in links.hrefs:
            target = urljoin(directory, href)
            parsed = urlparse(target)
            if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
                continue
            if not parsed.path.startswith(root.path) or parsed.query or parsed.fragment:
                continue
            name = Path(parsed.path).name
            match = FILENAME_RE.fullmatch(name)
            if match:
                file_start = date.fromisoformat(
                    f"{match['start'][:4]}-{match['start'][4:6]}-{match['start'][6:]}"
                )
                file_end = date.fromisoformat(
                    f"{match['end'][:4]}-{match['end'][4:6]}-{match['end'][6:]}"
                )
                if file_start <= end and file_end >= start:
                    prior = files.get(name)
                    if prior and prior != target:
                        raise ValueError(f"NSIDC archive returned conflicting URLs for {name}.")
                    files[name] = target
            elif parsed.path.endswith("/") and depth < 2:
                subdir = Path(parsed.path.rstrip("/")).name
                if subdir.isdigit() and len(subdir) == 4 and int(subdir) in wanted_years:
                    pending.append((target, depth + 1))
                elif depth == 0 and not subdir.isdigit():
                    pending.append((target, depth + 1))
    return [(name, files[name]) for name in sorted(files)]


def select_overlapping_files(names_and_urls: list[tuple[str, str]],
                             start: date, end: date) -> list[tuple[str, str]]:
    """Filter archive inventory to daily SH files overlapping the request."""
    selected = {}
    for name, url in names_and_urls:
        match = FILENAME_RE.fullmatch(name)
        if not match:
            continue
        file_start = date.fromisoformat(
            f"{match['start'][:4]}-{match['start'][4:6]}-{match['start'][6:]}"
        )
        file_end = date.fromisoformat(
            f"{match['end'][:4]}-{match['end'][4:6]}-{match['end'][6:]}"
        )
        if file_start <= end and file_end >= start:
            if name in selected and selected[name] != url:
                raise ValueError(f"NSIDC archive returned conflicting URLs for {name}.")
            selected[name] = url
    return [(name, selected[name]) for name in sorted(selected)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_motion_file(path: Path) -> dict:
    """Check the downloaded container and required daily vector fields."""
    with xr.open_dataset(path, decode_cf=True) as dataset:
        required = {"u", "v"}
        if not required.issubset(dataset.data_vars):
            raise ValueError(f"{path.name} lacks required vector fields u/v.")
        if "time" not in dataset.coords or dataset.sizes.get("time", 0) < 1:
            raise ValueError(f"{path.name} has no usable time coordinate.")
        time_values = dataset.time.values
        if not all(str(value)[:10] != "NaT" for value in time_values):
            raise ValueError(f"{path.name} contains an invalid time coordinate.")
        crs = dataset.get("crs")
        axes = {}
        for name in ("x", "y"):
            if name in dataset.coords:
                values = np.asarray(dataset[name].values, dtype=np.float64)
                axes[name] = {"count": int(values.size), "minimum": float(np.nanmin(values)),
                              "maximum": float(np.nanmax(values)),
                              "median_spacing": float(np.nanmedian(np.abs(np.diff(values)))) if values.size > 1 else None}
        return {
            "time_start": str(time_values[0])[:10],
            "time_end": str(time_values[-1])[:10],
            "time_count": int(dataset.sizes["time"]),
            "variables": sorted(name for name in ("u", "v", "icemotion_error_estimate", "icemotion_error_variance") if name in dataset),
            "crs_attributes": {key: str(value) for key, value in crs.attrs.items()} if crs is not None else {},
            "grid_axes": axes,
        }


def existing_complete_manifest(output_dir: Path, start: date, end: date,
                               selected: list[tuple[str, str]]) -> bool:
    """Accept an existing directory only when its frozen acquisition verifies."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Output directory already exists without a verified manifest: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = {name for name, _ in selected}
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    requested = manifest.get("requested_period") if isinstance(manifest, dict) else None
    if (manifest.get("dataset_id") != DATASET_ID or not manifest.get("complete")
            or requested != {"start": start.isoformat(), "end": end.isoformat()}
            or not isinstance(entries, list)
            or {entry.get("file") for entry in entries if isinstance(entry, dict)} != expected_names):
        raise ValueError("Existing output directory is not a complete acquisition of this exact request; preserve it and choose a new output path.")
    for entry in entries:
        name = entry.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Existing manifest contains an unsafe filename.")
        path = output_dir / name
        if (not path.is_file() or path.stat().st_size != entry.get("bytes")
                or sha256_file(path) != entry.get("sha256")):
            raise ValueError(f"Existing acquisition failed integrity verification: {name}")
        validate_motion_file(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--list-only", action="store_true", help="Authenticate, list matching archive files, and print URLs without downloading")
    parser.add_argument("--interactive-auth", action="store_true",
                        help="Prompt for Earthdata credentials for this run without persisting them")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not be after --end")
    if (args.end - args.start).days > 366:
        parser.error("A single request is limited to 367 calendar days; use smaller reviewed windows.")
    try:
        import earthaccess
    except ImportError:
        parser.error("Install the optional dependency with: python -m pip install -r requirements-research.txt")

    try:
        auth = earthaccess.login(
            strategy="interactive" if args.interactive_auth else "netrc",
            persist=False,
        )
        session = auth.get_session()
        available = archive_files(session, args.start, args.end)
        selected = select_overlapping_files(available, args.start, args.end)
        if not selected:
            parser.error("NSIDC's authenticated daily Southern Hemisphere archive contained no files overlapping this period.")
        if args.list_only:
            print(json.dumps({"dataset_id": DATASET_ID, "doi": DOI,
                              "requested_period": [args.start.isoformat(), args.end.isoformat()],
                              "file_count": len(selected), "files": [url for _, url in selected]}, indent=2))
            return 0

        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            existing_complete_manifest(args.output_dir, args.start, args.end, selected)
            print(json.dumps({"complete": True, "already_verified": True,
                              "manifest": str(args.output_dir / "manifest.json")}, indent=2))
            return 0
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        inventory = []
        with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}.download-",
                                         dir=args.output_dir.parent) as temporary_dir:
            staging = Path(temporary_dir)
            for name, url in selected:
                downloaded = staging / name
                with session.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with downloaded.open("wb") as target:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                target.write(chunk)
                if not downloaded.is_file() or downloaded.stat().st_size == 0:
                    raise ValueError(f"Expected download {name} is missing or empty.")
                validation = validate_motion_file(downloaded)
                file_start = date.fromisoformat(validation["time_start"])
                file_end = date.fromisoformat(validation["time_end"])
                if file_end < args.start or file_start > args.end:
                    raise ValueError(f"Downloaded time coordinates for {name} do not overlap the requested period.")
                inventory.append({"file": name, "url": url, "bytes": downloaded.stat().st_size,
                                  "sha256": sha256_file(downloaded), **validation})
            manifest = {
                "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION, "doi": DOI,
                "citation": "Tschudi et al. (2019), Polar Pathfinder Daily 25 km EASE-Grid Sea Ice Motion Vectors, Version 4, NSIDC DAAC, https://doi.org/10.5067/INAWUWO7QH7B",
                "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
                "requested_period": {"start": args.start.isoformat(), "end": args.end.isoformat()},
                "hemisphere": "south", "temporal_resolution": "daily",
                "complete": True,
                "completeness_note": "All matching files listed by the authenticated NSIDC HTTPS archive were downloaded and validated; this does not mean every requested day or every grid cell has a valid motion vector.",
                "files": sorted(inventory, key=lambda item: item["file"]),
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            staging.rename(args.output_dir)
        final_manifest = args.output_dir / "manifest.json"
        print(json.dumps({"complete": True, "file_count": len(inventory),
                          "time_records": sum(item["time_count"] for item in inventory),
                          "manifest": str(final_manifest)}, indent=2))
    except Exception as exc:
        print(f"NSIDC-0116 acquisition failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
