"""Fetch annual Antarctic NOAA/NSIDC G02202 v6 NetCDF aggregates with hashes.

Example: python ml/fetch_g02202.py work/datasets/g02202 --start 2007 --end 2016
This downloads about 0.7 GB for that decade. Keep the downloaded data out of Git.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/aggregate"
CHUNK = 2 * 1024 * 1024


def fetch(year: int, output: Path) -> dict[str, object]:
    name = f"sic_pss25_{year}0101-{year}1231_v06r00.nc"
    url = f"{BASE}/{name}"
    target = output / name
    head = urllib.request.Request(url, headers={"User-Agent": "SouthernPassage-research/0.1"}, method="HEAD")
    with urllib.request.urlopen(head, timeout=30) as response:
        expected = int(response.headers["Content-Length"])
    if target.exists():
        if target.stat().st_size != expected:
            raise FileExistsError(f"Existing file has unexpected size; refusing to replace: {target}")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
        return {"year": year, "file": name, "url": url, "bytes": expected,
                "sha256": digest.hexdigest(), "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "verified_existing_file": True}
    part = output / f"{name}.part"
    if not part.exists():
        leftovers = sorted(output.glob(f"{name}.*.part"), key=lambda p: p.stat().st_size, reverse=True)
        usable = next((p for p in leftovers if p.stat().st_size > 0), None)
        if usable:
            os.replace(usable, part)
    total = part.stat().st_size if part.exists() else 0
    if total > expected:
        raise IOError(f"{name}: partial file is larger than source ({total} > {expected})")
    digest = hashlib.sha256()
    if total:
        with part.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
    with part.open("ab") as handle:
        while total < expected:
            start = total
            end = min(start + CHUNK, expected) - 1
            last_error = None
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "SouthernPassage-research/0.1",
                                                                   "Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=180) as response:
                        received = response.read()
                        content_range = response.headers.get("Content-Range", "")
                    if len(received) != end - start + 1 or content_range != f"bytes {start}-{end}/{expected}":
                        raise IOError(f"{name}: unexpected range response {content_range!r}, {len(received)} bytes")
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            if last_error and not received:
                raise last_error
            handle.write(received)
            handle.flush()
            digest.update(received)
            total += len(received)
            print(f"{year}: {total:,}/{expected:,} bytes", flush=True)
    if total != expected:
        raise IOError(f"{name}: expected {expected} bytes, received {total}")
    os.replace(part, target)
    return {"year": year, "file": name, "url": url, "bytes": total,
            "sha256": digest.hexdigest(), "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=int, default=2007)
    parser.add_argument("--end", type=int, default=2016)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.start < 1979 or args.end > 2025 or args.end < args.start:
        parser.error("Choose an available Antarctic year range from 1979 through 2025")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    manifest_path = args.output_dir / "manifest.json"

    def write_manifest(complete: bool) -> None:
        results.sort(key=lambda row: int(row["year"]))
        manifest = {"dataset": "NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 6",
                    "dataset_id": "G02202", "doi": "10.7265/b18j-z797", "hemisphere": "Southern",
                    "source": BASE, "accessed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "requested_years": [args.start, args.end], "complete": complete,
                    "files": list(results), "terms": "Retain provider citation; review current NSIDC terms before redistribution."}
        temp_manifest = args.output_dir / "manifest.json.tmp"
        temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, year, args.output_dir): year for year in range(args.start, args.end + 1)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            write_manifest(complete=False)
            print(f"Downloaded {result['file']} ({result['bytes']:,} bytes)", flush=True)
    results.sort(key=lambda row: int(row["year"]))
    write_manifest(complete=True)
    print(f"Manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
