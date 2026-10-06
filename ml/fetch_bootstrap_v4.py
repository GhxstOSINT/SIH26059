"""Acquire a bounded, checksummed Southern NSIDC-0079 v4 daily SIC archive."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse

import numpy as np
import xarray as xr


DATASET_ID = "NSIDC-0079"
DOI = "10.5067/X5LG68MH013O"
NAME = re.compile(r"^NSIDC0079_SEAICE_PS_S25km_(\d{8})_v4\.0\.nc$")
ALLOWED_HOST = "data.nsidc.earthdatacloud.nasa.gov"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_daily_granules(granules, start: date, end: date) -> list[tuple[date, str, str]]:
    selected = {}
    for granule in granules:
        name = granule["meta"].get("native-id", "")
        match = NAME.fullmatch(name)
        if not match:
            continue
        day = datetime.strptime(match.group(1), "%Y%m%d").date()
        if not start <= day <= end:
            continue
        links = [link for link in granule.data_links()
                 if urlparse(link).scheme == "https"
                 and urlparse(link).hostname == ALLOWED_HOST
                 and Path(urlparse(link).path).name == name]
        if len(links) != 1 or day in selected:
            raise ValueError(f"Expected one official South daily NetCDF link for {day}.")
        selected[day] = (day, name, links[0])
    expected = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    missing = [day.isoformat() for day in expected if day not in selected]
    if missing:
        raise ValueError(f"CMR has no Southern NSIDC-0079 v4 daily granules for: {missing[:5]}")
    return [selected[day] for day in expected]


def validate_daily_file(path: Path, expected_day: date) -> dict:
    with xr.open_dataset(path, decode_cf=True) as ds:
        variables = [name for name in ds.data_vars if name.endswith("_ICECON")]
        if len(variables) != 1 or not {"crs", "x", "y", "t"}.issubset(ds.variables):
            raise ValueError(f"Unexpected NSIDC-0079 daily variable/coordinate schema: {path.name}")
        if (ds.sizes.get("x"), ds.sizes.get("y"), ds.sizes.get("t")) != (316, 332, 1):
            raise ValueError(f"Unexpected Southern 25 km grid shape in {path.name}")
        observed_day = date.fromisoformat(str(ds.t.values[0])[:10])
        if observed_day != expected_day:
            raise ValueError(f"NetCDF time does not match requested day in {path.name}")
        crs = ds.crs.attrs.get("crs_wkt") or ds.crs.attrs.get("spatial_ref")
        if not crs:
            raise ValueError(f"NetCDF lacks a CRS definition: {path.name}")
        values = np.asarray(ds[variables[0]].isel(t=0).values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        sentinels = np.isclose(finite, 1.1, atol=1e-4) | np.isclose(finite, 1.2, atol=1e-4)
        invalid = finite[(finite < 0) | ((finite > 1) & ~sentinels)]
        valid_count = int(np.count_nonzero((finite >= 0) & (finite <= 1)))
        if invalid.size or not valid_count:
            raise ValueError(f"Decoded concentration is missing or outside [0,1] plus documented sentinels in {path.name}")
        return {"sic_variable": variables[0], "valid_cell_count": valid_count,
                "grid_shape": [332, 316], "time": observed_day.isoformat(),
                "units": "fraction_0_to_1_after_CF_decoding"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--interactive-auth", action="store_true",
                        help="Prompt for one-run Earthdata credentials without persisting them")
    args = parser.parse_args()
    if args.start > args.end or (args.end - args.start).days > 30:
        parser.error("Select 1-31 dates per acquisition, with start on or before end.")
    try:
        import earthaccess
        granules = earthaccess.search_data(short_name=DATASET_ID, version="4",
                                           temporal=(args.start.isoformat(), args.end.isoformat()),
                                           count=200)
        selected = select_daily_granules(granules, args.start, args.end)
        if args.list_only:
            print(json.dumps({"dataset_id": DATASET_ID, "version": "4", "doi": DOI,
                              "files": [name for _, name, _ in selected]}, indent=2))
            return 0
        if args.output_dir.exists():
            raise FileExistsError(f"Output directory already exists; refusing to overwrite: {args.output_dir}")
        auth = earthaccess.login(strategy="interactive" if args.interactive_auth else "netrc",
                                 persist=False)
        session = auth.get_session()
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}.download-",
                                         dir=args.output_dir.parent) as temporary_dir:
            staging = Path(temporary_dir)
            files = []
            for day, name, url in selected:
                target = staging / name
                with session.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with target.open("wb") as output:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                output.write(block)
                details = validate_daily_file(target, day)
                files.append({"file": name, "url": url, "bytes": target.stat().st_size,
                              "sha256": sha256_file(target), **details})
            manifest = {"dataset_id": DATASET_ID, "version": "4", "doi": DOI,
                        "hemisphere": "Southern", "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "requested_period": {"start": args.start.isoformat(), "end": args.end.isoformat()},
                        "files": files, "complete": True,
                        "purpose": "Cross-product SIC sensitivity research; not independent truth or operational input."}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            staging.rename(args.output_dir)
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"),
                          "file_count": len(files), "complete": True}, indent=2))
        return 0
    except Exception as exc:
        print(f"NSIDC-0079 v4 acquisition failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
