"""Fetch a small regional NOAA PolarWatch VIIRS Antarctic SIC NetCDF subset."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import xarray as xr
from pyproj import Transformer

DATASET_ID = "noaacwVIIRSnppiceconcSP06Daily"
DATASETS = {
    "noaacwVIIRSnppiceconcSP06Daily": {
        "title": "Ice Concentration, NOAA S-NPP VIIRS, Near Real-Time, Polar Stereographic (South), Daily",
        "platform": "S-NPP",
    },
    "noaacwVIIRSn21iceconcSP06Daily": {
        "title": "Ice Concentration, NOAA-21 VIIRS, Near Real-Time, Polar Stereographic (South), Daily",
        "platform": "NOAA-21",
    },
    "noaacwVIIRSn20iceconcSP06Daily": {
        "title": "Ice Concentration, NOAA-20 VIIRS, Near Real-Time, Polar Stereographic (South), Daily",
        "platform": "NOAA-20",
    },
}
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3976", always_xy=True)
GRID_SPACING_M = 795.0


def query_url(start: dt.date, end: dt.date, lon: float, lat: float, half_width_km: float,
              dataset_id: str = DATASET_ID) -> tuple[str, dict[str, object]]:
    if end < start:
        raise ValueError("End date must not precede start date")
    if dataset_id not in DATASETS:
        raise ValueError("Dataset ID must be one of the supported NOAA Antarctic VIIRS products")
    if not -180 <= lon <= 180 or not -90 <= lat <= -30 or not 10 <= half_width_km <= 250:
        raise ValueError("Expected an Antarctic center (latitude -90 to -30) and a 10–250 km half-width")
    x_center, y_center = TRANSFORMER.transform(lon, lat)
    half_width_m = half_width_km * 1000
    x_min, x_max = x_center - half_width_m, x_center + half_width_m
    y_max, y_min = y_center + half_width_m, y_center - half_width_m
    query = (f"IceConc[({start.isoformat()}T00:00:00Z):1:({end.isoformat()}T23:59:59Z)]"
             f"[(0.0)][({y_max:.1f}):1:({y_min:.1f})][({x_min:.1f}):1:({x_max:.1f})]")
    base = f"https://polarwatch.noaa.gov/erddap/griddap/{dataset_id}"
    url = f"{base}.nc?{quote(query, safe='')}"
    selection = {"start_date": start.isoformat(), "end_date": end.isoformat(),
                 "center_lon_lat": [lon, lat], "half_width_km": half_width_km,
                 "projected_bounds_m": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
                 "crs": "EPSG:3976", "nominal_grid_spacing_m": GRID_SPACING_M}
    return url, selection


def trim_netcdf(source: str | Path, target: str | Path, lower: dt.date, upper: dt.date,
                include_lower: bool) -> dict[str, object]:
    """Keep actual observations inside this requested chunk and remove ERDDAP edge overlap."""
    with xr.open_dataset(source) as dataset:
        day = dataset.time.dt.floor("D")
        lower_value, upper_value = np.datetime64(lower), np.datetime64(upper)
        mask = (day >= lower_value if include_lower else day > lower_value) & (day <= upper_value)
        indices = np.flatnonzero(np.asarray(mask.values))
        if not len(indices):
            raise IOError(f"ERDDAP returned no observations within requested dates {lower} to {upper}")
        trimmed = dataset.isel(time=indices).load()
        sic_variable = "IceConc" if "IceConc" in trimmed.data_vars else next(iter(trimmed.data_vars))
        trimmed.to_netcdf(target, encoding={sic_variable: {"zlib": True, "complevel": 4}})
        times = np.asarray(trimmed.time.values)
        return {"actual_time_start_utc": np.datetime_as_string(times[0], unit="s") + "Z",
                "actual_time_end_utc": np.datetime_as_string(times[-1], unit="s") + "Z",
                "time_count": len(times)}


def download_to_temp(url: str, temp_name: str) -> tuple[int, str]:
    for attempt in range(5):
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SouthernPassage-research/0.1"})
            with urllib.request.urlopen(request, timeout=180) as response, open(temp_name, "wb") as output:
                if response.status != 200:
                    raise IOError(f"Unexpected ERDDAP HTTP status {response.status}")
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
            return size, digest.hexdigest()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                raise
        delay = min(30, 2 ** attempt)
        print(f"ERDDAP request failed; retrying in {delay}s (attempt {attempt + 2}/5)", flush=True)
        time.sleep(delay)
    raise RuntimeError("unreachable retry state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 6, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2024, 8, 31))
    parser.add_argument("--lon", type=float, default=-60.0, help="ROI center longitude, degrees east")
    parser.add_argument("--lat", type=float, default=-66.0, help="ROI center latitude, degrees north")
    parser.add_argument("--half-width-km", type=float, default=100.0)
    parser.add_argument("--dataset-id", choices=sorted(DATASETS), default=DATASET_ID,
                        help="NOAA Antarctic VIIRS platform product (S-NPP or NOAA-21)")
    parser.add_argument("--chunk-days", type=int, choices=range(1, 32), default=14,
                        help="Maximum consecutive days per NetCDF request (smaller chunks are more reliable)")
    args = parser.parse_args()
    if (args.end - args.start).days < 2:
        parser.error("Choose an inclusive period of at least three consecutive days")
    try:
        _, selection = query_url(args.start, args.end, args.lon, args.lat, args.half_width_km, args.dataset_id)
    except ValueError as exc:
        parser.error(str(exc))
    selection["chunk_days"] = args.chunk_days
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    temp_manifest = args.output_dir / "manifest.json.tmp"
    dataset_info = DATASETS[args.dataset_id]
    manifest = {"dataset_title": dataset_info["title"],
                "dataset_id": args.dataset_id, "platform": dataset_info["platform"],
                "source_info": f"https://polarwatch.noaa.gov/erddap/info/{args.dataset_id}/index.csv",
                "accessed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "selection": selection,
                "complete": False, "files": [],
                "terms_note": "NOAA data service metadata permits free redistribution but disclaims accuracy/warranty; cite NOAA CoastWatch/PolarWatch and verify current terms before operational use.",
                "variable": "IceConc", "units": "fraction [0,1]", "dimensions": ["time", "altitude", "rows", "cols"],
                "product_scope": f"{dataset_info['platform']} VIIRS near-real-time product; different sensor/algorithm and resolution from G02202 CDR. Independent research evaluation only."}
    if manifest_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid existing manifest: {exc}")
        if old_manifest.get("dataset_id") != args.dataset_id or old_manifest.get("selection") != selection:
            parser.error("Output directory already contains a different subset; select another output directory")
    results = []
    chunk_start = args.start
    while chunk_start <= args.end:
        chunk_end = min(args.end, chunk_start + dt.timedelta(days=args.chunk_days - 1))
        url, _ = query_url(chunk_start, chunk_end, args.lon, args.lat, args.half_width_km, args.dataset_id)
        lower = args.start if chunk_start == args.start else chunk_start - dt.timedelta(days=1)
        include_lower = chunk_start == args.start
        safe_roi = f"lon{args.lon:+.2f}_lat{args.lat:+.2f}_w{args.half_width_km:.1f}km".replace("+", "p").replace("-", "m").replace(".", "p")
        platform_tag = {DATASET_ID: "", "noaacwVIIRSn21iceconcSP06Daily": "_noaa21",
                        "noaacwVIIRSn20iceconcSP06Daily": "_noaa20"}[args.dataset_id]
        name = f"polarwatch_viirs_south{platform_tag}_{safe_roi}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.nc"
        target = args.output_dir / name
        if target.exists():
            fd, filtered_name = tempfile.mkstemp(prefix=name + ".", suffix=".filtered.part", dir=args.output_dir)
            os.close(fd)
            try:
                time_metadata = trim_netcdf(target, filtered_name, lower, chunk_end, include_lower)
                os.replace(filtered_name, target)
            finally:
                if os.path.exists(filtered_name):
                    os.unlink(filtered_name)
        else:
            fd, temp_name = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=args.output_dir)
            try:
                os.close(fd)
                size, _ = download_to_temp(url, temp_name)
                with open(temp_name, "rb") as downloaded:
                    header = downloaded.read(8)
                    if (header[:3] != b"CDF" and header != b"\x89HDF\r\n\x1a\n") or size < 1024:
                        raise IOError("ERDDAP response is not a plausible NetCDF file")
                fd, filtered_name = tempfile.mkstemp(prefix=name + ".", suffix=".filtered.part", dir=args.output_dir)
                os.close(fd)
                try:
                    time_metadata = trim_netcdf(temp_name, filtered_name, lower, chunk_end, include_lower)
                    os.replace(filtered_name, target)
                finally:
                    if os.path.exists(filtered_name):
                        os.unlink(filtered_name)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        size = target.stat().st_size
        results.append({"file": name, "start_date": chunk_start.isoformat(), "end_date": chunk_end.isoformat(),
                        "access_url": url, "bytes": size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        **time_metadata})
        manifest.update({"accessed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "complete": False, "files": results})
        temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        print(f"Saved {name}: {size:,} bytes", flush=True)
        chunk_start = chunk_end + dt.timedelta(days=1)
    manifest["complete"] = True
    temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    print(json.dumps({"files": len(results), "manifest": str(manifest_path), "selection": selection}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
