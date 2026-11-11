"""Export checksum-verified GLO12 forecast files to a small regional Parquet package.

The package is a derived Peninsula-Weddell subset, not a replacement for original
NetCDF files used by prospective verification.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from ml.archive_glo12_forecast import (
    DATASET_ID,
    DATASET_VERSION,
    PRODUCT_ID,
    PRODUCT_URL,
    inventory_forecast,
)

REQUIRED_LEADS = (1, 3, 5, 7)
NATIVE_NAME = re.compile(
    r"^glo12_rg_1d-m_(?P<valid>\d{8})-(?P=valid)_2D_fcst_R(?P<run>\d{8})\.nc$"
)
ATTRIBUTION = "E.U. Copernicus Marine Service Information"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def date_from_filename(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def find_run(input_dir: Path, requested: date | None) -> tuple[date, list[Path]]:
    grouped: dict[date, dict[int, Path]] = {}
    for source in input_dir.glob("*.nc"):
        match = NATIVE_NAME.fullmatch(source.name)
        if not match:
            continue
        run_day = date_from_filename(match["run"])
        valid_day = date_from_filename(match["valid"])
        lead = (valid_day - run_day).days
        if lead in REQUIRED_LEADS:
            grouped.setdefault(run_day, {})[lead] = source

    candidates = [requested] if requested else sorted(grouped, reverse=True)
    for run_day in candidates:
        if run_day not in grouped:
            continue
        selected = grouped[run_day]
        missing = [lead for lead in REQUIRED_LEADS if lead not in selected]
        if not missing:
            return run_day, [selected[lead] for lead in REQUIRED_LEADS]
        if requested:
            raise ValueError(f"Bulletin {run_day} is missing required leads: {missing}")
    raise ValueError("No complete 1/3/5/7-day GLO12 bulletin was found")


def check_source(source: Path) -> dict:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Source must be a regular NetCDF file: {source}")
    manifest_path = source.with_suffix(".manifest.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"Checksum manifest is missing: {manifest_path}")
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        captured = datetime.fromisoformat(recorded["captured_at_utc"])
        modified = (
            datetime.fromisoformat(recorded["provider_last_modified_at_utc"])
            if recorded.get("provider_last_modified_at_utc")
            else None
        )
        current = inventory_forecast(
            source,
            captured_at=captured,
            provider_last_modified_at=modified,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid forecast provenance for {source.name}") from exc
    if current != recorded:
        raise ValueError(f"Source does not match its checksum manifest: {source.name}")
    if not recorded.get("eligible_for_prospective_verification"):
        raise ValueError(f"Forecast was not captured before valid time: {source.name}")
    return recorded


def export_one(source: Path, record: dict, region: dict, output_path: Path, staging: Path) -> dict:
    west, east, south, north = region["bbox_west_east_south_north"]
    with xr.open_dataset(source) as dataset:
        if "time" not in dataset.coords or dataset.sizes.get("time") != 1:
            raise ValueError(f"Expected exactly one forecast time in {source.name}")
        valid_day = date.fromisoformat(
            np.datetime_as_string(dataset.time.values[0], unit="D")
        )
        if valid_day.isoformat() != record["valid_date"]:
            raise ValueError(f"NetCDF time disagrees with its manifest: {source.name}")

        latitude = np.asarray(dataset.latitude.values, dtype=np.float32)
        raw_longitude = np.asarray(dataset.longitude.values, dtype=np.float32)
        longitude = ((raw_longitude + 180.0) % 360.0) - 180.0
        if (latitude.ndim != 1 or longitude.ndim != 1
                or np.any(np.diff(latitude) <= 0)
                or np.any(np.diff(longitude) <= 0)):
            raise ValueError(f"Expected increasing one-dimensional coordinates: {source.name}")

        lat_indices = np.flatnonzero((latitude >= south) & (latitude <= north))
        lon_indices = np.flatnonzero((longitude >= west) & (longitude <= east))
        if not len(lat_indices) or not len(lon_indices):
            raise ValueError(f"Frozen region does not overlap {source.name}")
        if (np.any(np.diff(lat_indices) != 1) or np.any(np.diff(lon_indices) != 1)):
            raise ValueError("Frozen region selection is not a contiguous grid window")

        lat_slice = slice(int(lat_indices[0]), int(lat_indices[-1]) + 1)
        lon_slice = slice(int(lon_indices[0]), int(lon_indices[-1]) + 1)
        regional_latitude = latitude[lat_slice]
        regional_longitude = longitude[lon_slice]
        columns: dict[str, np.ndarray] = {
            "latitude": np.repeat(regional_latitude, len(regional_longitude)),
            "longitude": np.tile(regional_longitude, len(regional_latitude)),
        }
        field_metadata: dict[str, dict] = {}
        field_names = []
        for name, variable in dataset.data_vars.items():
            if variable.dims != ("time", "latitude", "longitude"):
                raise ValueError(f"Unexpected dimensions for {name} in {source.name}")
            values = np.asarray(
                variable.isel(time=0, latitude=lat_slice, longitude=lon_slice).values,
                dtype=np.float32,
            )
            if values.shape != (len(regional_latitude), len(regional_longitude)):
                raise ValueError(f"Unexpected grid shape for {name} in {source.name}")
            columns[name] = values.reshape(-1)
            field_names.append(name)
            field_metadata[name] = {
                key: variable.attrs[key]
                for key in ("long_name", "standard_name", "units")
                if key in variable.attrs
            }
        if not field_names:
            raise ValueError(f"No gridded fields were found in {source.name}")

        metadata = {
            "source_file": source.name,
            "source_sha256": record["source_sha256"],
            "source_captured_at_utc": record["captured_at_utc"],
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "product_id": PRODUCT_ID,
            "product_doi": "10.48670/moi-00016",
            "valid_date": valid_day.isoformat(),
            "bulletin_date": record["bulletin_date"],
            "lead_days": record["lead_days"],
            "region_id": region["region_id"],
            "crs": region["crs"],
            "bbox_west_east_south_north": [west, east, south, north],
            "longitude_convention": "degrees east in [-180, 180)",
            "attribution": ATTRIBUTION,
            "encoding": "float32; Zstandard compression level 9; decoded missing cells are NaN",
            "field_metadata": field_metadata,
        }
        table = pa.table(columns).replace_schema_metadata({
            key.encode("utf-8"): json.dumps(value, sort_keys=True).encode("utf-8")
            for key, value in metadata.items()
        })

    staged_path = staging / output_path.name
    pq.write_table(
        table,
        staged_path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        row_group_size=min(65536, table.num_rows),
    )
    return {
        "file": output_path.name,
        "bytes": staged_path.stat().st_size,
        "sha256": sha256_file(staged_path),
        "rows": table.num_rows,
        "columns": table.schema.names,
        "source_file": source.name,
        "source_bytes": record["source_bytes"],
        "source_sha256": record["source_sha256"],
        "valid_date": record["valid_date"],
        "bulletin_date": record["bulletin_date"],
        "lead_days": record["lead_days"],
        "source_manifest": source.with_suffix(".manifest.json").name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("work/datasets/copernicus-glo12-forecast"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/glo12-peninsula-weddell-parquet"),
    )
    parser.add_argument("--run-date", type=date.fromisoformat)
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir() or input_dir.is_symlink():
        raise ValueError(f"Forecast input directory is unavailable: {input_dir}")
    run_day, sources = find_run(input_dir, args.run_date)
    records = [check_source(source) for source in sources]

    config_path = Path(__file__).resolve().parents[1] / "config" / "pilot-verification-region.json"
    region = json.loads(config_path.read_text(encoding="utf-8"))
    bounds = region["bbox_west_east_south_north"]
    if (len(bounds) != 4 or not bounds[0] < bounds[1] or not bounds[2] < bounds[3]
            or region.get("crs") != "EPSG:4326"):
        raise ValueError("Frozen region configuration is invalid")

    output_dir = args.output_dir
    if output_dir.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    output_records = []
    with tempfile.TemporaryDirectory(prefix=".glo12-parquet-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for source, record in zip(sources, records, strict=True):
            output_name = (
                f"{source.stem}__{region['region_id']}.parquet"
            )
            output_records.append(
                export_one(source, record, region, output_dir / output_name, staging)
            )

        manifest = {
            "schema_version": 1,
            "kind": "derived_regional_forecast_subset",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "product_id": PRODUCT_ID,
            "product_url": PRODUCT_URL,
            "product_doi": "10.48670/moi-00016",
            "attribution": ATTRIBUTION,
            "redistribution_attribution": (
                "E.U. Copernicus Marine Service Information; "
                "Source: European Union, Copernicus Marine Service Information or Data (2026), ©Mercator Ocean."
            ),
            "source_run_date": run_day.isoformat(),
            "region": region,
            "coordinates": ["latitude", "longitude"],
            "fields": output_records[0]["columns"][2:],
            "compression": "Zstandard level 9",
            "source_format": "original Copernicus Marine NetCDF",
            "notes": [
                "This package contains only the frozen Peninsula-Weddell bounding box.",
                "It is sea-ice/ocean forecast context, not an iceberg-track observation dataset.",
                "Original NetCDF files remain the source for exact as-issued verification.",
                "Missing NetCDF fill values are decoded to NaN in the Parquet files.",
            ],
            "files": output_records,
        }
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for entry in output_records:
            (staging / entry["file"]).replace(output_dir / entry["file"])
        staged_manifest.replace(output_dir / "manifest.json")

    summary = {
        "run_date": run_day.isoformat(),
        "files": len(output_records),
        "source_bytes": sum(item["source_bytes"] for item in output_records),
        "parquet_bytes": sum(item["bytes"] for item in output_records),
        "region": region["region_id"],
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
