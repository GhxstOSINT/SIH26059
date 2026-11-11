"""Export the source-measured daily BYU/NIC iceberg track positions to Parquet."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.evaluate_iceberg_tracks import (
    PUBLISHED_COVERAGE_END,
    PUBLISHED_COVERAGE_START,
    read_observed_tracks,
    restrict_tracks_to_published_coverage,
)


DATASET_ID = "BYU/NIC Antarctic Iceberg Tracking Database"
DATASET_VERSION = "8.0"
DEFAULT_ARCHIVE = Path("work/datasets/byu-iceberg-v8/consolidated_database_v8.0.zip")
DEFAULT_SOURCE_MANIFEST = Path("work/datasets/byu-iceberg-v8/manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/byu-iceberg-v8-tracks-parquet")
OUTPUT_NAME = "byu_iceberg_observed_track_days_v8.0.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.archive.is_symlink() or not args.archive.is_file():
        raise ValueError(f"Regular BYU/NIC archive is required: {args.archive}")
    if args.source_manifest.is_symlink() or not args.source_manifest.is_file():
        raise ValueError(f"BYU/NIC source manifest is required: {args.source_manifest}")
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    archive_record = source.get("archive", {})
    if (source.get("dataset") != DATASET_ID or source.get("version") != DATASET_VERSION
            or source.get("complete") is not True
            or archive_record.get("file") != args.archive.name
            or archive_record.get("bytes") != args.archive.stat().st_size
            or archive_record.get("sha256") != sha256_file(args.archive)):
        raise ValueError("BYU/NIC archive does not match its pinned source manifest")

    source_tracks, source_counts = read_observed_tracks(args.archive)
    tracks, coverage_counts = restrict_tracks_to_published_coverage(source_tracks)
    inspection = source.get("inspection", {})
    source_dates = [day for points in source_tracks.values() for day, _, _ in points]
    archive_start = min(source_dates).isoformat()
    archive_end = max(source_dates).isoformat()
    if (inspection.get("earliest_source_measured_date") not in (None, archive_start)
            or inspection.get("latest_source_measured_date") not in (None, archive_end)):
        raise ValueError("BYU/NIC archive dates do not match its pinned source manifest")
    published_coverage = {
        "start": PUBLISHED_COVERAGE_START.isoformat(),
        "end": PUBLISHED_COVERAGE_END.isoformat(),
        "source_track_days": coverage_counts["source_track_days"],
        "published_track_days": coverage_counts["published_track_days"],
        "excluded_before_start_track_days": coverage_counts["excluded_before_start_track_days"],
        "excluded_after_end_track_days": coverage_counts["excluded_after_end_track_days"],
        "source_icebergs": coverage_counts["source_icebergs"],
        "published_icebergs": coverage_counts["published_icebergs"],
    }
    rows = [
        (iceberg_id, day, latitude, longitude)
        for iceberg_id in sorted(tracks)
        for day, latitude, longitude in tracks[iceberg_id]
    ]
    if not rows:
        raise ValueError("No source-measured iceberg track days were found")

    table = pa.table({
        "iceberg_id": pa.array([row[0] for row in rows], type=pa.string()),
        "observation_date": pa.array([row[1] for row in rows], type=pa.date32()),
        "latitude": pa.array([row[2] for row in rows], type=pa.float64()),
        "longitude": pa.array([row[3] for row in rows], type=pa.float64()),
    })
    schema_metadata = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "source_archive_file": args.archive.name,
        "source_archive_sha256": archive_record["sha256"],
        "source_url": source.get("download_url"),
        "citation": source.get("citation") or source.get("paper"),
        "source_counts": source_counts,
        "published_coverage": published_coverage,
        "position_method": (
            "Per-sensor interpolation marker 0 only; valid Antarctic coordinates; "
            "same-day sensor positions averaged, with circular-mean longitude."
        ),
        "date_coverage_note": (
            f"BYU's v8.0 page declares coverage from {PUBLISHED_COVERAGE_START.isoformat()} "
            f"through {PUBLISHED_COVERAGE_END.isoformat()}. The source ZIP contains "
            f"measured track days from {archive_start} through {archive_end}; this Parquet "
            "excludes all daily points outside BYU's declared interval."
        ),
    }
    table = table.replace_schema_metadata({
        key.encode("utf-8"): json.dumps(value, sort_keys=True).encode("utf-8")
        for key, value in schema_metadata.items()
    })

    output_dir = args.output_dir
    if output_dir.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        expected_files = {OUTPUT_NAME, "manifest.json"}
        existing_files = {item.name for item in output_dir.iterdir()}
        if existing_files != expected_files or any(item.is_dir() for item in output_dir.iterdir()):
            raise ValueError(f"Refusing to overwrite an output directory with unexpected contents: {output_dir}")
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if (previous.get("dataset") != DATASET_ID or previous.get("version") != DATASET_VERSION
                or previous.get("kind") != "source_measured_daily_track_positions"):
            raise ValueError(f"Refusing to overwrite an unrelated Parquet dataset: {output_dir}")

    with tempfile.TemporaryDirectory(prefix=".byu-tracks-", dir=output_dir) as temporary:
        staging = Path(temporary)
        staged_parquet = staging / OUTPUT_NAME
        pq.write_table(
            table,
            staged_parquet,
            compression="zstd",
            compression_level=9,
            use_dictionary=["iceberg_id"],
            row_group_size=65536,
        )
        parquet_size = staged_parquet.stat().st_size
        digest = sha256_file(staged_parquet)
        file_record = {
            "file": OUTPUT_NAME,
            "bytes": parquet_size,
            "sha256": digest,
            "rows": table.num_rows,
            "icebergs": len(tracks),
            "columns": table.schema.names,
            "compression": "Zstandard level 9",
        }
        manifest = {
            "schema_version": 2,
            "dataset": DATASET_ID,
            "version": DATASET_VERSION,
            "complete": True,
            "kind": "source_measured_daily_track_positions",
            "source_url": source["download_url"],
            "citation": source.get("citation") or source.get("paper"),
            "source_terms_note": source.get("terms") or source.get("terms_note"),
            "source_archive": {
                "file": args.archive.name,
                "bytes": args.archive.stat().st_size,
                "sha256": archive_record["sha256"],
            },
            "source_coverage": {
                "provider_page_url": "https://www.scp.byu.edu/iceberg/default.html",
                "provider_page_start": PUBLISHED_COVERAGE_START.isoformat(),
                "provider_page_through": PUBLISHED_COVERAGE_END.isoformat(),
                "archive_measured_from": archive_start,
                "archive_measured_through": archive_end,
                "service_start": PUBLISHED_COVERAGE_START.isoformat(),
                "service_cutoff": PUBLISHED_COVERAGE_END.isoformat(),
                "excluded_before_start_track_days": coverage_counts["excluded_before_start_track_days"],
                "excluded_after_end_track_days": coverage_counts["excluded_after_end_track_days"],
            },
            "processing": {
                "source_csv_files": source_counts["files"],
                "source_rows": source_counts["rows"],
                "source_measured_sensor_positions": source_counts["observed_sensor_positions"],
                "source_daily_track_points": source_counts["track_days"],
                "daily_track_points": table.num_rows,
                "source_icebergs_with_measured_positions": source_counts["icebergs"],
                "icebergs_in_published_coverage": len(tracks),
                "interpolated_positions_included": False,
                "same_day_sensor_positions": "averaged by latitude and circular-mean longitude",
            },
            "parquet": file_record,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
        staged_parquet.replace(output_dir / OUTPUT_NAME)
        staged_manifest.replace(output_dir / "manifest.json")

    print(json.dumps({
        "parquet": str(output_dir / OUTPUT_NAME),
        "source_zip_bytes": args.archive.stat().st_size,
        "parquet_bytes": parquet_size,
        "reduction_percent_vs_zip": round(100 * (1 - parquet_size / args.archive.stat().st_size), 2),
        "source_csv_files": source_counts["files"],
        "source_rows": source_counts["rows"],
        "source_measured_sensor_positions": source_counts["observed_sensor_positions"],
        "source_daily_track_points": source_counts["track_days"],
        "published_daily_track_points": table.num_rows,
        "excluded_before_start_track_days": coverage_counts["excluded_before_start_track_days"],
        "excluded_after_end_track_days": coverage_counts["excluded_after_end_track_days"],
        "icebergs": len(tracks),
    }, indent=2))


if __name__ == "__main__":
    main()
