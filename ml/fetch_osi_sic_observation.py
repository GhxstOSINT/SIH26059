"""Capture a dated OSI-SAF South SIC observation for prospective verification.

The default small Peninsula box is a research verification region, not a route
or Antarctic-wide safety assessment. Capture time is recorded separately from
the observation date; historical subsets do not prove earlier availability.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from ml.inventory_copernicus_sic import DATASET_ID, DATASET_VERSION, inventory


PENINSULA_BBOX = (-65.0, -55.0, -68.0, -64.0)
PILOT_REGION_PATH = Path(__file__).resolve().parents[1] / "config" / "pilot-verification-region.json"


def load_pilot_region(path: Path = PILOT_REGION_PATH) -> tuple[str, tuple[float, float, float, float], str]:
    raw = path.read_bytes()
    record = json.loads(raw)
    region_id = record.get("region_id")
    bbox = record.get("bbox_west_east_south_north")
    if (record.get("schema_version") != 1 or record.get("navigation_clearance") is not False
            or not isinstance(region_id, str) or not re.fullmatch(r"[a-z0-9-]{2,40}", region_id)
            or not isinstance(bbox, list) or len(bbox) != 4
            or any(not isinstance(value, (float, int)) for value in bbox)):
        raise ValueError("Invalid frozen pilot research region")
    west, east, south, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 0):
        raise ValueError("Invalid Southern Hemisphere research region bounds")
    return region_id, (west, east, south, north), hashlib.sha256(raw).hexdigest()


def fetch_observation(day: date, output_dir: Path,
                      bbox: tuple[float, float, float, float] = PENINSULA_BBOX,
                      region_id: str = "peninsula", region_config_sha256: str | None = None) -> dict:
    import copernicusmarine

    if day >= datetime.now(timezone.utc).date():
        raise ValueError("The observation date must have passed before acquisition")
    west, east, south, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 0):
        raise ValueError("Expected a bounded Southern Hemisphere geographic box")
    if not re.fullmatch(r"[a-z0-9-]{2,40}", region_id):
        raise ValueError("Invalid research region id")
    if output_dir.is_symlink():
        raise ValueError("Observation directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"osi-saf-south-{day:%Y%m%d}-{region_id}.nc"
    source = output_dir / name
    manifest_path = output_dir / f"{source.stem}.manifest.json"
    if manifest_path.exists():
        if manifest_path.is_symlink():
            raise ValueError("Observation manifest may not be a symlink")
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = inventory(source, day, day)
        for field in ("source_file", "source_bytes", "source_sha256", "dataset_id", "dataset_version", "period"):
            if recorded.get(field) != current[field]:
                raise ValueError("Existing observation or manifest changed")
        if recorded.get("requested_bbox") != list(bbox):
            raise ValueError("Existing observation box differs")
        if recorded.get("region_id") != region_id or recorded.get("region_config_sha256") != region_config_sha256:
            raise ValueError("Existing observation region definition differs")
        return recorded
    if source.exists():
        raise ValueError("Observation file exists without its capture manifest")
    response = copernicusmarine.subset(
        dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
        variables=["ice_conc", "status_flag", "total_uncertainty"],
        minimum_longitude=west, maximum_longitude=east,
        minimum_latitude=south, maximum_latitude=north,
        start_datetime=day.isoformat(), end_datetime=day.isoformat(),
        output_filename=name, output_directory=output_dir,
        coordinates_selection_method="inside", skip_existing=True,
        disable_progress_bar=True,
    )
    if Path(response.file_path).resolve() != source.resolve() or not source.is_file():
        raise ValueError("Provider response did not match the requested observation file")
    report = inventory(source, day, day)
    report.update({
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_bbox": list(bbox),
        "region_id": region_id,
        "region_config_sha256": region_config_sha256,
        "availability_before_capture_verified": False,
        "navigation_clearance": False,
    })
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("work/datasets/copernicus-osi-sic-prospective"))
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "EAST", "SOUTH", "NORTH"),
                        default=PENINSULA_BBOX)
    parser.add_argument("--pilot-region", action="store_true", help="Use the frozen Peninsula–Weddell research envelope")
    args = parser.parse_args()
    region_id, bbox, region_sha = (load_pilot_region() if args.pilot_region
                                   else ("peninsula", tuple(args.bbox), None))
    result = fetch_observation(args.date, args.output_dir, bbox, region_id, region_sha)
    print(json.dumps({"observation_date": args.date.isoformat(), "source_sha256": result["source_sha256"],
                      "captured_at_utc": result["captured_at_utc"], "navigation_clearance": False}, indent=2))


if __name__ == "__main__":
    main()
