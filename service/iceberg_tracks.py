"""Checksummed, historical-only access to source-measured BYU/NIC tracks."""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
import zipfile

from fastapi import HTTPException
from pyproj import Geod

from ml.evaluate_iceberg_tracks import read_observed_tracks


DATASET_ID = "BYU/NIC Antarctic Iceberg Tracking Database"
DATASET_VERSION = "8.0"
# The v8.0 landing page declares coverage through this date. The local archive
# contains later records, but BYU has not confirmed the discrepancy.
PUBLISHED_COVERAGE_END = date(2025, 4, 22)
MAX_WINDOW_DAYS = 90
GEOD = Geod(ellps="WGS84")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _read_archive(path_string: str, expected_sha256: str,
                  file_size: int, modified_ns: int,
                  manifest_size: int, manifest_modified_ns: int) -> tuple[dict[str, list], str]:
    """Load a manifest-bound archive once per process and file signature."""
    del file_size, modified_ns, manifest_size, manifest_modified_ns  # Cache-key-only metadata.
    path = Path(path_string)
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise HTTPException(status_code=503, detail="The iceberg archive checksum does not match its manifest.")
    try:
        tracks, _ = read_observed_tracks(path)
    except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=503, detail="The checksummed iceberg archive could not be parsed safely.") from exc
    return tracks, actual


def load_verified_tracks(archive: Path) -> tuple[dict[str, list], str]:
    manifest_path = archive.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["archive"]
        stat = archive.stat()
        manifest_stat = manifest_path.stat()
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="A valid BYU/NIC archive manifest is required.") from exc
    if (not isinstance(entry, dict)
            or manifest.get("dataset") != DATASET_ID or manifest.get("version") != DATASET_VERSION
            or manifest.get("complete") is not True or entry.get("file") != archive.name
            or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64):
        raise HTTPException(status_code=503, detail="The iceberg archive manifest is incomplete or identifies another dataset.")
    try:
        tracks, actual_sha = _read_archive(str(archive.resolve()), entry["sha256"],
                                           stat.st_size, stat.st_mtime_ns,
                                           manifest_stat.st_size, manifest_stat.st_mtime_ns)
    except HTTPException:
        raise
    return tracks, actual_sha


def observed_tracks_geojson(archive: Path, as_of: date,
                            window_days: int = 30) -> dict[str, Any]:
    if not 1 <= window_days <= MAX_WINDOW_DAYS:
        raise HTTPException(status_code=422, detail=f"window_days must be between 1 and {MAX_WINDOW_DAYS}.")
    if as_of > PUBLISHED_COVERAGE_END:
        raise HTTPException(status_code=422, detail=(
            f"The published BYU/NIC v8.0 coverage ends on {PUBLISHED_COVERAGE_END.isoformat()}; "
            "later records in the local archive are withheld pending source confirmation."
        ))

    tracks, archive_sha = load_verified_tracks(archive)
    first_day = as_of - timedelta(days=window_days - 1)
    features = []
    for iceberg_id, points in sorted(tracks.items()):
        selected = [(day, lat, lon) for day, lat, lon in points if first_day <= day <= as_of]
        if not selected:
            continue
        last_day = selected[-1][0]
        coordinates = [[float(lon), float(lat)] for _, lat, lon in selected]
        geometry = ({"type": "Point", "coordinates": coordinates[-1]} if len(coordinates) == 1
                    else {"type": "LineString", "coordinates": coordinates})
        features.append({
            "type": "Feature",
            "id": iceberg_id,
            "geometry": geometry,
            "properties": {
                "iceberg_id": iceberg_id,
                "first_observed_date": selected[0][0].isoformat(),
                "last_observed_date": last_day.isoformat(),
                "age_days": (as_of - last_day).days,
                "source_measured_position_count": len(selected),
                "position_method": "source-measured sensor locations; interpolated markers excluded; same-day sensor positions averaged",
            },
        })
    return {
        "type": "FeatureCollection",
        "dataset": {
            "id": DATASET_ID,
            "version": DATASET_VERSION,
            "source_url": "https://www.scp.byu.edu/iceberg/default.html",
            "paper_doi": "10.1109/JSTARS.2017.2784186",
            "archive_sha256": archive_sha,
            "published_coverage_end": PUBLISHED_COVERAGE_END.isoformat(),
        },
        "query": {"as_of_date": as_of.isoformat(), "window_days": window_days,
                  "window_start": first_day.isoformat()},
        "features": features,
        "status": "historical_observations_only",
        "warning": (
            "Historical tracks of selected large icebergs only. These observations are not a complete iceberg census, "
            "live positions, drift forecasts, encounter probabilities, or navigation guidance."
        ),
    }


def historical_short_iceberg_projection(archive: Path, iceberg_id: str,
                                        as_of: date, lead_days: int,
                                        evidence_path: Path,
                                        evidence_sha256: str) -> dict[str, Any]:
    """Project a known historical track for 1-2 days from measured locations."""
    if not 1 <= lead_days <= 2:
        raise HTTPException(status_code=422, detail="lead_days must be 1 or 2.")
    if as_of > PUBLISHED_COVERAGE_END:
        raise HTTPException(status_code=422, detail=(
            f"The published BYU/NIC v8.0 coverage ends on {PUBLISHED_COVERAGE_END.isoformat()}."
        ))
    tracks, archive_sha = load_verified_tracks(archive)
    points = tracks.get(iceberg_id)
    if not points:
        raise HTTPException(status_code=404, detail="No source-measured track exists for this iceberg ID.")
    prior = [point for point in points if point[0] <= as_of]
    if len(prior) < 2 or prior[-1][0] != as_of:
        raise HTTPException(status_code=422, detail="Two source-measured positions ending on as_of_date are required.")
    previous, origin = prior[-2:]
    gap = (origin[0] - previous[0]).days
    if not 1 <= gap <= 30:
        raise HTTPException(status_code=422, detail="Measured position history must be 1-30 days apart.")
    try:
        if _sha256_file(evidence_path) != evidence_sha256:
            raise HTTPException(status_code=503, detail="The iceberg evaluation report checksum changed.")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="The iceberg evaluation report is unavailable.") from exc
    if (evidence.get("archive_sha256") != archive_sha
            or evidence.get("task") != "Observed-position-only next-report displacement error; not a fixed-time operational forecast"):
        raise HTTPException(status_code=503, detail="The iceberg evaluation does not match the mounted archive.")
    bearing, _, previous_distance_m = GEOD.inv(previous[2], previous[1], origin[2], origin[1])
    projected_lon, projected_lat, _ = GEOD.fwd(origin[2], origin[1], bearing,
                                               previous_distance_m * lead_days / gap)
    benchmark = evidence["metrics"]["1-2d"]
    return {
        "iceberg_id": iceberg_id,
        "status": "historical_research_projection",
        "as_of_date": as_of.isoformat(),
        "target_date": (as_of + timedelta(days=lead_days)).isoformat(),
        "lead_days": lead_days,
        "observations": [
            {"date": day.isoformat(), "longitude": lon, "latitude": lat}
            for day, lat, lon in (previous, origin)
        ],
        "projected_position": {"longitude": projected_lon, "latitude": projected_lat},
        "method": "Geodesic constant velocity from the last two source-measured positions",
        "evidence": {
            "source": DATASET_ID, "version": DATASET_VERSION,
            "archive_sha256": archive_sha, "report_sha256": evidence_sha256,
            "holdout_period": evidence["test_target_period"],
            "one_to_two_day_next_report_mean_error_km": benchmark["constant_velocity"]["mean_error_km"],
            "persistence_next_report_mean_error_km": benchmark["persistence"]["mean_error_km"],
            "n_next_reports": benchmark["constant_velocity"]["count"],
        },
        "warning": "The benchmark scores the next measured report, not fixed-time positions. Known large bergs only; this projection is neither live detection nor a route-encounter or navigation forecast.",
    }
