"""Evaluate observed-position iceberg drift baselines on the BYU/NIC archive.

Only source-measured positions (per-sensor interpolation marker equal to zero)
are used. Consolidated interpolated daily positions (marker equal to one) are
deliberately excluded to avoid scoring against labels synthesized from
neighboring observations. This is offline research, not an operational forecast.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

from pyproj import Geod

TRACK_COLUMNS = re.compile(r"^(?P<sensor>[a-z0-9]+)_(?P<part>[123])$", re.I)
GEOD = Geod(ellps="WGS84")


def parse_doy(value: str) -> date:
    """Parse BYU YYYYDDD integer dates without accepting invalid day-of-year."""
    raw = int(value)
    year, day_of_year = divmod(raw, 1000)
    if year < 100:
        year += 1900 if year >= 70 else 2000
    if year < 1900 or not 1 <= day_of_year <= (366 if _is_leap(year) else 365):
        raise ValueError(f"Invalid YYYYDDD date: {value!r}")
    return date(year, 1, 1) + timedelta(days=day_of_year - 1)


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_observed_tracks(archive: Path) -> tuple[dict[str, list[tuple[date, float, float]]], dict]:
    """Read per-sensor non-interpolated locations, averaging same-day sensors."""
    tracks: dict[str, list[tuple[date, float, float]]] = {}
    counts = {"files": 0, "rows": 0, "observed_sensor_positions": 0, "track_days": 0}
    with zipfile.ZipFile(archive) as source:
        members = [name for name in source.namelist() if name.lower().endswith(".csv")]
        for member in members:
            if Path(member).name != member.split("/")[-1] or ".." in Path(member).parts:
                raise ValueError(f"Unsafe archive member: {member!r}")
            iceberg_id = Path(member).stem
            daily: dict[date, list[tuple[float, float]]] = {}
            with source.open(member) as binary:
                text = (line.decode("utf-8-sig") for line in binary)
                reader = csv.DictReader(text)
                if not reader.fieldnames or "date" not in reader.fieldnames:
                    raise ValueError(f"Missing date column in {member}")
                sensors: dict[str, dict[str, str]] = {}
                for column in reader.fieldnames:
                    match = TRACK_COLUMNS.fullmatch(column)
                    if match:
                        sensors.setdefault(match["sensor"].lower(), {})[match["part"]] = column
                sensors = {key: cols for key, cols in sensors.items()
                           if {"1", "2"}.issubset(cols)}
                if not sensors:
                    continue
                for row in reader:
                    counts["rows"] += 1
                    day = parse_doy(row["date"])
                    for cols in sensors.values():
                        try:
                            lat = float(row[cols["1"]])
                            lon = float(row[cols["2"]])
                            # BYU's own plotting utility treats absent _3 as no
                            # interpolated observations, and _3 == 1 as interpolated.
                            flag = float(row[cols["3"]]) if "3" in cols else 0.0
                        except (TypeError, ValueError):
                            continue
                        if (flag != 0 or not math.isfinite(lat) or not math.isfinite(lon)
                                or not -90 <= lat <= -50 or not -180 <= lon <= 180):
                            continue
                        daily.setdefault(day, []).append((lat, lon))
                        counts["observed_sensor_positions"] += 1
            if daily:
                track = []
                for day, positions in sorted(daily.items()):
                    # Circular-mean longitude handles tracks crossing +/-180°.
                    lat = sum(point[0] for point in positions) / len(positions)
                    sin_lon = sum(math.sin(math.radians(point[1])) for point in positions)
                    cos_lon = sum(math.cos(math.radians(point[1])) for point in positions)
                    lon = math.degrees(math.atan2(sin_lon, cos_lon))
                    track.append((day, lat, lon))
                tracks[iceberg_id] = track
                counts["track_days"] += len(track)
        counts["files"] = len(members)
    return tracks, counts


def displacement(lon1: float, lat1: float, lon2: float, lat2: float) -> tuple[float, float]:
    """Return geodesic distance in metres and forward azimuth in degrees."""
    azimuth, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    return distance, azimuth


def examples(tracks, max_lead_days: int, max_history_gap_days: int):
    """Yield only triplets of actual observations with bounded time gaps."""
    for iceberg_id, points in tracks.items():
        for index in range(2, len(points)):
            prev, origin, target = points[index - 2:index + 1]
            history_gap = (origin[0] - prev[0]).days
            lead = (target[0] - origin[0]).days
            if not (1 <= history_gap <= max_history_gap_days and 1 <= lead <= max_lead_days):
                continue
            past_distance, past_bearing = displacement(prev[2], prev[1], origin[2], origin[1])
            true_distance, true_bearing = displacement(origin[2], origin[1], target[2], target[1])
            if not all(math.isfinite(x) for x in (past_distance, past_bearing, true_distance, true_bearing)):
                continue
            yield {"iceberg": iceberg_id, "origin_date": origin[0], "target_date": target[0],
                   "lead_days": lead, "origin": origin, "target_lat": target[1], "target_lon": target[2],
                   "past_speed_m_day": past_distance / history_gap,
                   "past_bearing_deg": past_bearing, "truth_distance_m": true_distance,
                   "truth_bearing_deg": true_bearing}


def fit_damping(training_examples) -> float:
    """Fit a pooled least-squares multiplier for constant-velocity vectors."""
    numerator = denominator = 0.0
    for item in training_examples:
        predicted = item["past_speed_m_day"] * item["lead_days"]
        true = item["truth_distance_m"]
        delta = math.radians(item["truth_bearing_deg"] - item["past_bearing_deg"])
        numerator += predicted * true * math.cos(delta)
        denominator += predicted * predicted
    if denominator <= 0:
        raise ValueError("Training split has no measurable non-zero iceberg displacement.")
    return max(0.0, min(2.0, numerator / denominator))


def summarize(errors_m: list[float]) -> dict:
    if not errors_m:
        return {"count": 0, "mean_error_km": None, "median_error_km": None,
                "p90_error_km": None}
    ordered = sorted(errors_m)
    return {"count": len(ordered), "mean_error_km": sum(ordered) / len(ordered) / 1000,
            "median_error_km": statistics.median(ordered) / 1000,
            "p90_error_km": ordered[math.ceil(0.9 * len(ordered)) - 1] / 1000}


def evaluate(archive: Path, train_through: date, test_start: date, test_end: date,
             max_lead_days: int = 14, max_history_gap_days: int = 30) -> dict:
    tracks, data_summary = read_observed_tracks(archive)
    all_examples = list(examples(tracks, max_lead_days, max_history_gap_days))
    training = [e for e in all_examples if e["target_date"] <= train_through]
    testing = [e for e in all_examples if test_start <= e["target_date"] <= test_end]
    damping = fit_damping(training)
    results = {}
    for bucket_name, lower, upper in (("1-2d", 1, 2), ("3-7d", 3, 7), ("8-14d", 8, 14)):
        chosen = [e for e in testing if lower <= e["lead_days"] <= min(upper, max_lead_days)]
        errors = {"persistence": [], "constant_velocity": [], "trained_damped_velocity": []}
        for item in chosen:
            for key, scale in (("persistence", 0.0), ("constant_velocity", 1.0),
                               ("trained_damped_velocity", damping)):
                predicted_distance = item["past_speed_m_day"] * item["lead_days"] * scale
                pred_lon, pred_lat, _ = GEOD.fwd(item["origin"][2], item["origin"][1],
                                                 item["past_bearing_deg"], predicted_distance)
                _, _, error = GEOD.inv(pred_lon, pred_lat,
                                       item["target_lon"], item["target_lat"])
                errors[key].append(error)
        results[bucket_name] = {key: summarize(values) for key, values in errors.items()}
    return {"schema_version": 1, "dataset": "BYU/NIC Antarctic Iceberg Tracking Database v8.0",
            "archive_sha256": sha256_file(archive),
            "task": "Observed-position-only next-report displacement error; not a fixed-time operational forecast",
            "train_target_end": train_through.isoformat(), "test_target_period": {
                "start": test_start.isoformat(), "end": test_end.isoformat()},
            "max_lead_days": max_lead_days, "max_history_gap_days": max_history_gap_days,
            "training_examples": len(training), "test_examples": len(testing),
            "training_icebergs": len({e["iceberg"] for e in training}),
            "test_icebergs": len({e["iceberg"] for e in testing}),
            "fitted_velocity_damping": damping, "data_summary": data_summary,
            "metrics": results,
            "limitations": ["Training targets end before the held-out period; model tuning on the test period is not permitted.",
                            "Only per-sensor rows with interpolation marker 0 are included; marker 1 rows are excluded.",
                            "The target is the next observed report (1-14 days later), not a complete fixed-horizon census.",
                            "Tracks represent a selected large-iceberg archive and have heterogeneous sensor precision/coverage.",
                            "The official landing page describes coverage through 2025-04-22, but this archive contains source-measured positions through 2026-04-29; this evaluation is restricted to the page-declared end pending confirmation.",
                            "Forecast cases from the same iceberg are serially dependent and are not independent samples.",
                            "Iceberg identities can occur in both chronological splits; this tests continuation of known tracks, not cold-start generalization to unseen bergs.",
                            "The source paper reports location accuracy no better than about one sensor pixel (roughly 2.2-8.9 km by sensor), comparable to or larger than some reported errors.",
                            "This retrospective benchmark does not establish live detection, route encounter probability, or navigation safety."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="BYU/NIC consolidated_database_v8.0.zip")
    parser.add_argument("--train-through", type=date.fromisoformat, default=date(2016, 12, 31))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2017, 1, 1))
    parser.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 4, 22))
    parser.add_argument("--max-lead-days", type=int, default=14)
    parser.add_argument("--max-history-gap-days", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.train_through >= args.test_start or args.test_start > args.test_end:
        parser.error("Require train-through < test-start <= test-end.")
    if not 1 <= args.max_lead_days <= 14 or args.max_history_gap_days < 1:
        parser.error("max lead must be 1-14 days and max history gap must be positive.")
    result = evaluate(args.archive, args.train_through, args.test_start, args.test_end,
                      args.max_lead_days, args.max_history_gap_days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "training_examples": result["training_examples"],
                      "test_examples": result["test_examples"],
                      "fitted_velocity_damping": result["fitted_velocity_damping"],
                      "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
