"""Calibrate radial next-report iceberg error corridors on a chronological split.

The product is a distributional research diagnostic for the BYU/NIC observed-track
archive, not a fixed-time trajectory or a live route-encounter probability.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path
import tempfile

try:
    from .evaluate_iceberg_tracks import (
        GEOD, examples, fit_damping, read_observed_tracks, sha256_file,
    )
except ImportError:
    from evaluate_iceberg_tracks import (
        GEOD, examples, fit_damping, read_observed_tracks, sha256_file,
    )


BUCKETS = (("1-2d", 1, 2), ("3-7d", 3, 7), ("8-14d", 8, 14))


def verify_archive_manifest(archive: Path) -> tuple[dict, str]:
    manifest_path = archive.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest.get("archive", {})
    if not manifest.get("complete") or item.get("file") != archive.name:
        raise ValueError("BYU/NIC archive does not match a complete adjacent manifest")
    if archive.stat().st_size != item.get("bytes"):
        raise ValueError("BYU/NIC archive byte count does not match its manifest")
    digest = sha256_file(archive)
    if digest != item.get("sha256"):
        raise ValueError("BYU/NIC archive SHA-256 does not match its manifest")
    return manifest, digest


def prediction_error_km(item: dict, velocity_scale: float) -> float:
    distance_m = item["past_speed_m_day"] * item["lead_days"] * velocity_scale
    origin = item["origin"]
    lon, lat, _ = GEOD.fwd(origin[2], origin[1], item["past_bearing_deg"], distance_m)
    _, _, error_m = GEOD.inv(lon, lat, item["target_lon"], item["target_lat"])
    return error_m / 1000.0


def upper_quantile(values: list[float], quantile: float) -> float:
    if not values or not 0 < quantile < 1:
        raise ValueError("A non-empty sample and a quantile in (0,1) are required")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def summarize_corridor(calibration_errors: list[float], test_errors: list[float]) -> dict:
    if not calibration_errors or not test_errors:
        return {"status": "insufficient_samples", "calibration_count": len(calibration_errors),
                "holdout_count": len(test_errors)}
    radii = {"80": upper_quantile(calibration_errors, 0.8),
             "90": upper_quantile(calibration_errors, 0.9)}
    return {
        "status": "scored_next_report_distribution",
        "calibration_count": len(calibration_errors),
        "holdout_count": len(test_errors),
        "calibration_error_km": {
            "median": upper_quantile(calibration_errors, 0.5),
            "p80": radii["80"], "p90": radii["90"],
        },
        "holdout_error_km": {
            "mean": sum(test_errors) / len(test_errors),
            "median": upper_quantile(test_errors, 0.5),
            "p80": upper_quantile(test_errors, 0.8),
            "p90": upper_quantile(test_errors, 0.9),
        },
        "calibrated_radius_km": radii,
        "holdout_empirical_coverage": {
            "80_percent_radius": sum(error <= radii["80"] for error in test_errors) / len(test_errors),
            "90_percent_radius": sum(error <= radii["90"] for error in test_errors) / len(test_errors),
        },
    }


def evaluate(archive: Path, fit_through: date, calibration_start: date,
             calibration_end: date, test_start: date, test_end: date,
             max_lead_days: int = 14, max_history_gap_days: int = 30) -> dict:
    if not fit_through < calibration_start <= calibration_end < test_start <= test_end:
        raise ValueError("Require fit end < calibration period < untouched test period")
    manifest, archive_hash = verify_archive_manifest(archive)
    tracks, data_summary = read_observed_tracks(archive)
    all_examples = list(examples(tracks, max_lead_days, max_history_gap_days))
    training = [item for item in all_examples if item["target_date"] <= fit_through]
    calibration = [item for item in all_examples
                   if calibration_start <= item["target_date"] <= calibration_end]
    holdout = [item for item in all_examples
               if test_start <= item["target_date"] <= test_end]
    damping = fit_damping(training)
    methods = {"persistence": 0.0, "constant_velocity": 1.0,
               "trained_damped_velocity": damping}
    result = {}
    for bucket_name, lower, upper in BUCKETS:
        calibration_group = [item for item in calibration
                             if lower <= item["lead_days"] <= min(upper, max_lead_days)]
        holdout_group = [item for item in holdout
                         if lower <= item["lead_days"] <= min(upper, max_lead_days)]
        result[bucket_name] = {
            name: summarize_corridor(
                [prediction_error_km(item, scale) for item in calibration_group],
                [prediction_error_km(item, scale) for item in holdout_group],
            ) for name, scale in methods.items()
        }
    return {
        "schema_version": 1,
        "dataset": "BYU/NIC Antarctic Iceberg Tracking Database v8.0",
        "archive_sha256": archive_hash,
        "manifest_sha256": sha256_file(archive.parent / "manifest.json"),
        "method": "Empirical radial error quantiles for the next source-measured report; calibration uses 2014-2016 and is frozen before the 2017-2025 holdout.",
        "fit_target_end": fit_through.isoformat(),
        "calibration_target_period": {"start": calibration_start.isoformat(), "end": calibration_end.isoformat()},
        "holdout_target_period": {"start": test_start.isoformat(), "end": test_end.isoformat()},
        "max_lead_days": max_lead_days,
        "max_history_gap_days": max_history_gap_days,
        "training_examples": len(training),
        "calibration_examples": len(calibration),
        "holdout_examples": len(holdout),
        "fitted_velocity_damping_from_fit_period": damping,
        "data_summary": data_summary,
        "by_next_report_lead": result,
        "limitations": [
            "The target is the next source-measured report, not a fixed-time forecast.",
            "Circular radii summarize historical position-error distributions and do not represent ellipse orientation, physical process uncertainty, or true-position accuracy.",
            "Temporal and within-track dependence violate the exchangeability assumptions behind formal conformal coverage guarantees; empirical test coverage is descriptive only.",
            "Coverage is evaluated on known tracks, not on independent bergs or independent trials.",
            "BYU/NIC positions have heterogeneous sensor precision; the published paper reports location accuracy of roughly one sensor pixel, about 2.2-8.9 km by sensor.",
            "The selected large-berg archive is not a complete live iceberg-obstacle feed; this report does not estimate route encounter probability or navigational clearance.",
        ],
    }


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--fit-through", type=date.fromisoformat, default=date(2013, 12, 31))
    parser.add_argument("--calibration-start", type=date.fromisoformat, default=date(2014, 1, 1))
    parser.add_argument("--calibration-end", type=date.fromisoformat, default=date(2016, 12, 31))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2017, 1, 1))
    parser.add_argument("--test-end", type=date.fromisoformat, default=date(2025, 4, 22))
    parser.add_argument("--max-lead-days", type=int, default=14)
    parser.add_argument("--max-history-gap-days", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/byu-iceberg-v8-error-corridors.json"))
    args = parser.parse_args()
    if not 1 <= args.max_lead_days <= 14 or args.max_history_gap_days < 1:
        parser.error("max lead must be 1-14 days and max history gap must be positive")
    try:
        report = evaluate(args.archive, args.fit_through, args.calibration_start,
                          args.calibration_end, args.test_start, args.test_end,
                          args.max_lead_days, args.max_history_gap_days)
        atomic_write(args.output, report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    compact = {bucket: {model: value.get("holdout_empirical_coverage")
                        for model, value in rows.items()}
               for bucket, rows in report["by_next_report_lead"].items()}
    print(json.dumps({"output": str(args.output), "calibration_examples": report["calibration_examples"],
                      "holdout_examples": report["holdout_examples"],
                      "coverage": compact}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
