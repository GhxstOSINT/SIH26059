"""Compare a pinned SIC model with persistence on frozen, cross-product corridors.

Each VIIRS platform is scored separately. Missing calendar days remain in the
coverage denominator; no platform scores are pooled into pseudo-independent rows.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_route_corridors import evaluate
    from .verify_initial_model import verify_initial_model
except ImportError:
    from evaluate_route_corridors import evaluate
    from verify_initial_model import verify_initial_model


ALLOWED_DATASETS = {
    "noaacwVIIRSnppiceconcSP06Daily": "S-NPP",
    "noaacwVIIRSn21iceconcSP06Daily": "NOAA-21",
    "noaacwVIIRSn20iceconcSP06Daily": "NOAA-20",
}


def calendar_coverage(route: dict[str, Any], start: date, end: date) -> dict[str, Any]:
    """Count every target date after the two required lag days, including gaps."""
    scheduled_days = max(0, (end - start).days - 1)
    days = route["days"]
    if scheduled_days < 1:
        raise ValueError("Insufficient calendar period")
    if not days:
        return {"scheduled_target_days": scheduled_days, "eligible_three_day_windows": 0,
                "missing_three_day_windows": scheduled_days, "scored_days": 0,
                "common_valid_points": 0, "calendar_coverage_fraction": 0.0}
    point_counts = {day["sample_points"] for day in days}
    if len(point_counts) != 1:
        raise ValueError("Inconsistent route point count or insufficient calendar period")
    sample_points = point_counts.pop()
    if sample_points < 1 or len(days) > scheduled_days:
        raise ValueError("Invalid corridor day inventory")
    expected_dates = {start + timedelta(days=offset) for offset in range(2, (end - start).days + 1)}
    actual_dates = [date.fromisoformat(day["date"]) for day in days]
    if len(actual_dates) != len(set(actual_dates)) or not set(actual_dates) <= expected_dates:
        raise ValueError("Duplicate or out-of-period evaluation date")
    valid_points = sum(day["common_valid_points"] for day in days)
    if valid_points > scheduled_days * sample_points:
        raise ValueError("Valid point count exceeds the calendar opportunity")
    return {
        "scheduled_target_days": scheduled_days,
        "eligible_three_day_windows": len(days),
        "missing_three_day_windows": scheduled_days - len(days),
        "scored_days": route["summary"]["scored_days"],
        "common_valid_points": valid_points,
        "calendar_coverage_fraction": valid_points / (scheduled_days * sample_points),
    }


def evaluate_bundle(
    sources: list[Path], model_path: Path, routes_path: Path,
    *, min_days: int = 30, min_calendar_coverage: float = .5,
) -> dict[str, Any]:
    if len(sources) < 2 or min_days < 1 or not 0 < min_calendar_coverage <= 1:
        raise ValueError("At least two platform archives and valid thresholds are required")
    selection = verify_initial_model(runtime_model_path=model_path)
    results = []
    dataset_ids: set[str] = set()
    frozen_selection = None
    for source in sources:
        manifest_raw = (source / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        dataset_id = manifest.get("dataset_id")
        if dataset_id not in ALLOWED_DATASETS or dataset_id in dataset_ids:
            raise ValueError("Each archive must be a distinct allowlisted Antarctic VIIRS platform")
        dataset_ids.add(dataset_id)
        if (manifest.get("complete") is not True
                or manifest.get("platform", ALLOWED_DATASETS[dataset_id]) != ALLOWED_DATASETS[dataset_id]):
            raise ValueError("Incomplete or mismatched VIIRS platform manifest")
        subset = manifest.get("selection") or {}
        period = (subset.get("start_date"), subset.get("end_date"))
        grid = (tuple(subset.get("center_lon_lat") or []), subset.get("half_width_km"),
                subset.get("crs"), subset.get("projected_bounds_m"))
        if grid[2] != "EPSG:3976" or period[0] is None or period[1] is None:
            raise ValueError("VIIRS subset lacks a supported frozen period or grid")
        if frozen_selection is None:
            frozen_selection = (period, grid)
        elif frozen_selection != (period, grid):
            raise ValueError("Cross-platform comparisons require the same period and region")
        start, end = date.fromisoformat(period[0]), date.fromisoformat(period[1])
        if start <= date(2016, 12, 31):
            raise ValueError("Evaluation period overlaps the pinned model training archive")
        report = evaluate(source, model_path, routes_path, min_days=min_days,
                          min_coverage=min_calendar_coverage)
        if report["dataset_id"] != dataset_id:
            raise ValueError("Evaluator dataset ID does not match source manifest")
        routes = []
        for route in report["routes"]:
            coverage = calendar_coverage(route, start, end)
            enough = (coverage["scored_days"] >= min_days
                      and coverage["calendar_coverage_fraction"] >= min_calendar_coverage)
            routes.append({"id": route["id"], "coverage": coverage,
                           "coverage_gate": "passed" if enough else "insufficient_evidence",
                           "metrics_on_common_valid_points": route["summary"]})
        results.append({"platform": ALLOWED_DATASETS[dataset_id], "dataset_id": dataset_id,
                        "source_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                        "routes": routes})
    route_ids = [route["id"] for route in results[0]["routes"]]
    if any([route["id"] for route in sensor["routes"]] != route_ids for sensor in results):
        raise ValueError("Platform corridor sets differ")
    # An adequate comparison requires at least two separately adequate platforms
    # for each route. Even a passed research gate is never navigation clearance.
    gate = "passed_research_coverage" if all(
        sum(sensor["routes"][i]["coverage_gate"] == "passed" for sensor in results) >= 2
        for i in range(len(route_ids))
    ) else "insufficient_evidence"
    return {"schema_version": 1, "scope": "research_only", "quality_gate": gate,
            "operational_decision_ready": False,
            "model_fingerprint": selection["model_fingerprint"],
            "model_artifact_sha256": selection["artifact_sha256"],
            "routes_sha256": hashlib.sha256(routes_path.read_bytes()).hexdigest(),
            "period": {"start_date": frozen_selection[0][0], "end_date": frozen_selection[0][1]},
            "thresholds": {"min_scored_days_per_route_platform": min_days,
                           "min_calendar_coverage_fraction": min_calendar_coverage,
                           "min_adequate_platforms_per_route": 2},
            "platforms": results,
            "limitations": [
                "VIIRS is a cross-product sensor check, not a vessel-outcome or navigation-safety label.",
                "The same frozen corridors have been examined before; this is not a pristine prospective release test.",
                "Nearby pixels and days are correlated; platform scores must not be pooled as independent trials.",
                "The report excludes weather, bathymetry, vessel class, live iceberg hazards and calibrated uncertainty.",
            ]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="Repeat for at least two complete platform archives")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--min-coverage", type=float, default=.5)
    args = parser.parse_args()
    report = evaluate_bundle(args.source, args.model, args.routes, min_days=args.min_days,
                             min_calendar_coverage=args.min_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"quality_gate": report["quality_gate"], "output": str(args.output),
                      "platforms": [{"platform": sensor["platform"], "routes": [
                          {"id": route["id"], **route["coverage"]} for route in sensor["routes"]]}
                          for sensor in report["platforms"]]}, indent=2))
    return 0 if report["quality_gate"] == "passed_research_coverage" else 2


if __name__ == "__main__":
    raise SystemExit(main())
