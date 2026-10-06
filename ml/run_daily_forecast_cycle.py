"""One idempotent daily research collection and verification cycle.

Run after the Copernicus bulletin is available. Observation retrievals may lag;
missing ones remain pending and are retried on the next cycle. No forecast is
shown as calibrated or suitable for navigation by this workflow.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from ml.assess_glo12_verification import assess_reports
from ml.fetch_glo12_forecast import fetch_run
from ml.fetch_osi_sic_observation import fetch_observation, load_pilot_region
from ml.research_transect import DEFAULT_ROUTE, load_research_transect
from ml.summarize_research_transect import summarize
from ml.verify_glo12_forecast import match_run, write_match


def run_cycle(day: date, forecast_dir: Path, observation_dir: Path,
              verification_dir: Path, *, fetch_forecast: bool = True,
              max_pending: int = 16) -> dict:
    if day > datetime.now(timezone.utc).date():
        raise ValueError("Cannot run a future daily cycle")
    if max_pending < 1 or max_pending > 100:
        raise ValueError("max_pending must be between 1 and 100")
    region_id, region_bbox, region_sha = load_pilot_region()
    research_route, research_route_sha = load_research_transect()
    if fetch_forecast:
        fetch_run(day, forecast_dir)
    observation_capture_pending = []
    if fetch_forecast:
        for offset in (2, 1):
            capture_day = day - timedelta(days=offset)
            try:
                fetch_observation(capture_day, observation_dir, region_bbox, region_id, region_sha)
            except Exception:
                observation_capture_pending.append(capture_day.isoformat())
    candidates = []
    for path in forecast_dir.glob("*.manifest.json") if forecast_dir.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            valid = date.fromisoformat(record["valid_date"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if valid >= day or not record.get("eligible_for_prospective_verification"):
            continue
        output = verification_dir / f"glo12-{record['bulletin_date']}-lead-{record['lead_days']}-{region_id}.json"
        if output.exists():
            if assess_reports([output])["rejected_reports"]:
                raise ValueError(f"Existing verification evidence failed checks: {output.name}")
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing.get("research_transect", {}).get("route_config_sha256") != research_route_sha:
                raise ValueError(f"Existing research transect definition differs: {output.name}")
            continue
        candidates.append((valid, path, output))
    candidates.sort(key=lambda item: (item[0], item[1].name))
    completed = []
    pending = []
    for valid, manifest, output in candidates[:max_pending]:
        try:
            observed = fetch_observation(valid, observation_dir, region_bbox, region_id, region_sha)
        except Exception:
            # Provider outages, delayed publication, and auth failures require
            # operator review; never leak account details from an exception.
            pending.append({"valid_date": valid.isoformat(), "reason": "observation_unavailable_or_fetch_failed"})
            continue
        observation_path = observation_dir / observed["source_file"]
        forecast_record = json.loads(manifest.read_text(encoding="utf-8"))
        candidates_for_baseline = []
        for prior_manifest in observation_dir.glob("*.manifest.json"):
            try:
                prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
                if (prior.get("region_id") == region_id
                        and prior.get("region_config_sha256") == region_sha
                        and datetime.fromisoformat(prior["captured_at_utc"])
                        < datetime.fromisoformat(forecast_record["forecast_base_time_utc"])
                        and date.fromisoformat(prior["period"]["start_date"])
                        < date.fromisoformat(forecast_record["bulletin_date"])):
                    candidates_for_baseline.append((prior["period"]["start_date"], prior_manifest))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        baseline_manifest = max(candidates_for_baseline)[1] if candidates_for_baseline else None
        report, samples = match_run(manifest, observation_path, baseline_manifest, DEFAULT_ROUTE)
        report["observation_region_id"] = region_id
        report["observation_region_config_sha256"] = region_sha
        write_match(report, samples, output)
        completed.append({"bulletin_date": report["bulletin_date"],
                          "valid_date": report["valid_date"], "lead_days": report["lead_days"],
                          "matched_cells": report["matched_cells"],
                          "as_issued_persistence_compared": report["as_issued_persistence_comparison"] is not None,
                          "research_transect_status": report["research_transect"]["evaluation_status"]})
    readiness = assess_reports(sorted(verification_dir.glob(f"*-{region_id}.json"))
                               if verification_dir.is_dir() else [])
    research_transect_evidence = summarize(verification_dir, DEFAULT_ROUTE)
    return {
        "cycle_date_utc": day.isoformat(),
        "observation_region_id": region_id,
        "observation_bbox_west_east_south_north": list(region_bbox),
        "observation_region_config_sha256": region_sha,
        "research_transect_id": research_route["id"],
        "research_transect_config_sha256": research_route_sha,
        "forecast_archive_attempted": fetch_forecast,
        "completed_verifications": completed,
        "observation_capture_pending_dates": observation_capture_pending,
        "pending_observations": pending,
        "remaining_backlog": max(0, len(candidates) - max_pending),
        "calibration": readiness,
        "research_transect_evidence": research_transect_evidence,
        "navigation_clearance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--forecast-dir", type=Path, default=Path("work/datasets/copernicus-glo12-forecast"))
    parser.add_argument("--observation-dir", type=Path, default=Path("work/datasets/copernicus-osi-sic-prospective"))
    parser.add_argument("--verification-dir", type=Path, default=Path("work/verification"))
    parser.add_argument("--skip-forecast-fetch", action="store_true",
                        help="Inspect/retry pending observations without contacting the forecast provider")
    parser.add_argument("--max-pending", type=int, default=16)
    parser.add_argument("--status-output", type=Path)
    args = parser.parse_args()
    status = run_cycle(args.date, args.forecast_dir, args.observation_dir, args.verification_dir,
                       fetch_forecast=not args.skip_forecast_fetch, max_pending=args.max_pending)
    if args.status_output:
        args.status_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.status_output.with_suffix(args.status_output.suffix + ".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.status_output)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
