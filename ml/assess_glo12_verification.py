"""Count independent prospective verification cases before any calibration claim.

Pixel counts are deliberately not treated as independent forecast cases. This
screen is only a prerequisite for later, stratified calibration and comparison
with an as-issued persistence baseline; it never grants navigation clearance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


LEADS = (1, 3, 5, 7)
MIN_EARLIER_BULLETINS = 30
MIN_LATER_BULLETINS = 10


def assess_reports(paths: list[Path]) -> dict:
    dates: dict[int, set[date]] = defaultdict(set)
    rejected = 0
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            bulletin = date.fromisoformat(record["bulletin_date"])
            valid = date.fromisoformat(record["valid_date"])
            lead = record["lead_days"]
            if (record.get("scope") != "single_run_research_verification_sample"
                    or lead not in LEADS or (valid - bulletin).days != lead
                    or record.get("matched_cells", 0) <= 0
                    or record.get("uncertainty_calibrated") is not False
                    or record.get("navigation_clearance") is not False
                    or len(record.get("forecast_source_sha256", "")) != 64
                    or len(record.get("observation_source_sha256", "")) != 64):
                raise ValueError("Invalid verification report")
            sample_name = record.get("samples_file")
            if (not isinstance(sample_name, str) or Path(sample_name).name != sample_name
                    or path.with_name(sample_name).suffix != ".npz"):
                raise ValueError("Invalid matched sample path")
            sample_path = path.with_name(sample_name)
            if not sample_path.is_file() or sample_path.is_symlink():
                raise ValueError("Missing matched sample")
            digest = hashlib.sha256()
            with sample_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record.get("samples_sha256"):
                raise ValueError("Matched sample hash changed")
            with np.load(sample_path, allow_pickle=False) as sample:
                required = {"forecast_sic_fraction", "observed_sic_fraction", "observation_uncertainty_fraction"}
                route_keys = {"transect_forecast_sic_fraction", "transect_observed_sic_fraction",
                              "transect_observation_uncertainty_fraction"}
                allowed = {frozenset(required), frozenset(required | {"persistence_sic_fraction"})}
                if record.get("research_transect", {}).get("evaluation_status") == "single_bulletin_research_sample":
                    allowed |= {frozenset(required | route_keys),
                                frozenset(required | route_keys | {"persistence_sic_fraction",
                                                                     "transect_persistence_sic_fraction"})}
                if (frozenset(sample.files) not in allowed
                        or any(sample[key].ndim != 1 or len(sample[key]) != record["matched_cells"]
                               for key in required)):
                    raise ValueError("Matched sample arrays changed")
                if route_keys <= set(sample.files):
                    count = record["research_transect"]["common_valid_cells"]
                    if any(sample[key].ndim != 1 or len(sample[key]) != count for key in route_keys):
                        raise ValueError("Research transect sample arrays changed")
            dates[lead].add(bulletin)
        except (OSError, ValueError, TypeError, KeyError, BadZipFile, EOFError, json.JSONDecodeError):
            rejected += 1
    lead_results = []
    for lead in LEADS:
        count = len(dates[lead])
        lead_results.append({
            "lead_days": lead,
            "independent_bulletin_dates": count,
            "minimum_earlier_fit_dates": MIN_EARLIER_BULLETINS,
            "minimum_later_holdout_dates": MIN_LATER_BULLETINS,
            "enough_dates_for_research_calibration": count >= MIN_EARLIER_BULLETINS + MIN_LATER_BULLETINS,
        })
    enough = all(item["enough_dates_for_research_calibration"] for item in lead_results)
    return {
        "schema_version": 1,
        "scope": "prospective_forecast_calibration_prerequisites",
        "leads": lead_results,
        "rejected_reports": rejected,
        "research_calibration_sample_gate_passed": enough and rejected == 0,
        "uncertainty_calibrated": False,
        "operator_decision_ready": False,
        "navigation_clearance": False,
        "remaining_requirements": [
            "Later observations must be matched to each original, as-issued forecast bulletin.",
            "Fit only on earlier bulletin dates and evaluate on later held-out dates.",
            "Compare skill and interval coverage with an as-issued persistence baseline.",
            "Check region, ice regime, season, and lead-specific reliability before any operational review.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_reports(sorted(args.directory.glob("*.json")) if args.directory.is_dir() else [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
