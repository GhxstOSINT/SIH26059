"""Date-blocked, fail-closed research evidence for a frozen SIC transect.

The interval here concerns the *mean concentration along a research line*.
It is not a cellwise hazard probability or a vessel passage assessment.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path

import numpy as np

from ml.assess_glo12_verification import LEADS, MIN_EARLIER_BULLETINS, MIN_LATER_BULLETINS, assess_reports
from ml.research_transect import DEFAULT_ROUTE, load_research_transect


MIN_COMMON_COVERAGE = 0.5
INTERVAL_TARGET = 0.9


class PreFreezeCase(Exception):
    """Valid source evidence that predates route selection, so cannot score it."""


def _case(path: Path, route: dict, route_sha: str) -> dict:
    if assess_reports([path])["rejected_reports"]:
        raise ValueError("Verification sample failed integrity checks")
    record = json.loads(path.read_text(encoding="utf-8"))
    sample = record.get("research_transect")
    if not isinstance(sample, dict) or sample.get("route_id") != route["id"] or sample.get("route_config_sha256") != route_sha:
        raise ValueError("Research transect binding differs from the frozen configuration")
    bulletin = date.fromisoformat(record["bulletin_date"])
    if bulletin < date.fromisoformat(route["eligible_bulletins_from"]):
        raise PreFreezeCase
    if sample.get("evaluation_status") != "single_bulletin_research_sample" or sample.get("navigation_clearance") is not False:
        raise ValueError("Research transect sample is not a scored research case")
    cells = sample.get("unique_observation_cells")
    common = sample.get("common_valid_cells")
    if (not isinstance(cells, int) or not isinstance(common, int) or cells < 1 or not 0 < common <= cells):
        raise ValueError("Invalid research transect cell counts")
    values = {}
    for key in ("forecast_mae_fraction", "forecast_rmse_fraction", "forecast_bias_fraction",
                "observed_mean_sic_fraction", "forecast_mean_sic_fraction", "ice_edge_disagreement_fraction"):
        value = sample.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1:
            raise ValueError(f"Invalid research transect metric: {key}")
        values[key] = float(value)
    if (values["forecast_mae_fraction"] < 0 or values["forecast_rmse_fraction"] < 0
            or values["ice_edge_disagreement_fraction"] < 0
            or values["forecast_rmse_fraction"] + 1e-9 < values["forecast_mae_fraction"]
            or not 0 <= values["observed_mean_sic_fraction"] <= 1
            or not 0 <= values["forecast_mean_sic_fraction"] <= 1
            or abs(values["forecast_mean_sic_fraction"] - values["observed_mean_sic_fraction"]
                   - values["forecast_bias_fraction"]) > 1e-5):
        raise ValueError("Inconsistent research transect metrics")
    baseline = sample.get("persistence_mae_fraction")
    if baseline is not None and (not isinstance(baseline, (int, float)) or not math.isfinite(baseline)
                                 or not 0 <= baseline <= 1):
        raise ValueError("Invalid as-issued persistence metric")
    sample_path = path.with_name(record["samples_file"])
    with np.load(sample_path, allow_pickle=False) as saved:
        required = {"transect_forecast_sic_fraction", "transect_observed_sic_fraction",
                    "transect_observation_uncertainty_fraction"}
        if not required <= set(saved.files):
            raise ValueError("Missing source-bound research transect cell samples")
        predicted = np.asarray(saved["transect_forecast_sic_fraction"], dtype=np.float64)
        observed = np.asarray(saved["transect_observed_sic_fraction"], dtype=np.float64)
        retrieval = np.asarray(saved["transect_observation_uncertainty_fraction"], dtype=np.float64)
        if (any(len(values) != common or not np.all(np.isfinite(values))
                or np.any((values < 0) | (values > 1)) for values in (predicted, observed, retrieval))):
            raise ValueError("Invalid research transect cell samples")
        delta = predicted - observed
        recomputed = {
            "forecast_mae_fraction": float(np.mean(np.abs(delta))),
            "forecast_rmse_fraction": float(np.sqrt(np.mean(delta ** 2))),
            "forecast_bias_fraction": float(np.mean(delta)),
            "forecast_mean_sic_fraction": float(np.mean(predicted)),
            "observed_mean_sic_fraction": float(np.mean(observed)),
            "ice_edge_disagreement_fraction": float(np.mean((predicted >= .15) != (observed >= .15))),
        }
        if any(abs(recomputed[key] - values[key]) > 1e-5 for key in recomputed):
            raise ValueError("Research transect metrics differ from bound cell samples")
        has_baseline = "transect_persistence_sic_fraction" in saved.files
        if (baseline is None) != (not has_baseline):
            raise ValueError("Research transect persistence sample mismatch")
        if has_baseline:
            persistence = np.asarray(saved["transect_persistence_sic_fraction"], dtype=np.float64)
            if (len(persistence) != common or not np.all(np.isfinite(persistence))
                    or np.any((persistence < 0) | (persistence > 1))
                    or abs(float(np.mean(np.abs(persistence - observed))) - baseline) > 1e-5):
                raise ValueError("Research transect persistence metric differs from bound cells")
    return {"bulletin_date": bulletin, "lead_days": record["lead_days"],
            "coverage": common / cells, "mean_error": values["forecast_bias_fraction"],
            "mae": values["forecast_mae_fraction"],
            "ice_edge_disagreement": values["ice_edge_disagreement_fraction"],
            "persistence_mae": baseline}


def summarize(directory: Path, route_path: Path = DEFAULT_ROUTE) -> dict:
    route, route_sha = load_research_transect(route_path)
    rejected = 0
    excluded_pre_freeze = 0
    cases: dict[tuple[int, date], dict] = {}
    paths = (sorted(directory.glob("glo12-*-peninsula-weddell.json"))
             if directory.is_dir() and not directory.is_symlink() else [])
    for path in paths:
        try:
            if path.is_symlink():
                raise ValueError("Symlinked verification report")
            case = _case(path, route, route_sha)
            key = (case["lead_days"], case["bulletin_date"])
            if key in cases:
                raise ValueError("Duplicate lead and bulletin date")
            cases[key] = case
        except PreFreezeCase:
            excluded_pre_freeze += 1
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            rejected += 1
    leads = []
    for lead in LEADS:
        ordered = sorted((case for (days, _), case in cases.items() if days == lead),
                         key=lambda case: case["bulletin_date"])
        evaluable = [case for case in ordered if case["coverage"] >= MIN_COMMON_COVERAGE]
        baseline = [case for case in evaluable if case["persistence_mae"] is not None]
        item = {
            "lead_days": lead,
            "independent_bulletins": len(ordered),
            "evaluable_bulletins": len(evaluable),
            "low_coverage_bulletins": len(ordered) - len(evaluable),
            "minimum_common_cell_coverage_fraction": MIN_COMMON_COVERAGE,
            "mean_bulletin_mae_fraction": (float(np.mean([case["mae"] for case in evaluable])) if evaluable else None),
            "mean_bulletin_ice_edge_disagreement_fraction": (
                float(np.mean([case["ice_edge_disagreement"] for case in evaluable])) if evaluable else None),
            "paired_persistence_bulletins": len(baseline),
            "mean_paired_mae_improvement_vs_persistence_fraction": (
                float(np.mean([case["persistence_mae"] - case["mae"] for case in baseline]))
                if baseline else None),
            "route_mean_sic_interval": None,
        }
        if len(evaluable) >= MIN_EARLIER_BULLETINS + MIN_LATER_BULLETINS and rejected == 0:
            # Keep the first 30 independent bulletin dates for fitting; every later
            # date is held out. Quantile is finite-sample split conformal on *dates*.
            fit = evaluable[:MIN_EARLIER_BULLETINS]
            held_out = evaluable[MIN_EARLIER_BULLETINS:]
            errors = np.sort(np.abs([case["mean_error"] for case in fit]))
            rank = math.ceil((len(errors) + 1) * INTERVAL_TARGET)
            radius = float(errors[min(rank, len(errors)) - 1])
            coverage = float(np.mean([abs(case["mean_error"]) <= radius for case in held_out]))
            item["route_mean_sic_interval"] = {
                "target_coverage": INTERVAL_TARGET,
                "half_width_sic_fraction": radius,
                "fit_bulletins": len(fit),
                "holdout_bulletins": len(held_out),
                "fit_end_bulletin": fit[-1]["bulletin_date"].isoformat(),
                "holdout_start_bulletin": held_out[0]["bulletin_date"].isoformat(),
                "observed_holdout_coverage": coverage,
                "research_reliability_screen_passed": coverage >= INTERVAL_TARGET,
            }
        leads.append(item)
    return {
        "schema_version": 1,
        "scope": "frozen_transect_prospective_research_evidence",
        "route_id": route["id"],
        "route_config_sha256": route_sha,
        "eligible_bulletins_from": route["eligible_bulletins_from"],
        "verified_case_count": len(cases),
        "excluded_pre_freeze_reports": excluded_pre_freeze,
        "rejected_reports": rejected,
        "leads": leads,
        "uncertainty_calibrated": False,
        "operator_decision_ready": False,
        "navigation_clearance": False,
        "limitation": "Date-blocked interval covers mean SIC along a research transect, not vessel safety, hazards, or route travel time.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory, args.route)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
