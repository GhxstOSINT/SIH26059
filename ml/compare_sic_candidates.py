"""Compare two frozen SIC artifacts on the exact same holdout raster and mask."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from .identity import model_artifact_sha256, model_fingerprint
except ImportError:
    from identity import model_artifact_sha256, model_fingerprint


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(current_model: Path, current_report: Path,
            candidate_model: Path, candidate_report: Path) -> dict:
    old_model, new_model = _load(current_model), _load(candidate_model)
    old, new = _load(current_report), _load(candidate_report)
    for label, model, report in (("current", old_model, old), ("candidate", new_model, new)):
        if model.get("model_fingerprint") != model_fingerprint(model):
            raise ValueError(f"{label} model fingerprint mismatch")
        if model.get("artifact_sha256") != model_artifact_sha256(model):
            raise ValueError(f"{label} model artifact checksum mismatch")
        if report.get("model_fingerprint") != model["model_fingerprint"]:
            raise ValueError(f"{label} evaluation does not belong to its model")
    for key in ("period", "spatial_evaluation", "source_manifest"):
        if old.get(key) != new.get(key):
            raise ValueError(f"Evaluations differ in {key}; common-case comparison is invalid")
    if old["overall"]["grid_cell_samples"] != new["overall"]["grid_cell_samples"]:
        raise ValueError("Evaluations differ in valid sample count")
    old_area = old["area_weighted"]["overall"]
    new_area = new["area_weighted"]["overall"]
    if old_area["evaluated_area_km2_samples"] != new_area["evaluated_area_km2_samples"]:
        raise ValueError("Evaluations differ in area weighted coverage")
    if old_area["persistence_rmse"] != new_area["persistence_rmse"]:
        raise ValueError("Evaluations differ in persistence baseline")
    old_edge = old["area_weighted"]["ice_edge"]["overall"]
    new_edge = new["area_weighted"]["ice_edge"]["overall"]
    if old_edge["persistence_mean_daily_iiee_million_km2"] != new_edge["persistence_mean_daily_iiee_million_km2"]:
        raise ValueError("Evaluations differ in ice-edge baseline")
    rmse_reduction = 100 * (1 - new_area["rmse"] / old_area["rmse"])
    mae_reduction = 100 * (1 - new_area["mae"] / old_area["mae"])
    edge_improves = new_edge["mean_daily_iiee_million_km2"] < old_edge["mean_daily_iiee_million_km2"]
    retain = rmse_reduction < 1.0 or mae_reduction <= 0.0 or not edge_improves
    result = {
        "purpose": "Diagnostic comparison of existing and expanded SIC models on a shared 2025 holdout",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_period": new["period"],
        "source_dataset_id": new["source_manifest"].get("dataset_id"),
        "embedded_source_manifest_canonical_sha256": hashlib.sha256(
            json.dumps(new["source_manifest"], sort_keys=True).encode()).hexdigest(),
        "grid_cell_samples": new["overall"]["grid_cell_samples"],
        "models": {
            "current": {"path": str(current_model), "fingerprint": old_model["model_fingerprint"],
                        "fit_last_target": old_model["training"]["last_train_target_date"]},
            "candidate": {"path": str(candidate_model), "fingerprint": new_model["model_fingerprint"],
                          "fit_last_target": new_model["training"]["last_train_target_date"]},
        },
        "metrics": {
            "area_weighted_rmse": {"current": old_area["rmse"], "candidate": new_area["rmse"],
                                   "persistence": new_area["persistence_rmse"]},
            "area_weighted_mae": {"current": old_area["mae"], "candidate": new_area["mae"]},
            "mean_daily_ice_edge_disagreement_million_km2": {
                "current": old_edge["mean_daily_iiee_million_km2"],
                "candidate": new_edge["mean_daily_iiee_million_km2"],
                "persistence": new_edge["persistence_mean_daily_iiee_million_km2"],
            },
        },
        "candidate_change_percent": {
            "rmse_reduction_vs_current": rmse_reduction,
            "mae_reduction_vs_current": mae_reduction,
        },
        "decision": "retain_current_model" if retain else "requires_new_independent_validation",
        "reason": ("The candidate fails the research screening rule: at least 1% lower area-weighted RMSE, "
                   "lower MAE, and lower ice-edge disagreement on this common 2025 holdout. "
                   "The same 2025 dataset has been examined before and is not a pristine release gate."
                   if retain else "Research screening criteria pass, but this previously examined 2025 holdout "
                   "cannot authorize promotion without a new independent evaluation."),
        "limitations": "Same G02202 product family; 2025 AMSR2 sensor transition; spatially and temporally correlated cells; no route-scale outcome validation.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-model", type=Path, required=True)
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.current_model, args.current_report,
                     args.candidate_model, args.candidate_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"],
                      "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
