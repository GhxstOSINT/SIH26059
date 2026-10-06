"""Evaluate recursive multi-day SIC rollouts against persistence on a frozen holdout.

The same one-day lag-regression artifact is applied repeatedly without
retraining. This is an evaluation diagnostic, not an operational forecast.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from .evaluate_sic import g02202_cell_geometry
    from .identity import model_fingerprint
    from .train_sic import grids, inventory, verify_g02202_manifest
except ImportError:  # Support direct execution as `python ml/evaluate_multilead_sic.py`.
    from evaluate_sic import g02202_cell_geometry
    from identity import model_fingerprint
    from train_sic import grids, inventory, verify_g02202_manifest


def _phase(day: dt.date) -> tuple[float, float]:
    angle = 2 * math.pi * (day.timetuple().tm_yday - 1) / 365.2425
    return math.sin(angle), math.cos(angle)


def recursive_step(previous: np.ndarray, older: np.ndarray, day: dt.date,
                   coefficients: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Make one clipped lag-regression step while keeping unsupported cells missing."""
    if previous.shape != older.shape or previous.shape != valid.shape:
        raise ValueError("Forecast history and validity mask must have identical grid shapes.")
    sin_phase, cos_phase = _phase(day)
    c0, c1, c2, c3, c4 = coefficients
    forecast = np.clip(c0 + c1 * previous + c2 * older
                       + c3 * sin_phase + c4 * cos_phase, 0.0, 1.0)
    forecast = np.asarray(forecast, dtype=np.float32)
    forecast[~valid] = np.nan
    return forecast


def _new_metrics() -> dict[str, float | int]:
    return {"model_sq": 0.0, "persistence_sq": 0.0, "model_abs": 0.0,
            "persistence_abs": 0.0, "count": 0, "area_model_sq": 0.0,
            "area_persistence_sq": 0.0, "area_model_abs": 0.0,
            "area_persistence_abs": 0.0, "area_sum": 0.0,
            "model_iiee": 0.0, "persistence_iiee": 0.0,
            "model_sie_abs": 0.0, "persistence_sie_abs": 0.0,
            "target_days": 0}


def _accumulate(metrics: dict[str, float | int], prediction: np.ndarray,
                persistence: np.ndarray, target: np.ndarray, valid: np.ndarray,
                area_km2: np.ndarray | None) -> None:
    mask = valid & np.isfinite(prediction) & np.isfinite(persistence) & np.isfinite(target)
    if not mask.any():
        return
    model_delta = (prediction[mask] - target[mask]).astype(np.float64)
    persistence_delta = (persistence[mask] - target[mask]).astype(np.float64)
    metrics["model_sq"] += float(model_delta @ model_delta)
    metrics["persistence_sq"] += float(persistence_delta @ persistence_delta)
    metrics["model_abs"] += float(np.abs(model_delta).sum())
    metrics["persistence_abs"] += float(np.abs(persistence_delta).sum())
    metrics["count"] += int(mask.sum())
    metrics["target_days"] += 1
    if area_km2 is None:
        return
    area = area_km2[mask].astype(np.float64, copy=False)
    model_weighted = area * model_delta ** 2
    persistence_weighted = area * persistence_delta ** 2
    metrics["area_model_sq"] += float(model_weighted.sum())
    metrics["area_persistence_sq"] += float(persistence_weighted.sum())
    metrics["area_model_abs"] += float((area * np.abs(model_delta)).sum())
    metrics["area_persistence_abs"] += float((area * np.abs(persistence_delta)).sum())
    metrics["area_sum"] += float(area.sum())
    threshold = 0.15
    observed_ice = target[mask] >= threshold
    model_ice = prediction[mask] >= threshold
    persistence_ice = persistence[mask] >= threshold
    observed_sie = float(area[observed_ice].sum())
    metrics["model_iiee"] += float(area[model_ice != observed_ice].sum())
    metrics["persistence_iiee"] += float(area[persistence_ice != observed_ice].sum())
    metrics["model_sie_abs"] += abs(float(area[model_ice].sum()) - observed_sie)
    metrics["persistence_sie_abs"] += abs(float(area[persistence_ice].sum()) - observed_sie)


def _summarize(metrics: dict[str, float | int], area_weighted: bool) -> dict[str, float | int | None]:
    count = int(metrics["count"])
    if count == 0:
        return {"target_days": 0, "valid_cell_samples": 0,
                "rmse": None, "mae": None, "persistence_rmse": None,
                "persistence_mae": None}
    result: dict[str, float | int | None] = {
        "target_days": int(metrics["target_days"]),
        "valid_cell_samples": count,
        "rmse": math.sqrt(float(metrics["model_sq"]) / count),
        "mae": float(metrics["model_abs"]) / count,
        "persistence_rmse": math.sqrt(float(metrics["persistence_sq"]) / count),
        "persistence_mae": float(metrics["persistence_abs"]) / count,
    }
    if area_weighted:
        weight = float(metrics["area_sum"])
        result["area_weighted"] = {
            "method": "projected pixel area divided by PROJ ellipsoidal areal_scale at pixel centres",
            "evaluated_area_km2_samples": weight,
            "rmse": math.sqrt(float(metrics["area_model_sq"]) / weight),
            "mae": float(metrics["area_model_abs"]) / weight,
            "persistence_rmse": math.sqrt(float(metrics["area_persistence_sq"]) / weight),
            "persistence_mae": float(metrics["area_persistence_abs"]) / weight,
            "ice_edge": {
                "threshold_sic_fraction": 0.15,
                "aggregation": "mean daily pan-domain error over shared valid cells",
                "mean_daily_iiee_million_km2": float(metrics["model_iiee"]) / (int(metrics["target_days"]) * 1_000_000.0),
                "persistence_mean_daily_iiee_million_km2": float(metrics["persistence_iiee"]) / (int(metrics["target_days"]) * 1_000_000.0),
                "mean_daily_absolute_sie_error_million_km2": float(metrics["model_sie_abs"]) / (int(metrics["target_days"]) * 1_000_000.0),
                "persistence_mean_daily_absolute_sie_error_million_km2": float(metrics["persistence_sie_abs"]) / (int(metrics["target_days"]) * 1_000_000.0),
            },
        }
    return result


def evaluate(data_dir: Path, model: dict, leads: tuple[int, ...] = (1, 3, 5, 7)) -> dict:
    if not leads or any(lead < 1 for lead in leads) or tuple(sorted(set(leads))) != leads:
        raise ValueError("Lead days must be unique, positive, and sorted.")
    coefficients = np.asarray(model.get("coefficients"), dtype=np.float64)
    if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Model must contain five finite lag-regression coefficients.")
    cutoff_value = model.get("training", {}).get("selected_period_end")
    if not cutoff_value:
        raise ValueError("Model must declare training.selected_period_end for leakage-safe evaluation.")
    cutoff = dt.date.fromisoformat(cutoff_value)
    paths = list(data_dir.rglob("*.nc"))
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    verify_g02202_manifest(data_dir, paths, manifest)
    groups = inventory([str(path) for path in paths])
    if not groups:
        raise ValueError("No readable NetCDF inputs found.")
    is_g02202 = isinstance(manifest, dict) and manifest.get("dataset_id") == "G02202"
    area_km2 = None
    first_grid = next(grids(groups), None)
    if first_grid is None:
        raise ValueError("No daily SIC fields found.")
    _, first_values, _ = first_grid
    if is_g02202:
        area_km2, _ = g02202_cell_geometry(data_dir, first_values.shape)

    totals = {lead: _new_metrics() for lead in leads}
    previous_actual: tuple[dt.date, np.ndarray] | None = None
    pending: list[dict] = []
    first_target: dict[int, dt.date | None] = {lead: None for lead in leads}
    last_target: dict[int, dt.date | None] = {lead: None for lead in leads}
    max_lead = max(leads)

    for day, current, ocean_mask in grids(groups):
        if previous_actual is not None and (day - previous_actual[0]).days != 1:
            pending.clear()
        if previous_actual is not None and current.shape != previous_actual[1].shape:
            pending.clear()

        remaining = []
        for forecast in pending:
            lead = (day - forecast["origin_date"]).days
            if day != forecast["next_date"] or not 1 <= lead <= max_lead:
                continue
            prediction = recursive_step(forecast["latest"], forecast["older"],
                                        day, coefficients, forecast["valid"])
            valid = forecast["valid"] & np.isfinite(current)
            if ocean_mask is not None:
                valid &= ocean_mask
            if lead in totals:
                _accumulate(totals[lead], prediction, forecast["origin"],
                            current, valid, area_km2)
                first_target[lead] = first_target[lead] or day
                last_target[lead] = day
            if lead < max_lead:
                forecast["older"] = forecast["latest"]
                forecast["latest"] = prediction
                forecast["next_date"] = day + dt.timedelta(days=1)
                remaining.append(forecast)
        pending = remaining

        if (day >= cutoff and previous_actual is not None
                and (day - previous_actual[0]).days == 1
                and current.shape == previous_actual[1].shape):
            valid = np.isfinite(current) & np.isfinite(previous_actual[1])
            if ocean_mask is not None:
                valid &= ocean_mask
            if valid.any():
                pending.append({"origin_date": day, "next_date": day + dt.timedelta(days=1),
                                "origin": current.copy(), "latest": current.copy(),
                                "older": previous_actual[1].copy(), "valid": valid})

        previous_actual = (day, current)

    if not any(int(metrics["count"]) for metrics in totals.values()):
        raise ValueError("No frozen post-cutoff multi-lead forecasts had valid target cells.")

    source_manifest_sha256 = None
    if manifest_path.exists():
        source_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result = {
        "status": "frozen_recursive_holdout_diagnostic",
        "model_id": model.get("model_id"),
        "model_fingerprint": model_fingerprint(model),
        "model_artifact_sha256": model.get("artifact_sha256"),
        "training_period_end": cutoff.isoformat(),
        "evaluation_product": manifest.get("dataset_id") if isinstance(manifest, dict) else None,
        "evaluation_manifest_sha256": source_manifest_sha256,
        "evaluation_method": {
            "type": "recursive autoregressive rollout of the frozen one-day lag-regression model",
            "leads_days": list(leads),
            "baseline": "persistence of the latest observed origin field",
            "quality": "source QA masking and ocean mask from ml.train_sic.grids; forecasts reset across date gaps",
            "target_scope": "chronological targets strictly after the model's selected training-period end",
        },
        "by_lead": {
            str(lead): {
                "period": {"first_target_date": first_target[lead].isoformat() if first_target[lead] else None,
                           "last_target_date": last_target[lead].isoformat() if last_target[lead] else None},
                **_summarize(totals[lead], area_km2 is not None),
            }
            for lead in leads
        },
        "warning": ("Retrospective research diagnostic only. Recursive errors can compound with lead; the held-out targets are from the same G02202 product family, not independent sensor validation. "
                    "Not an operational forecast, route-skill estimate, or navigation guidance."),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing chronological daily NetCDF fields")
    parser.add_argument("--model", default="models/sic_g02202_1989_2016.json")
    parser.add_argument("--output", help="Optional path for the evaluation report")
    parser.add_argument("--leads", default="1,3,5,7", help="Comma-separated positive forecast lead days")
    args = parser.parse_args()
    try:
        leads = tuple(sorted(set(int(value) for value in args.leads.split(","))))
        if not leads or any(lead < 1 or lead > 30 for lead in leads):
            raise ValueError("Lead days must be between 1 and 30.")
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))
        report = evaluate(Path(args.data_dir), model, leads)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
