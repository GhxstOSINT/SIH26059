"""Run expanding-window, year-blocked validation for Antarctic SIC lag regression.

The archive is streamed once. Each fold is fit only on earlier calendar years and
scored on later, disjoint years against persistence and a fold-specific monthly
climatology. This is a research diagnostic, not an operational forecast evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from .evaluate_sic import g02202_cell_geometry
    from .train_sic import features, grids, inventory, verify_g02202_manifest
except ImportError:  # Support direct execution as `python ml/rolling_evaluate_sic.py`.
    from evaluate_sic import g02202_cell_geometry
    from train_sic import features, grids, inventory, verify_g02202_manifest


def build_folds(first_year: int, last_year: int, train_years: int, validation_years: int,
                step_years: int) -> list[dict]:
    if min(train_years, validation_years, step_years) < 1:
        raise ValueError("Window lengths and step must be positive whole years.")
    folds = []
    train_end = first_year + train_years - 1
    while train_end + validation_years <= last_year:
        validation_start = train_end + 1
        validation_end = validation_start + validation_years - 1
        folds.append({"train_start": dt.date(first_year, 1, 1),
                      "train_end": dt.date(train_end, 12, 31),
                      "validation_start": dt.date(validation_start, 1, 1),
                      "validation_end": dt.date(validation_end, 12, 31),
                      "weights": None,
                      "gram": None,
                      "rhs": None,
                      "climatology_sum": None,
                      "climatology_count": None,
                      "train_samples": 0,
                      "totals": _new_metrics(),
                      "climatology_totals": _new_baseline_metrics(),
                      "climatology_months": {},
                      "months": {}})
        train_end += step_years
    if not folds:
        raise ValueError("The archive is too short for one complete train/validation fold.")
    return folds


def _new_metrics() -> dict[str, float | int]:
    return {"model_sq": 0.0, "persistence_sq": 0.0, "absolute": 0.0, "samples": 0}


def _new_area_metrics() -> dict[str, float]:
    return {"model_sq": 0.0, "persistence_sq": 0.0, "absolute": 0.0, "weight_sum": 0.0}


def _new_baseline_metrics() -> dict[str, float | int]:
    return {"squared": 0.0, "absolute": 0.0, "samples": 0}


def _new_area_baseline_metrics() -> dict[str, float]:
    return {"squared": 0.0, "absolute": 0.0, "weight_sum": 0.0}


def _accumulate_baseline(metrics: dict, predicted: np.ndarray, target: np.ndarray) -> None:
    difference = predicted - target
    metrics["squared"] += float(difference @ difference)
    metrics["absolute"] += float(np.abs(difference).sum())
    metrics["samples"] += int(difference.size)


def _accumulate_area_baseline(metrics: dict, predicted: np.ndarray, target: np.ndarray,
                              area_km2: np.ndarray) -> None:
    difference = predicted - target
    metrics["squared"] += float(np.sum(area_km2 * difference ** 2))
    metrics["absolute"] += float(np.sum(area_km2 * np.abs(difference)))
    metrics["weight_sum"] += float(np.sum(area_km2))


def _accumulate(metrics: dict, predicted: np.ndarray, target: np.ndarray,
                baseline: np.ndarray) -> None:
    difference = predicted - target
    persistence_difference = baseline - target
    metrics["model_sq"] += float(difference @ difference)
    metrics["persistence_sq"] += float(persistence_difference @ persistence_difference)
    metrics["absolute"] += float(np.abs(difference).sum())
    metrics["samples"] += int(difference.size)


def _accumulate_area(metrics: dict[str, float], predicted: np.ndarray, target: np.ndarray,
                     baseline: np.ndarray, area_km2: np.ndarray) -> None:
    difference = predicted - target
    persistence_difference = baseline - target
    metrics["model_sq"] += float(np.sum(area_km2 * difference ** 2))
    metrics["persistence_sq"] += float(np.sum(area_km2 * persistence_difference ** 2))
    metrics["absolute"] += float(np.sum(area_km2 * np.abs(difference)))
    metrics["weight_sum"] += float(np.sum(area_km2))


def _summarize(metrics: dict) -> dict:
    count = int(metrics["samples"])
    if count == 0:
        return {"grid_cell_samples": 0, "mae": None, "rmse": None, "persistence_rmse": None}
    return {"grid_cell_samples": count, "mae": metrics["absolute"] / count,
            "rmse": math.sqrt(metrics["model_sq"] / count),
            "persistence_rmse": math.sqrt(metrics["persistence_sq"] / count)}


def _summarize_area(metrics: dict[str, float]) -> dict[str, float | None]:
    area_samples = float(metrics["weight_sum"])
    if area_samples <= 0:
        return {"evaluated_area_km2_samples": 0.0, "mae": None, "rmse": None,
                "persistence_rmse": None}
    return {"evaluated_area_km2_samples": area_samples,
            "mae": float(metrics["absolute"]) / area_samples,
            "rmse": math.sqrt(float(metrics["model_sq"]) / area_samples),
            "persistence_rmse": math.sqrt(float(metrics["persistence_sq"]) / area_samples)}


def _summarize_baseline(metrics: dict) -> dict[str, float | int | None]:
    count = int(metrics["samples"])
    if count == 0:
        return {"grid_cell_samples": 0, "mae": None, "rmse": None}
    return {"grid_cell_samples": count, "mae": float(metrics["absolute"]) / count,
            "rmse": math.sqrt(float(metrics["squared"]) / count)}


def _summarize_area_baseline(metrics: dict[str, float]) -> dict[str, float | None]:
    area_samples = float(metrics["weight_sum"])
    if area_samples <= 0:
        return {"evaluated_area_km2_samples": 0.0, "mae": None, "rmse": None}
    return {"evaluated_area_km2_samples": area_samples,
            "mae": float(metrics["absolute"]) / area_samples,
            "rmse": math.sqrt(float(metrics["squared"]) / area_samples)}


def evaluate(data_dir: Path, train_years: int, validation_years: int, step_years: int) -> dict:
    paths = [str(path) for path in data_dir.rglob("*.nc")]
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    verify_g02202_manifest(data_dir, paths, manifest)
    groups = inventory(paths)
    if not groups:
        raise ValueError("No readable NetCDF inputs found.")
    dates = sorted(date for _, _, records, _, _ in groups for date, _ in records)
    folds = build_folds(dates[0].year, dates[-1].year, train_years, validation_years, step_years)
    area_weights = latitude = None
    if (manifest or {}).get("dataset_id") == "G02202":
        area_weights, latitude = g02202_cell_geometry(data_dir, (332, 316))
        with xr.open_dataset(groups[0][0]) as first_dataset:
            sic_variable = first_dataset[groups[0][1]]
            grid_shape = (sic_variable.isel(time=0).squeeze().shape if "time" in sic_variable.dims
                          else sic_variable.squeeze().shape)
        if tuple(area_weights.shape) != tuple(grid_shape) or latitude.shape != area_weights.shape:
            raise ValueError("G02202 area/latitude geometry does not match the input raster grid.")
    latitude_bands = None
    if latitude is not None:
        latitude_bands = {
            "south_of_80S": latitude <= -80,
            "80S_to_70S": (latitude > -80) & (latitude <= -70),
            "70S_to_60S": (latitude > -70) & (latitude <= -60),
            "north_of_60S": latitude > -60,
        }
    for fold in folds:
        fold["area_totals"] = _new_area_metrics()
        fold["area_months"] = {}
        fold["area_climatology_totals"] = _new_area_baseline_metrics()
        fold["area_climatology_months"] = {}
        fold["area_latitude"] = {name: _new_area_metrics() for name in latitude_bands or {}}
    previous_older = previous = None
    trained = 0
    for day, current, ocean_mask in grids(groups):
        if previous is not None and previous_older is not None:
            contiguous = ((day - previous[0]).days == 1 and (previous[0] - previous_older[0]).days == 1
                          and current.shape == previous[1].shape == previous_older[1].shape)
            if contiguous:
                valid = np.isfinite(current) & np.isfinite(previous[1]) & np.isfinite(previous_older[1])
                if ocean_mask is not None:
                    valid &= ocean_mask
                if valid.any():
                    x = features(previous[1], previous_older[1], day)
                    for fold in folds:
                        if fold["train_start"] <= day <= fold["train_end"]:
                            if fold["gram"] is None:
                                fold["gram"] = np.zeros((5, 5), dtype=np.float64)
                                fold["rhs"] = np.zeros(5, dtype=np.float64)
                                fold["climatology_sum"] = np.zeros((12, *current.shape), dtype=np.float64)
                                fold["climatology_count"] = np.zeros((12, *current.shape), dtype=np.uint32)
                            xv = x[valid].astype(np.float64)
                            yv = current[valid].astype(np.float64)
                            fold["gram"] += xv.T @ xv
                            fold["rhs"] += xv.T @ yv
                            climatology_month = day.month - 1
                            fold["climatology_sum"][climatology_month][valid] += current[valid]
                            fold["climatology_count"][climatology_month][valid] += 1
                            fold["train_samples"] += int(yv.size)
                        elif day == fold["validation_start"]:
                            if fold["gram"] is None or fold["train_samples"] < 10_000:
                                raise ValueError(f"Fold ending {fold['train_end']} has insufficient fit samples.")
                            regularizer = np.diag([0.0, 0.02, 0.02, 0.02, 0.02])
                            fold["weights"] = np.linalg.solve(fold["gram"] + regularizer, fold["rhs"])
                            fold["gram"] = fold["rhs"] = None
                            trained += 1
                        if fold["weights"] is not None and fold["validation_start"] <= day <= fold["validation_end"]:
                            prediction = np.clip(np.einsum("...k,k->...", x, fold["weights"]), 0, 1)
                            _accumulate(fold["totals"], prediction[valid], current[valid], previous[1][valid])
                            month = fold["months"].setdefault(f"{day.month:02d}", _new_metrics())
                            _accumulate(month, prediction[valid], current[valid], previous[1][valid])
                            climatology_month = day.month - 1
                            climatology_available = valid & (fold["climatology_count"][climatology_month] > 0)
                            if climatology_available.any():
                                counts = fold["climatology_count"][climatology_month][climatology_available]
                                climate = (fold["climatology_sum"][climatology_month][climatology_available]
                                           / counts)
                                climate_target = current[climatology_available]
                                _accumulate_baseline(fold["climatology_totals"], climate, climate_target)
                                climate_month = fold["climatology_months"].setdefault(
                                    f"{day.month:02d}", _new_baseline_metrics())
                                _accumulate_baseline(climate_month, climate, climate_target)
                            if area_weights is not None:
                                _accumulate_area(fold["area_totals"], prediction[valid], current[valid],
                                                 previous[1][valid], area_weights[valid])
                                area_month = fold["area_months"].setdefault(f"{day.month:02d}", _new_area_metrics())
                                _accumulate_area(area_month, prediction[valid], current[valid],
                                                 previous[1][valid], area_weights[valid])
                                for band_name, band_mask in latitude_bands.items():
                                    band_valid = valid & band_mask
                                    if band_valid.any():
                                        _accumulate_area(fold["area_latitude"][band_name], prediction[band_valid],
                                                         current[band_valid], previous[1][band_valid],
                                                         area_weights[band_valid])
                                if climatology_available.any():
                                    climate_area = area_weights[climatology_available]
                                    _accumulate_area_baseline(fold["area_climatology_totals"], climate,
                                                              climate_target, climate_area)
                                    climate_area_month = fold["area_climatology_months"].setdefault(
                                        f"{day.month:02d}", _new_area_baseline_metrics())
                                    _accumulate_area_baseline(climate_area_month, climate, climate_target,
                                                              climate_area)
        if previous is not None and (day - previous[0]).days != 1:
            previous_older = None
        else:
            previous_older = previous
        previous = (day, current)

    if trained != len(folds):
        details = [(fold["train_samples"], fold["validation_start"].isoformat(), fold["weights"] is not None)
                   for fold in folds]
        raise ValueError(f"One or more folds could not be fitted; fold progress={details}. Inspect date continuity and source coverage.")
    rendered_folds = []
    for fold in folds:
        if not fold["totals"]["samples"]:
            raise ValueError(f"Validation window {fold['validation_start']}–{fold['validation_end']} has no samples.")
        rendered = {
            "training_period": {"start": fold["train_start"].isoformat(), "end": fold["train_end"].isoformat()},
            "validation_period": {"start": fold["validation_start"].isoformat(), "end": fold["validation_end"].isoformat()},
            "training_grid_cell_samples": fold["train_samples"],
            "metrics": _summarize(fold["totals"]),
            "monthly_climatology": _summarize_baseline(fold["climatology_totals"]),
            "monthly_climatology_by_target_month": {
                month: _summarize_baseline(metrics)
                for month, metrics in sorted(fold["climatology_months"].items())},
            "by_target_month": {month: _summarize(metrics) for month, metrics in sorted(fold["months"].items())},
        }
        if area_weights is not None:
            rendered["area_weighted"] = {
                "overall": _summarize_area(fold["area_totals"]),
                "monthly_climatology": _summarize_area_baseline(fold["area_climatology_totals"]),
                "monthly_climatology_by_target_month": {
                    month: _summarize_area_baseline(metrics)
                    for month, metrics in sorted(fold["area_climatology_months"].items())},
                "by_target_month": {month: _summarize_area(metrics)
                                    for month, metrics in sorted(fold["area_months"].items())},
                "by_latitude_band": {name: _summarize_area(metrics)
                                     for name, metrics in sorted(fold["area_latitude"].items())
                                     if metrics["weight_sum"] > 0},
            }
        rendered_folds.append(rendered)
    totals = _new_metrics()
    month_totals: dict[str, dict] = {}
    climatology_totals = _new_baseline_metrics()
    climatology_month_totals: dict[str, dict] = {}
    area_totals = _new_area_metrics()
    area_climatology_totals = _new_area_baseline_metrics()
    area_month_totals: dict[str, dict] = {}
    area_climatology_month_totals: dict[str, dict] = {}
    area_latitude_totals = {name: _new_area_metrics() for name in latitude_bands or {}}
    for fold in folds:
        for key in ("model_sq", "persistence_sq", "absolute", "samples"):
            totals[key] += fold["totals"][key]
        for month, metrics in fold["months"].items():
            bucket = month_totals.setdefault(month, _new_metrics())
            for key in bucket:
                bucket[key] += metrics[key]
        for key in climatology_totals:
            climatology_totals[key] += fold["climatology_totals"][key]
        for month, metrics in fold["climatology_months"].items():
            bucket = climatology_month_totals.setdefault(month, _new_baseline_metrics())
            for key in bucket:
                bucket[key] += metrics[key]
        if area_weights is not None:
            for key in area_totals:
                area_totals[key] += fold["area_totals"][key]
            for key in area_climatology_totals:
                area_climatology_totals[key] += fold["area_climatology_totals"][key]
            for month, metrics in fold["area_months"].items():
                bucket = area_month_totals.setdefault(month, _new_area_metrics())
                for key in bucket:
                    bucket[key] += metrics[key]
            for month, metrics in fold["area_climatology_months"].items():
                bucket = area_climatology_month_totals.setdefault(month, _new_area_baseline_metrics())
                for key in bucket:
                    bucket[key] += metrics[key]
            for band, metrics in fold["area_latitude"].items():
                for key in area_latitude_totals[band]:
                    area_latitude_totals[band][key] += metrics[key]
    warning = ("Retrospective SIC research diagnostic only. Monthly climatology is fit separately on each fold's earlier training dates. Cell-pooled scores give each valid cell equal weight; area-weighted scores estimate ground area from the native polar projection. Neither metric is route-scale or navigational validation."
               if area_weights is not None else
               "Retrospective SIC research diagnostic only. Monthly climatology is fit separately on each fold's earlier training dates. Cell-pooled scores give each valid cell equal weight and are not route-scale or navigational validation.")
    aggregation = ("Pooled cell-equal and estimated ground-area-weighted metrics across disjoint validation periods."
                   if area_weights is not None else
                   "Pooled cell-equal metrics across disjoint validation periods.")
    report = {
        "evaluation": "expanding-window calendar-year rolling origin",
        "dataset_id": (manifest or {}).get("dataset_id"),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None,
        "source_manifest": manifest,
        "fold_count": len(folds),
        "window_configuration": {"initial_training_years": train_years,
                                 "validation_years": validation_years, "step_years": step_years},
        "aggregation": aggregation,
        "overall": _summarize(totals),
        "monthly_climatology": _summarize_baseline(climatology_totals),
        "monthly_climatology_by_target_month_pooled": {
            month: _summarize_baseline(metrics)
            for month, metrics in sorted(climatology_month_totals.items())},
        "by_target_month_pooled": {month: _summarize(metrics) for month, metrics in sorted(month_totals.items())},
        "folds": rendered_folds,
        "warning": warning,
    }
    if area_weights is not None:
        report["area_weighted"] = {
            "method": "projected pixel area divided by PROJ ellipsoidal areal_scale at each pixel centre",
            "area_unit": "km2",
            "overall": _summarize_area(area_totals),
            "monthly_climatology": _summarize_area_baseline(area_climatology_totals),
            "monthly_climatology_by_target_month": {
                month: _summarize_area_baseline(metrics)
                for month, metrics in sorted(area_climatology_month_totals.items())},
            "by_target_month": {month: _summarize_area(metrics)
                                for month, metrics in sorted(area_month_totals.items())},
            "by_latitude_band": {name: _summarize_area(metrics)
                                 for name, metrics in sorted(area_latitude_totals.items())
                                 if metrics["weight_sum"] > 0},
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Daily Antarctic NetCDF archive")
    parser.add_argument("--train-years", type=int, default=8)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--step-years", type=int, default=2)
    parser.add_argument("--output", help="Optional JSON report output path")
    args = parser.parse_args()
    try:
        report = evaluate(Path(args.data_dir), args.train_years, args.validation_years, args.step_years)
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
