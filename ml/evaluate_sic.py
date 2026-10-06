"""Evaluate a trained SIC model on a separate daily NetCDF dataset/product."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Proj, Transformer

try:
    from .identity import model_fingerprint
    from .train_sic import features, grids, inventory, verify_g02202_manifest
except ImportError:  # Support direct execution as `python ml/evaluate_sic.py`.
    from identity import model_fingerprint
    from train_sic import features, grids, inventory, verify_g02202_manifest


def coarsen_grid(values: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return values
    height = values.shape[0] // factor * factor
    width = values.shape[1] // factor * factor
    if not height or not width:
        raise ValueError(f"Coarsening factor {factor} exceeds grid shape {values.shape}")
    blocks = values[:height, :width].reshape(height // factor, factor, width // factor, factor)
    valid = np.isfinite(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0).sum(axis=(1, 3))
    return np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float32), where=count > 0)


def g02202_cell_geometry(data_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Derive true cell-area weights and latitude from the product's own projected grid."""
    first_file = next(data_dir.rglob("*.nc"), None)
    if first_file is None:
        raise ValueError("No NetCDF grid available for area weighting")
    with xr.open_dataset(first_file) as ds:
        if not {"x", "y", "crs"}.issubset(ds.variables):
            raise ValueError("G02202 area weighting requires x, y and crs variables")
        x, y = np.asarray(ds.x.values, dtype=np.float64), np.asarray(ds.y.values, dtype=np.float64)
        crs_attrs = ds.crs.attrs
        crs_text = crs_attrs.get("crs_wkt") or crs_attrs.get("spatial_ref")
    if (len(y), len(x)) != shape or not crs_text:
        raise ValueError("G02202 CRS/grid metadata does not match the evaluated raster")
    dx, dy = float(np.median(np.abs(np.diff(x)))), float(np.median(np.abs(np.diff(y))))
    if dx <= 0 or dy <= 0 or not np.allclose(np.diff(x), dx) or not np.allclose(np.abs(np.diff(y)), dy):
        raise ValueError("Area weighting requires a regular rectilinear projected grid")
    source_crs = CRS.from_wkt(crs_text)
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    xx, yy = np.meshgrid(x, y)
    longitude, latitude = transformer.transform(xx, yy)
    # PROJ's areal_scale is projected area / ellipsoidal ground area. A map
    # pixel's projected area divided by that factor estimates its ground area.
    areal_scale = np.asarray(Proj(source_crs).get_factors(longitude, latitude).areal_scale, dtype=np.float64)
    area_km2 = (dx * dy / areal_scale) / 1_000_000.0
    if area_km2.shape != shape or not np.all(np.isfinite(area_km2)) or np.any(area_km2 <= 0):
        raise ValueError("Projection factors produced invalid G02202 ground-cell areas")
    return area_km2, np.asarray(latitude, dtype=np.float64)


def blank_totals() -> dict[str, float | int]:
    return {"model_sq": 0.0, "baseline_sq": 0.0, "abs": 0.0, "count": 0}


def add_deltas(metrics: dict[str, float | int], model_delta: np.ndarray,
               baseline_delta: np.ndarray) -> None:
    metrics["model_sq"] += float(model_delta @ model_delta)
    metrics["abs"] += float(np.abs(model_delta).sum())
    metrics["baseline_sq"] += float(baseline_delta @ baseline_delta)
    metrics["count"] += len(model_delta)


def add_weighted_deltas(metrics: dict[str, float], model_delta: np.ndarray,
                        baseline_delta: np.ndarray, weights: np.ndarray) -> None:
    metrics["model_sq"] += float(np.sum(weights * model_delta ** 2))
    metrics["baseline_sq"] += float(np.sum(weights * baseline_delta ** 2))
    metrics["abs"] += float(np.sum(weights * np.abs(model_delta)))
    metrics["weight_sum"] += float(np.sum(weights))


def blank_ice_edge_totals() -> dict[str, float | int]:
    return {"model_iiee_km2": 0.0, "persistence_iiee_km2": 0.0,
            "model_sie_abs_error_km2": 0.0, "persistence_sie_abs_error_km2": 0.0,
            "days": 0}


def add_ice_edge_metrics(metrics: dict[str, float | int], prediction: np.ndarray,
                         persistence: np.ndarray, target: np.ndarray,
                         valid: np.ndarray, cell_area_km2: np.ndarray,
                         threshold: float = 0.15) -> None:
    """Accumulate daily ice-edge disagreement and absolute SIE error.

    Ice extent uses the conventional 15% SIC threshold. Areas are estimated
    from native-grid cell-area weights and only the shared valid mask is scored.
    """
    weights = cell_area_km2[valid].astype(np.float64, copy=False)
    observed_ice = target[valid] >= threshold
    predicted_ice = prediction[valid] >= threshold
    persisted_ice = persistence[valid] >= threshold
    metrics["model_iiee_km2"] += float(weights[predicted_ice != observed_ice].sum())
    metrics["persistence_iiee_km2"] += float(weights[persisted_ice != observed_ice].sum())
    observed_sie = float(weights[observed_ice].sum())
    metrics["model_sie_abs_error_km2"] += abs(float(weights[predicted_ice].sum()) - observed_sie)
    metrics["persistence_sie_abs_error_km2"] += abs(float(weights[persisted_ice].sum()) - observed_sie)
    metrics["days"] += 1


def summarize_ice_edge(metrics: dict[str, float | int], threshold: float = 0.15) -> dict[str, float | int]:
    days = int(metrics["days"])
    if days <= 0:
        return {"valid_daily_fields": 0, "threshold_sic_fraction": threshold,
                "mean_daily_iiee_million_km2": None,
                "persistence_mean_daily_iiee_million_km2": None,
                "mean_daily_absolute_sie_error_million_km2": None,
                "persistence_mean_daily_absolute_sie_error_million_km2": None}
    divisor = days * 1_000_000.0
    return {"valid_daily_fields": days, "threshold_sic_fraction": threshold,
            "mean_daily_iiee_million_km2": float(metrics["model_iiee_km2"]) / divisor,
            "persistence_mean_daily_iiee_million_km2": float(metrics["persistence_iiee_km2"]) / divisor,
            "mean_daily_absolute_sie_error_million_km2": float(metrics["model_sie_abs_error_km2"]) / divisor,
            "persistence_mean_daily_absolute_sie_error_million_km2": float(metrics["persistence_sie_abs_error_km2"]) / divisor}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Independent evaluation NetCDF directory")
    parser.add_argument("--model", default="models/sic_baseline.json")
    parser.add_argument("--output", help="Optional path for the evaluation report")
    parser.add_argument("--coarsen-factor", type=int, default=1,
                        help="Block-average native pixels by this factor (31 approximates 25 km for 795 m VIIRS pixels)")
    args = parser.parse_args()
    if not 1 <= args.coarsen_factor <= 100:
        parser.error("coarsen-factor must be between 1 and 100")
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    weights = np.asarray(model.get("coefficients"), dtype=np.float64)
    if weights.shape != (5,) or not np.all(np.isfinite(weights)):
        parser.error("Model must contain five finite lag-regression coefficients")
    data_dir = Path(args.data_dir)
    paths = list(data_dir.rglob("*.nc"))
    source_manifest_path = data_dir / "manifest.json"
    source_manifest = None
    if source_manifest_path.exists():
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid source manifest {source_manifest_path}: {exc}")
    try:
        verify_g02202_manifest(data_dir, paths, source_manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    groups = inventory(list(str(path) for path in paths))
    if not groups:
        parser.error("No readable NetCDF inputs found")

    totals = blank_totals()
    by_month: dict[str, dict[str, float | int]] = {}
    by_year: dict[str, dict[str, float | int]] = {}
    is_g02202 = (source_manifest or {}).get("dataset_id") == "G02202" and args.coarsen_factor == 1
    area_totals = {"model_sq": 0.0, "baseline_sq": 0.0, "abs": 0.0, "weight_sum": 0.0}
    area_by_month: dict[str, dict[str, float]] = {}
    area_by_year: dict[str, dict[str, float]] = {}
    area_by_latitude: dict[str, dict[str, float]] = {}
    ice_edge_totals = blank_ice_edge_totals()
    ice_edge_by_month: dict[str, dict[str, float | int]] = {}
    ice_edge_by_year: dict[str, dict[str, float | int]] = {}
    area_weights = latitude = None
    if is_g02202:
        try:
            area_weights, latitude = g02202_cell_geometry(data_dir, (332, 316))
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"Cannot derive G02202 area-weighted metrics: {exc}")
    first = last = None
    older = previous = None
    for day, current, ocean_mask in grids(groups):
        if args.coarsen_factor > 1:
            if ocean_mask is not None:
                current = np.where(ocean_mask, current, np.nan)
            current = coarsen_grid(current, args.coarsen_factor)
            ocean_mask = None
        if previous and older and (day - previous[0]).days == 1 and (previous[0] - older[0]).days == 1 and current.shape == previous[1].shape == older[1].shape:
            valid = np.isfinite(current) & np.isfinite(previous[1]) & np.isfinite(older[1])
            if ocean_mask is not None:
                valid &= ocean_mask
            prediction = np.clip(np.einsum("...k,k->...", features(previous[1], older[1], day), weights), 0, 1)
            diff = prediction[valid] - current[valid]
            persist_diff = previous[1][valid] - current[valid]
            bucket = by_month.setdefault(f"{day.month:02d}", blank_totals())
            year_bucket = by_year.setdefault(str(day.year), blank_totals())
            for metrics in (totals, bucket, year_bucket):
                add_deltas(metrics, diff, persist_diff)
            if area_weights is not None and latitude is not None:
                cell_weights = area_weights[valid]
                add_weighted_deltas(area_totals, diff, persist_diff, cell_weights)
                add_weighted_deltas(area_by_month.setdefault(f"{day.month:02d}", {"model_sq": 0.0, "baseline_sq": 0.0, "abs": 0.0, "weight_sum": 0.0}), diff, persist_diff, cell_weights)
                add_weighted_deltas(area_by_year.setdefault(str(day.year), {"model_sq": 0.0, "baseline_sq": 0.0, "abs": 0.0, "weight_sum": 0.0}), diff, persist_diff, cell_weights)
                add_ice_edge_metrics(ice_edge_totals, prediction, previous[1], current, valid, area_weights)
                add_ice_edge_metrics(ice_edge_by_month.setdefault(f"{day.month:02d}", blank_ice_edge_totals()),
                                     prediction, previous[1], current, valid, area_weights)
                add_ice_edge_metrics(ice_edge_by_year.setdefault(str(day.year), blank_ice_edge_totals()),
                                     prediction, previous[1], current, valid, area_weights)
                band_names = ("south_of_80S", "80S_to_70S", "70S_to_60S", "north_of_60S")
                band_masks = (latitude <= -80, (latitude > -80) & (latitude <= -70),
                              (latitude > -70) & (latitude <= -60), latitude > -60)
                for name, band in zip(band_names, band_masks):
                    band_valid = valid & band
                    if band_valid.any():
                        add_weighted_deltas(area_by_latitude.setdefault(name, {"model_sq": 0.0, "baseline_sq": 0.0, "abs": 0.0, "weight_sum": 0.0}),
                                            prediction[band_valid] - current[band_valid],
                                            previous[1][band_valid] - current[band_valid], area_weights[band_valid])
            first = first or day
            last = day
        if previous is not None and (day - previous[0]).days != 1:
            older = None
        else:
            older = previous
        previous = (day, current)
    if not totals["count"]:
        parser.error("No contiguous valid target/input triples found")

    def summarize(metrics: dict[str, float | int]) -> dict[str, float | int]:
        count = int(metrics["count"])
        return {"grid_cell_samples": count, "mae": float(metrics["abs"]) / count,
                "rmse": math.sqrt(float(metrics["model_sq"]) / count),
                "persistence_rmse": math.sqrt(float(metrics["baseline_sq"]) / count)}
    def summarize_area(metrics: dict[str, float]) -> dict[str, float]:
        weight = float(metrics["weight_sum"])
        return {"evaluated_area_km2_samples": weight,
                "mae": float(metrics["abs"]) / weight,
                "rmse": math.sqrt(float(metrics["model_sq"]) / weight),
                "persistence_rmse": math.sqrt(float(metrics["baseline_sq"]) / weight)}

    is_g02202_holdout = ((source_manifest or {}).get("dataset_id") == "G02202"
                         and first is not None and model.get("training", {}).get("selected_period_end") is not None
                         and first > dt.date.fromisoformat(model["training"]["selected_period_end"]))
    is_2025_sensor_era = (is_g02202 and first is not None and first >= dt.date(2025, 1, 1)
                          and model.get("training", {}).get("selected_period_end") is not None
                          and dt.date.fromisoformat(model["training"]["selected_period_end"]) < dt.date(2025, 1, 1))
    if is_2025_sensor_era:
        warning = ("Separate 2025 sensor-era transfer diagnostic: same G02202 product family, but NSIDC documents AMSR2 input beginning 2025-01-01. "
                   "Do not pool with the pre-2025 holdout or use for tuning. Area weights are projection-scale estimates. Research only; not navigation guidance.")
    elif is_g02202_holdout:
        warning = ("Same-product post-training chronological holdout. Cell-pooled scores are not area-weighted; a separate projection-scale area estimate is included. "
                   "Independent research evaluation only; not navigation guidance.")
    else:
        warning = "Independent research evaluation only; product, period and grid may differ from training. Not navigation guidance."
    report = {"model_id": model.get("model_id"), "model_fingerprint": model_fingerprint(model),
              "training_scope": model.get("training"),
              "evaluation_data_dir": str(Path(args.data_dir)),
              "spatial_evaluation": {"coarsen_factor": args.coarsen_factor,
                                     "method": "native grid" if args.coarsen_factor == 1 else "simple finite-value block mean; not a formal reprojection",
                                     "nominal_pixel_spacing_m": ((source_manifest or {}).get("selection", {}).get("nominal_grid_spacing_m", 0) * args.coarsen_factor) or None},
              "source_manifest": source_manifest,
              "period": {"first_target_date": first.isoformat(), "last_target_date": last.isoformat()},
              "overall": summarize(totals), "by_target_month": {month: summarize(metrics) for month, metrics in sorted(by_month.items())},
              "by_target_year": {year: summarize(metrics) for year, metrics in sorted(by_year.items())},
              "metric_weighting": "Cell-pooled metrics give each finite valid grid cell equal weight; G02202 reports separate estimated ground-area-weighted metrics.",
              "warning": warning}
    if area_weights is not None:
        report["area_weighted"] = {"method": "projected pixel area divided by PROJ ellipsoidal areal_scale at each pixel centre",
                                   "area_unit": "km2", "overall": summarize_area(area_totals),
                                   "by_target_month": {k: summarize_area(v) for k, v in sorted(area_by_month.items())},
                                   "by_target_year": {k: summarize_area(v) for k, v in sorted(area_by_year.items())},
                                   "by_latitude_band": {k: summarize_area(v) for k, v in sorted(area_by_latitude.items())},
                                   "ice_edge": {"aggregation": "mean of per-day pan-domain errors over shared valid cells",
                                                "overall": summarize_ice_edge(ice_edge_totals),
                                                "by_target_month": {k: summarize_ice_edge(v) for k, v in sorted(ice_edge_by_month.items())},
                                                "by_target_year": {k: summarize_ice_edge(v) for k, v in sorted(ice_edge_by_year.items())}}}
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
