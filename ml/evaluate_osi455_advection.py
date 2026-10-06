"""Research-only OSI-455 one-day SIC advection diagnostic against persistence."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


def read_sic(ds: xr.Dataset, day: date) -> tuple[np.ndarray, np.ndarray]:
    index = int(np.where(ds.time.values.astype("datetime64[D]") == np.datetime64(day))[0][0])
    values = np.asarray(ds["cdr_seaice_conc"].isel(time=index).values, dtype=np.float32)
    qa = np.asarray(ds["cdr_seaice_conc_qa_flag"].isel(time=index).values, dtype=np.uint8)
    temporal = np.asarray(ds["cdr_seaice_conc_interp_temporal_flag"].isel(time=index).values, dtype=np.uint8)
    # Exclude no-input, invalid-mask, spatial interpolation, and all temporal interpolation.
    valid = np.isfinite(values) & (values >= 0) & (values <= 1) & ((qa & (8 | 16 | 32 | 64)) == 0) & (temporal == 0)
    values[~valid] = np.nan
    return values, valid


def to_osi_grid(values: np.ndarray, valid: np.ndarray, source: xr.Dataset,
                target_x: np.ndarray, target_y: np.ndarray, target_crs: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise RuntimeError(
            "OSI-455 reprojection requires the optional dependencies in requirements-motion-research.txt"
        ) from exc
    sx = source.x.values
    sy = source.y.values
    src_transform = from_origin(float(sx[0] - (sx[1] - sx[0]) / 2),
                                float(sy[0] - (sy[1] - sy[0]) / 2),
                                float(sx[1] - sx[0]), abs(float(sy[1] - sy[0])))
    dx = float(target_x[1] - target_x[0]) * 1000
    dy = float(target_y[1] - target_y[0]) * 1000
    dst_transform = from_origin(float(target_x[0] * 1000 - dx / 2),
                                float(target_y[0] * 1000 - dy / 2), dx, abs(dy))
    dst_shape = (len(target_y), len(target_x))
    numerator = np.zeros(dst_shape, np.float32)
    coverage = np.zeros(dst_shape, np.float32)
    reproject(np.where(valid, values, 0).astype(np.float32), numerator,
              src_transform=src_transform, src_crs="EPSG:3412", src_nodata=None,
              dst_transform=dst_transform, dst_crs=target_crs, dst_nodata=0,
              resampling=Resampling.average)
    reproject(valid.astype(np.float32), coverage,
              src_transform=src_transform, src_crs="EPSG:3412", src_nodata=None,
              dst_transform=dst_transform, dst_crs=target_crs, dst_nodata=0,
              resampling=Resampling.average)
    good = coverage >= 0.75
    out = np.full(dst_shape, np.nan, np.float32)
    out[good] = numerator[good] / coverage[good]
    return out, good


def day_metrics(errors: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    persistence = np.concatenate([p[m].ravel() for p, _, m in errors])
    advection = np.concatenate([a[m].ravel() for _, a, m in errors])
    return {"n_cells": int(persistence.size),
            "persistence_mae": float(np.mean(np.abs(persistence))),
            "advection_mae": float(np.mean(np.abs(advection))),
            "persistence_rmse": float(np.sqrt(np.mean(persistence ** 2))),
            "advection_rmse": float(np.sqrt(np.mean(advection ** 2)))}


def geographic_group_masks(longitude: np.ndarray, latitude: np.ndarray) -> dict[str, np.ndarray]:
    """Broad, non-route-specific Antarctic longitude sectors and latitude bands."""
    lon, lat = np.broadcast_arrays(np.asarray(longitude, dtype=np.float64),
                                   np.asarray(latitude, dtype=np.float64))
    finite = np.isfinite(lon) & np.isfinite(lat)
    lon_360 = np.mod(lon, 360.0)
    groups = {
        "longitude_000_090E": finite & (lon_360 >= 0) & (lon_360 < 90),
        "longitude_090_180E": finite & (lon_360 >= 90) & (lon_360 < 180),
        "longitude_180_270E": finite & (lon_360 >= 180) & (lon_360 < 270),
        "longitude_270_360E": finite & (lon_360 >= 270) & (lon_360 < 360),
        "latitude_50S_60S": finite & (lat <= -50) & (lat > -60),
        "latitude_60S_70S": finite & (lat <= -60) & (lat > -70),
        "latitude_70S_80S": finite & (lat <= -70) & (lat > -80),
        "latitude_80S_90S": finite & (lat <= -80) & (lat >= -90),
    }
    return groups


def stratified_metrics(entries: list[tuple[date, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                       coefficient: float, groups: dict[str, np.ndarray]) -> dict[str, dict]:
    """Score on strict common-valid cells within broad geographic strata."""
    report = {}
    for name, group in groups.items():
        rows = []
        for _, base, truth, advected, valid in entries:
            mask = valid & group
            if not np.any(mask):
                continue
            prediction = np.clip(base + coefficient * (advected - base), 0, 1)
            rows.append((truth - base, truth - prediction, mask))
        if not rows:
            report[name] = {"n_cells": 0, "evaluated_days": 0,
                            "persistence_mae": None, "advection_mae": None,
                            "mae_reduction_percent": None, "persistence_rmse": None,
                            "advection_rmse": None, "rmse_reduction_percent": None}
            continue
        metrics = day_metrics(rows)
        metrics["evaluated_days"] = len(rows)
        metrics["mae_reduction_percent"] = 100 * (1 - metrics["advection_mae"] / metrics["persistence_mae"])
        metrics["rmse_reduction_percent"] = 100 * (1 - metrics["advection_rmse"] / metrics["persistence_rmse"])
        report[name] = metrics
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backtrace_coordinates(dx_km: np.ndarray, dy_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return source array coordinates for displaced targets on x-right/y-up grids."""
    rows, cols = np.indices(dx_km.shape, dtype=np.float32)
    return rows + dy_km / 75.0, cols - dx_km / 75.0


def latest_causal_motion_date(issue_date: date) -> date:
    """OSI files end at 12Z on their label date; SIC issue is 00Z next date."""
    return issue_date - timedelta(days=1)


def manifest_covers_experiment(manifest: dict, start: date, end: date) -> bool:
    """Allow a verified contiguous archive to serve multiple smaller hindcasts."""
    period = manifest.get("requested_period", {})
    try:
        archive_start = date.fromisoformat(period["start"])
        archive_end = date.fromisoformat(period["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(manifest.get("complete") is True
                and archive_start <= start - timedelta(days=1)
                and archive_end >= end - timedelta(days=1))


def main() -> None:
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise RuntimeError(
            "OSI-455 advection requires the optional dependencies in requirements-motion-research.txt"
        ) from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--sic-file", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 9, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2020, 9, 30))
    parser.add_argument("--fixed-lambda", type=float,
                        help="Evaluate a previously selected blend coefficient without refitting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.motion_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_motion_start = args.start - timedelta(days=1)
    if not manifest_covers_experiment(manifest, args.start, args.end):
        raise ValueError("OSI-455 manifest is incomplete or does not cover the required causal motion dates")
    recorded_motion_hashes = {entry["file"]: entry["sha256"] for entry in manifest["files"]}
    sic_ds = xr.open_dataset(args.sic_file)
    sic_dates = sic_ds.time.values.astype("datetime64[D]")
    required_sic_end = np.datetime64(args.end + timedelta(days=1))
    if not np.any(sic_dates == np.datetime64(args.start)) or not np.any(sic_dates == required_sic_end):
        sic_ds.close()
        raise ValueError(f"SIC file must contain both issue date {args.start} and next-day target date {required_sic_end}")
    lambda_grid = np.arange(0, 1.501, 0.1)
    train: list[tuple[date, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    test: list[tuple[date, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    coverage_days: list[dict] = []
    geographic_groups = None
    reference_lon = reference_lat = None
    sample_days = (args.end - args.start).days + 1
    train_days = sample_days // 2 + 1
    train_end = args.start + timedelta(days=train_days - 1)
    for n in range(sample_days):
        issue = args.start + timedelta(days=n)
        target = issue + timedelta(days=1)
        # A file dated D spans D-1 12Z through D 12Z, which leaks beyond a D 00Z
        # SIC issue time. Use the latest completed vector interval ending by D-1 12Z.
        motion_day = latest_causal_motion_date(issue)
        motion_file = args.motion_dir / f"ice_drift_sh_ease2-750_cdr-v1p0_24h-{motion_day:%Y%m%d}1200.nc"
        if not motion_file.exists():
            raise FileNotFoundError(motion_file)
        if recorded_motion_hashes.get(motion_file.name) != sha256_file(motion_file):
            raise ValueError(f"OSI-455 file is absent from its manifest or fails its SHA-256 check: {motion_file.name}")
        motion_ds = xr.open_dataset(motion_file)
        longitude = np.asarray(motion_ds["lon"].values, dtype=np.float64)
        latitude = np.asarray(motion_ds["lat"].values, dtype=np.float64)
        if geographic_groups is None:
            reference_lon, reference_lat = longitude.copy(), latitude.copy()
            geographic_groups = geographic_group_masks(longitude, latitude)
        elif (longitude.shape != reference_lon.shape or latitude.shape != reference_lat.shape
              or not np.allclose(longitude, reference_lon, equal_nan=True)
              or not np.allclose(latitude, reference_lat, equal_nan=True)):
            raise ValueError(f"OSI-455 geolocation grid changed unexpectedly in {motion_file.name}")
        base, base_valid = read_sic(sic_ds, issue)
        truth, truth_valid = read_sic(sic_ds, target)
        grid_crs = motion_ds["Lambert_Azimuthal_Equal_Area"].attrs["proj4_string"]
        regrid_base, base_grid_valid = to_osi_grid(base, base_valid, sic_ds,
                                                    motion_ds.xc.values, motion_ds.yc.values, grid_crs)
        regrid_truth, truth_grid_valid = to_osi_grid(truth, truth_valid, sic_ds,
                                                      motion_ds.xc.values, motion_ds.yc.values, grid_crs)
        status = np.asarray(motion_ds.status_flag.isel(time=0).values)
        dx = np.asarray(motion_ds.dX.isel(time=0).values, dtype=np.float32)
        dy = np.asarray(motion_ds.dY.isel(time=0).values, dtype=np.float32)
        finite_vectors = np.isfinite(dx) & np.isfinite(dy)
        quality = np.isin(status, [21, 30]) & finite_vectors
        # OSI y coordinates descend. Backtrace the displacement to sample issue-day SIC.
        valid = truth_grid_valid & base_grid_valid & quality & np.isfinite(regrid_truth) & np.isfinite(regrid_base)
        sample = backtrace_coordinates(dx, dy)
        adv = map_coordinates(np.nan_to_num(regrid_base, nan=0), sample, order=1,
                              mode="constant", cval=0, prefilter=False)
        weight = map_coordinates(base_grid_valid.astype(np.float32), sample, order=1,
                                 mode="constant", cval=0, prefilter=False)
        adv[weight < 0.75] = np.nan
        valid &= np.isfinite(adv)
        coverage_days.append({
            "issue_date": issue.isoformat(),
            "status_21_cells": int(((status == 21) & finite_vectors).sum()),
            "status_30_cells": int(((status == 30) & finite_vectors).sum()),
            "strict_motion_cells": int(quality.sum()),
            "common_valid_forecast_cells": int(valid.sum()),
            "grid_cells": int(status.size),
        })
        entry = (issue, regrid_base, regrid_truth, adv.astype(np.float32), valid)
        if args.fixed_lambda is None:
            (train if issue <= train_end else test).append(entry)
        else:
            test.append(entry)
        motion_ds.close()
    if args.fixed_lambda is None and (not train or not test):
        raise ValueError("Chronological training and holdout samples must both be nonempty")
    if args.fixed_lambda is None and not any(np.any(entry[-1]) for entry in train):
        raise ValueError("No quality-screened OSI-455/G02202 cells are available in the training dates")
    if not any(np.any(entry[-1]) for entry in test):
        raise ValueError("No quality-screened OSI-455/G02202 cells are available in the holdout dates")
    scores = []
    if args.fixed_lambda is None:
        for lam in lambda_grid:
            rows = []
            for _, base, truth, adv, mask in train:
                pred = np.clip(base + lam * (adv - base), 0, 1)
                rows.append((truth - base, truth - pred, mask))
            scores.append(day_metrics(rows)["advection_mae"])
        best_lambda = float(lambda_grid[int(np.argmin(scores))])
    else:
        if not 0 <= args.fixed_lambda <= 1.5:
            raise ValueError("--fixed-lambda must be between 0 and 1.5")
        best_lambda = args.fixed_lambda

    def evaluate(entries):
        all_rows = []
        daily = []
        for issue, base, truth, adv, mask in entries:
            if not np.any(mask):
                continue
            pred = np.clip(base + best_lambda * (adv - base), 0, 1)
            row = (truth - base, truth - pred, mask)
            all_rows.append(row)
            daily.append({"issue_date": issue.isoformat(), **day_metrics([row])})
        return {"pooled": day_metrics(all_rows), "evaluated_days": len(daily), "daily": daily,
                "geographic_strata": stratified_metrics(entries, best_lambda, geographic_groups)}

    result = {
        "experiment": "Lagged OSI-455 one-day SIC advection diagnostic under a date-label timing convention; not an operational forecast",
        "period": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "sic": {"dataset": "NOAA/NSIDC G02202 v6", "file": args.sic_file.name,
                "sha256": sha256_file(args.sic_file),
                "quality": "excluded QA missing/invalid/spatially interpolated flags and every temporal interpolation; regridded by area average; >=75% valid source coverage"},
        "motion": {"dataset": "EUMETSAT OSI-455 CDR", "quality": "status flags 21/30 only; OSI vectors are 24h displacement in km on 75km grid",
                   "timing": "use file date issue_date-1; interval ends 12h before the experiment's assumed 00Z issue label"},
        "provenance": {"motion_manifest": str(manifest_path), "motion_manifest_sha256": sha256_file(manifest_path),
                       "motion_manifest_period": manifest["requested_period"]},
        "motion_coverage": {
            "strict_status_flags": [21, 30],
            "issue_days": len(coverage_days),
            "days_with_strict_motion": sum(day["strict_motion_cells"] > 0 for day in coverage_days),
            "days_with_common_valid_forecast_cells": sum(day["common_valid_forecast_cells"] > 0 for day in coverage_days),
            "status_21_cell_days": sum(day["status_21_cells"] for day in coverage_days),
            "status_30_cell_days": sum(day["status_30_cells"] for day in coverage_days),
            "strict_motion_cell_days": sum(day["strict_motion_cells"] for day in coverage_days),
            "common_valid_forecast_cell_days": sum(day["common_valid_forecast_cells"] for day in coverage_days),
            "possible_grid_cell_days": sum(day["grid_cells"] for day in coverage_days),
            "by_month": [
                {
                    "month": month,
                    "issue_days": sum(day["issue_date"][5:7] == f"{month:02d}" for day in coverage_days),
                    "days_with_strict_motion": sum(day["issue_date"][5:7] == f"{month:02d}" and day["strict_motion_cells"] > 0 for day in coverage_days),
                    "strict_motion_cell_days": sum(day["strict_motion_cells"] for day in coverage_days if day["issue_date"][5:7] == f"{month:02d}"),
                    "common_valid_forecast_cell_days": sum(day["common_valid_forecast_cells"] for day in coverage_days if day["issue_date"][5:7] == f"{month:02d}"),
                    "possible_grid_cell_days": sum(day["grid_cells"] for day in coverage_days if day["issue_date"][5:7] == f"{month:02d}"),
                }
                for month in sorted({int(day["issue_date"][5:7]) for day in coverage_days})
            ],
        },
        "geographic_grouping": {
            "longitude_sectors": "four equal 90-degree bins; longitude normalized to [0, 360)",
            "latitude_bands": "50S-60S, 60S-70S, 70S-80S, 80S-90S",
            "interpretation": "broad grid-level diagnostics only; not route-corridor or vessel-specific validation",
        },
        "protocol": {"issue_time": "Experiment convention: SIC date D is treated as an issue label at 00Z; use OSI file D-1, whose completed 24h motion interval ends 12 hours before that label, avoiding post-label vector leakage. G02202 is a daily product whose midnight coordinate is a date label, not evidence that the daily SIC field was operationally available at that instant.",
                     "split": (f"fit shrinkage lambda on issue dates through {train_end.isoformat()}; evaluate chronologically from {(train_end + timedelta(days=1)).isoformat()} onward"
                               if args.fixed_lambda is None else "evaluate every requested date using a coefficient selected independently before this evaluation period; no fitting on these dates"),
                     "baseline": "persistence; both methods evaluated on identical strict common-valid cells",
                     "motion_advection": "backtrace SIC using the most recent causal 24h displacement; assumes that lagged drift represents the next 24h; blend forecast = persistence + lambda*(advected-persistence)"},
        "selected_lambda": best_lambda,
        "training_mae_by_lambda": {str(round(float(l), 1)): float(s) for l, s in zip(lambda_grid, scores)},
        "train": evaluate(train) if train else None, "holdout": evaluate(test),
        "limitations": ["The retrospective 00Z issue convention prevents post-label OSI-vector leakage, but the G02202 daily product does not establish a single observation or operational availability time; strict real-time causality is not proven.",
                        "Short period(s) and limited season/year coverage; spatially correlated cell samples are not independent trials.",
                        "Retrospective CDR can include interpolated/corrected vectors; only flags 21 and 30 retained here.",
                        "This evaluates gridded SIC changes, not ice-edge safety, iceberg drift, route clearance, or voyage decisions.",
                        "A small apparent gain is not evidence of generalizable forecast skill; multi-year blocked tests and independent products are required."],
    }
    # Save per-day and pooled scores for review; strict cross-product scale/mask matching is not an operational claim.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sic_ds.close()
    print(json.dumps({"selected_lambda": best_lambda,
                      "train": result["train"]["pooled"] if result["train"] else None,
                      "holdout": result["holdout"]["pooled"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
