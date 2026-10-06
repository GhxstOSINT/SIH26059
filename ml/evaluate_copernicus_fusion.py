"""Retrospective, time-held-out SIC experiment on matched Copernicus fields.

This is a research comparison, not a forecast-origin validation. GLORYS and
reprocessed L4 wind may incorporate information unavailable at the time.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


FEATURES = ("sic_t", "sic_change_t", "current_u_t", "current_v_t", "wind_u_t", "wind_v_t")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_ridge(x: np.ndarray, delta: np.ndarray, penalty: float = 10.0) -> dict:
    """Fit a standardised linear change model with unpenalised intercept."""
    if x.ndim != 2 or delta.ndim != 1 or len(x) != len(delta) or len(x) < 20:
        raise ValueError("Insufficient matched training samples")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    regulariser = np.eye(design.shape[1]) * penalty
    regulariser[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + regulariser, design.T @ delta)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficient": coef.tolist(),
            "ridge_penalty": penalty}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    coef = np.asarray(model["coefficient"])
    return np.clip(x[:, 0] + coef[0] + z @ coef[1:], 0, 100)


def scores(target: np.ndarray, estimate: np.ndarray) -> dict:
    error = estimate - target
    return {"mae_percentage_points": float(np.mean(np.abs(error))),
            "rmse_percentage_points": float(np.sqrt(np.mean(error ** 2)))}


def run(sic_path: Path, current_path: Path, wind_path: Path, split: date) -> dict:
    for path in (sic_path, current_path, wind_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular source file: {path}")
    with xr.open_dataset(sic_path) as sic_ds, xr.open_dataset(current_path) as current_ds, xr.open_dataset(wind_path) as wind_ds:
        if sic_ds.ice_conc.attrs.get("units") != "%":
            raise ValueError("SIC must be in percent")
        if current_ds.uo.attrs.get("units") != "m s-1" or wind_ds.eastward_wind.attrs.get("units") != "m s-1":
            raise ValueError("Vector fields must be in m s-1")
        days = np.asarray(sic_ds.time.values).astype("datetime64[D]")
        if len(days) < 30 or np.any(np.diff(days).astype(int) != 1):
            raise ValueError("SIC dates must be daily and continuous")
        target_grid = {"latitude": sic_ds.latitude, "longitude": sic_ds.longitude}
        currents = current_ds[["uo", "vo"]].isel(depth=0, drop=True).interp(target_grid, method="linear")
        currents = currents.reindex(time=sic_ds.time, method="nearest", tolerance=np.timedelta64(12, "h"))
        winds = wind_ds[["eastward_wind", "northward_wind"]].resample(time="1D").mean()
        winds = winds.interp(target_grid, method="linear").reindex(time=sic_ds.time, method="nearest", tolerance=np.timedelta64(12, "h"))
        sic = np.asarray(sic_ds.ice_conc.values, dtype=np.float64)
        u = np.asarray(currents.uo.values, dtype=np.float64)
        v = np.asarray(currents.vo.values, dtype=np.float64)
        wu = np.asarray(winds.eastward_wind.values, dtype=np.float64)
        wv = np.asarray(winds.northward_wind.values, dtype=np.float64)
        if not all(a.shape == sic.shape for a in (u, v, wu, wv)):
            raise ValueError("Aligned fields have unequal dimensions")

        fields = np.stack((sic[1:-1], sic[1:-1] - sic[:-2], u[1:-1], v[1:-1], wu[1:-1], wv[1:-1]), axis=-1)
        target = sic[2:]
        sample_dates = days[2:]
        valid = np.all(np.isfinite(fields), axis=-1) & np.isfinite(target)
        valid &= (fields[..., 0] >= 0) & (fields[..., 0] <= 100) & (target >= 0) & (target <= 100)
        train_days = sample_dates < np.datetime64(split)
        test_days = sample_dates >= np.datetime64(split)
        if train_days.sum() < 15 or test_days.sum() < 10:
            raise ValueError("Need at least 15 train dates and 10 held-out dates")
        train_mask = valid & train_days[:, None, None]
        test_mask = valid & test_days[:, None, None]
        x_train, y_train = fields[train_mask], target[train_mask]
        x_test, y_test = fields[test_mask], target[test_mask]
        if len(x_train) < 100 or len(x_test) < 100:
            raise ValueError("Insufficient fully matched cells; inspect wind coverage")
        sic_only = fit_ridge(x_train[:, :2], y_train - x_train[:, 0])
        full = fit_ridge(x_train, y_train - x_train[:, 0])
        estimates = {
            "persistence": x_test[:, 0],
            "sic_lag_ridge": predict(sic_only, x_test[:, :2]),
            "sic_current_wind_ridge": predict(full, x_test),
        }
        per_day = []
        for day, mask in zip(sample_dates[test_days], valid[test_days]):
            per_day.append({"date": str(day), "matched_cells": int(mask.sum())})
        metrics = {name: scores(y_test, estimate) for name, estimate in estimates.items()}
        return {
            "schema_version": 1,
            "scope": "retrospective_research_experiment",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": {"sea_ice": {"file": str(sic_path), "sha256": sha256(sic_path)},
                        "glorys_reanalysis_currents": {"file": str(current_path), "sha256": sha256(current_path)},
                        "l4_reprocessed_wind": {"file": str(wind_path), "sha256": sha256(wind_path)}},
            "target": "next-day Copernicus sea ice concentration, percent",
            "alignment": "Daily mean hourly wind; bilinear vector interpolation to SIC grid; prior-day features; common finite-cell mask; no extrapolation across missing values.",
            "feature_names": list(FEATURES),
            "train": {"start": str(sample_dates[train_days][0]), "end": str(sample_dates[train_days][-1]),
                      "days": int(train_days.sum()), "matched_cells": int(len(x_train))},
            "test": {"start": str(sample_dates[test_days][0]), "end": str(sample_dates[test_days][-1]),
                     "days": int(test_days.sum()), "matched_cells": int(len(x_test)),
                     "possible_sic_grid_cells": int(test_days.sum() * sic.shape[1] * sic.shape[2]),
                     "daily": per_day},
            "metrics": metrics,
            "models": {"sic_lag_ridge": sic_only, "sic_current_wind_ridge": full},
            "asof_verified": False,
            "operational_eligibility": False,
            "limitations": [
                "GLORYS and reprocessed L4 wind are retrospective, potentially assimilating observations unavailable at the forecast origin.",
                "One seasonal holdout is insufficient to establish generalisation across years, regions, or ice regimes.",
                "Spatial pixels are correlated and are not independent test voyages.",
                "This predicts sea ice concentration, not individual iceberg trajectories or vessel safety.",
                "Do not replace the pinned production candidate or issue navigation guidance from this result.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sic", type=Path, required=True)
    parser.add_argument("--currents", type=Path, required=True)
    parser.add_argument("--wind", type=Path, required=True)
    parser.add_argument("--split", type=date.fromisoformat, default=date(2024, 8, 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.sic, args.currents, args.wind, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "test": result["test"],
                      "metrics": result["metrics"], "operational_eligibility": False}, indent=2))


if __name__ == "__main__":
    main()
