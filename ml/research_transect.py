"""Frozen geodesic research transect sampling; never a vessel route planner."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from ml.evaluate_route_corridors import route_points
from ml.fetch_osi_sic_observation import load_pilot_region
from ml.inventory_copernicus_sic import inventory as inventory_observation


DEFAULT_ROUTE = Path(__file__).resolve().parents[1] / "config" / "research-transect-weddell-ice-gradient.json"


def load_research_transect(path: Path = DEFAULT_ROUTE) -> tuple[dict, str]:
    raw = path.read_bytes()
    record = json.loads(raw)
    points = record.get("coordinates_lon_lat")
    if (record.get("schema_version") != 1 or record.get("scope") != "research_only"
            or record.get("navigation_clearance") is not False
            or record.get("coordinate_reference_system") != "EPSG:4326"
            or not isinstance(points, list) or len(points) < 2
            or record.get("sampling_interval_km") != 2.0):
        raise ValueError("Invalid frozen research transect")
    frozen = datetime.fromisoformat(record["frozen_at_utc"])
    start = date.fromisoformat(record["eligible_bulletins_from"])
    if frozen.tzinfo is None or start <= frozen.date():
        raise ValueError("Research transect must begin with a post-freeze bulletin")
    region_id, (west, east, south, north), _ = load_pilot_region()
    if region_id != "peninsula-weddell":
        raise ValueError("Research transect pilot region changed")
    for point in points:
        if (not isinstance(point, list) or len(point) != 2
                or not all(isinstance(value, (int, float)) for value in point)
                or not west < point[0] < east or not south < point[1] < north):
            raise ValueError("Research transect vertex lies outside the frozen study area")
    route_points(points, spacing_km=2.0)  # also enforces a bounded length
    return record, hashlib.sha256(raw).hexdigest()


def sampled_cells(latitudes: np.ndarray, longitudes: np.ndarray,
                  route: dict) -> tuple[np.ndarray, np.ndarray, int]:
    latitude = np.asarray(latitudes, dtype=np.float64)
    longitude = np.asarray(longitudes, dtype=np.float64)
    if (latitude.ndim != 1 or longitude.ndim != 1 or len(latitude) < 2 or len(longitude) < 2
            or not np.all(np.diff(latitude) > 0) or not np.all(np.diff(longitude) > 0)):
        raise ValueError("Expected ascending geographic observation coordinates")
    lons, lats = route_points(route["coordinates_lon_lat"], spacing_km=route["sampling_interval_km"])
    if (np.any(lats < latitude[0]) or np.any(lats > latitude[-1])
            or np.any(lons < longitude[0]) or np.any(lons > longitude[-1])):
        raise ValueError("Research transect extends outside the observation grid")

    def nearest(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        right = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
        left = right - 1
        return np.where(np.abs(values - grid[left]) <= np.abs(values - grid[right]), left, right)

    rows, cols = nearest(latitude, lats), nearest(longitude, lons)
    unique = list(dict.fromkeys(zip(rows.tolist(), cols.tolist())))
    return (np.array([row for row, _ in unique], dtype=np.int64),
            np.array([col for _, col in unique], dtype=np.int64), len(lons))


def sample_transect(predicted: np.ndarray | None, observed: np.ndarray,
                    retrieval_uncertainty: np.ndarray, status_flags: np.ndarray,
                    latitudes: np.ndarray, longitudes: np.ndarray,
                    route: dict, route_sha256: str,
                    *, bulletin_date: date | None = None,
                    persistence: np.ndarray | None = None,
                    sample_arrays: dict[str, np.ndarray] | None = None) -> dict:
    rows, cols, raw_points = sampled_cells(latitudes, longitudes, route)
    truth = np.asarray(observed, dtype=np.float64)
    uncertainty = np.asarray(retrieval_uncertainty, dtype=np.float64)
    flags = np.asarray(status_flags)
    shape = (len(latitudes), len(longitudes))
    if any(value.shape != shape for value in (truth, uncertainty, flags)):
        raise ValueError("Research transect grids must share the observation shape")
    if predicted is not None and np.asarray(predicted).shape != shape:
        raise ValueError("Forecast grid must match the observation shape")
    if persistence is not None and np.asarray(persistence).shape != shape:
        raise ValueError("Persistence grid must match the observation shape")
    valid = (np.isfinite(truth[rows, cols]) & (truth[rows, cols] >= 0) & (truth[rows, cols] <= 1)
             & np.isfinite(uncertainty[rows, cols])
             & (uncertainty[rows, cols] >= 0) & (uncertainty[rows, cols] <= 1)
             & np.isfinite(flags[rows, cols]) & (flags[rows, cols] == 0))
    result = {
        "scope": "frozen_research_ice_gradient_transect",
        "route_id": route["id"],
        "route_config_sha256": route_sha256,
        "eligible_bulletins_from": route["eligible_bulletins_from"],
        "sample_points_at_2km": raw_points,
        "unique_observation_cells": len(rows),
        "nominal_observation_cells": int(valid.sum()),
        "nominal_cell_coverage_fraction": float(valid.mean()),
        "navigation_clearance": False,
    }
    if predicted is None:
        result["evaluation_status"] = "observation_selection_check_only"
        if valid.any():
            result["observed_sic_range_fraction"] = [float(np.min(truth[rows[valid], cols[valid]])),
                                                     float(np.max(truth[rows[valid], cols[valid]]))]
        return result
    if bulletin_date is None or bulletin_date < date.fromisoformat(route["eligible_bulletins_from"]):
        result["evaluation_status"] = "not_eligible_pre_freeze"
        return result
    prediction = np.asarray(predicted, dtype=np.float64)[rows, cols]
    valid &= np.isfinite(prediction) & (prediction >= 0) & (prediction <= 1)
    baseline = np.asarray(persistence, dtype=np.float64)[rows, cols] if persistence is not None else None
    if baseline is not None:
        valid &= np.isfinite(baseline) & (baseline >= 0) & (baseline <= 1)
    result["common_valid_cells"] = int(valid.sum())
    result["common_valid_fraction"] = float(valid.mean())
    if not valid.any():
        result["evaluation_status"] = "insufficient_coverage"
        return result
    delta = prediction[valid] - truth[rows[valid], cols[valid]]
    if sample_arrays is not None:
        sample_arrays.update({
            "transect_forecast_sic_fraction": prediction[valid].astype(np.float32),
            "transect_observed_sic_fraction": truth[rows[valid], cols[valid]].astype(np.float32),
            "transect_observation_uncertainty_fraction": uncertainty[rows[valid], cols[valid]].astype(np.float32),
        })
        if baseline is not None:
            sample_arrays["transect_persistence_sic_fraction"] = baseline[valid].astype(np.float32)
    result.update({
        "evaluation_status": "single_bulletin_research_sample",
        "forecast_mae_fraction": float(np.mean(np.abs(delta))),
        "forecast_rmse_fraction": float(np.sqrt(np.mean(delta ** 2))),
        "forecast_bias_fraction": float(np.mean(delta)),
        "observed_mean_sic_fraction": float(np.mean(truth[rows[valid], cols[valid]])),
        "forecast_mean_sic_fraction": float(np.mean(prediction[valid])),
        "ice_edge_disagreement_fraction": float(np.mean(
            (prediction[valid] >= 0.15) != (truth[rows[valid], cols[valid]] >= 0.15))),
        "persistence_mae_fraction": (float(np.mean(np.abs(baseline[valid] - truth[rows[valid], cols[valid]])))
                                     if baseline is not None else None),
        "persistence_mean_sic_fraction": (float(np.mean(baseline[valid])) if baseline is not None else None),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route, route_sha = load_research_transect(args.route)
    day = date.fromisoformat(route["selection_observation_date"])
    source = inventory_observation(args.observation, day, day)
    if source["source_sha256"] != route["selection_observation_sha256"]:
        raise ValueError("Selection observation does not match the frozen route provenance")
    with xr.open_dataset(args.observation) as dataset:
        report = sample_transect(
            None, np.asarray(dataset.ice_conc.values[0], dtype=np.float32) / 100,
            np.asarray(dataset.total_uncertainty.values[0], dtype=np.float32) / 100,
            np.asarray(dataset.status_flag.values[0]), dataset.latitude.values, dataset.longitude.values,
            route, route_sha)
    report["selection_observation_sha256"] = source["source_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
