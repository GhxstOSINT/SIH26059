"""Frozen-route, day-blocked one-day SIC transfer evaluation against persistence.

The routes are research sampling transects, never navigational alternatives.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Geod, Transformer

try:
    from .train_sic import features
except ImportError:
    from train_sic import features


def route_points(coordinates: list[list[float]], spacing_km: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(coordinates, list) or not 2 <= len(coordinates) <= 20:
        raise ValueError("Each corridor needs 2–20 WGS84 vertices")
    geod = Geod(ellps="WGS84")
    lons, lats = [], []
    total = 0.0
    for index, (start, end) in enumerate(zip(coordinates, coordinates[1:])):
        if any(not math.isfinite(v) for v in (*start, *end)) or any(not -90 <= p[1] <= -30 for p in (start, end)):
            raise ValueError("Route vertices must be finite Antarctic WGS84 coordinates")
        azimuth, _, length = geod.inv(*start, *end)
        total += length
        if length <= 0 or total > 1_000_000:
            raise ValueError("Research corridor must be positive and at most 1,000 km")
        steps = max(1, math.ceil(length / (spacing_km * 1000)))
        if index == 0:
            lons.append(start[0]); lats.append(start[1])
        for step in range(1, steps + 1):
            lon, lat, _ = geod.fwd(*start, azimuth, length * step / steps)
            lons.append(lon); lats.append(lat)
    return np.asarray(lons), np.asarray(lats)


def validate_routes(payload: dict) -> list[dict]:
    routes = payload.get("routes")
    if payload.get("schema_version") != 1 or payload.get("scope") != "research_only" or not isinstance(routes, list) or not routes:
        raise ValueError("A frozen research_only route file is required")
    ids = set()
    for route in routes:
        route_id = route.get("id")
        if not isinstance(route_id, str) or not route_id or route_id in ids:
            raise ValueError("Route IDs must be unique nonempty strings")
        ids.add(route_id)
        route_points(route.get("coordinates"))
    return routes


def grid_indices(lons: np.ndarray, lats: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    east, north = Transformer.from_crs("EPSG:4326", "EPSG:3976", always_xy=True).transform(lons, lats)
    dx, dy = float(np.median(np.diff(x))), float(np.median(np.diff(y)))
    if dx <= 0 or dy >= 0 or not np.allclose(np.diff(x), dx, rtol=.01) or not np.allclose(np.diff(y), dy, rtol=.01):
        raise ValueError("VIIRS grid is not a supported rectilinear EPSG:3976 grid")
    cols = np.rint((east - x[0]) / dx).astype(int)
    rows = np.rint((north - y[0]) / dy).astype(int)
    inside = (rows >= 0) & (rows < len(y)) & (cols >= 0) & (cols < len(x))
    return rows, cols, inside


def score_day(model: np.ndarray, baseline: np.ndarray, observed: np.ndarray,
              rows: np.ndarray, cols: np.ndarray, inside: np.ndarray) -> dict:
    count = int(len(rows))
    valid = inside.copy()
    valid[inside] = (np.isfinite(model[rows[inside], cols[inside]])
                     & np.isfinite(baseline[rows[inside], cols[inside]])
                     & np.isfinite(observed[rows[inside], cols[inside]]))
    n = int(np.count_nonzero(valid))
    if n == 0:
        return {"sample_points": count, "common_valid_points": 0, "coverage_fraction": 0.0}
    r, c = rows[valid], cols[valid]
    m, b, o = model[r, c], baseline[r, c], observed[r, c]
    return {"sample_points": count, "common_valid_points": n, "coverage_fraction": n / count,
            "model_abs_error_sum": float(np.abs(m - o).sum()),
            "baseline_abs_error_sum": float(np.abs(b - o).sum()),
            "model_sq_error_sum": float(np.square(m - o).sum()),
            "baseline_sq_error_sum": float(np.square(b - o).sum()),
            "model_ice_edge_errors": int(np.count_nonzero((m >= .15) != (o >= .15))),
            "baseline_ice_edge_errors": int(np.count_nonzero((b >= .15) != (o >= .15)))}


def aggregate(days: list[dict]) -> dict:
    scored = [day for day in days if day["common_valid_points"] > 0]
    n = sum(day["common_valid_points"] for day in scored)
    points = sum(day["sample_points"] for day in days)
    if not n:
        return {"scored_days": 0, "coverage_fraction": 0.0}
    return {"scored_days": len(scored), "common_valid_points": n,
            "coverage_fraction": n / points,
            "model_mae": sum(day["model_abs_error_sum"] for day in scored) / n,
            "persistence_mae": sum(day["baseline_abs_error_sum"] for day in scored) / n,
            "model_rmse": math.sqrt(sum(day["model_sq_error_sum"] for day in scored) / n),
            "persistence_rmse": math.sqrt(sum(day["baseline_sq_error_sum"] for day in scored) / n),
            "model_ice_edge_error_fraction": sum(day["model_ice_edge_errors"] for day in scored) / n,
            "persistence_ice_edge_error_fraction": sum(day["baseline_ice_edge_errors"] for day in scored) / n}


def evaluate(data_dir: Path, model_path: Path, routes_path: Path, *, min_days: int = 10,
             min_coverage: float = .5) -> dict:
    if min_days < 1 or not 0 < min_coverage <= 1:
        raise ValueError("Invalid evaluation gates")
    route_bytes = routes_path.read_bytes()
    routes = validate_routes(json.loads(route_bytes))
    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)
    weights = np.asarray(model.get("coefficients"), dtype=np.float64)
    if weights.shape != (5,) or not np.all(np.isfinite(weights)):
        raise ValueError("Model lacks five finite coefficients")
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("complete") is not True or not manifest.get("files"):
        raise ValueError("Evaluation requires a complete manifest")
    # Archive integrity is checked once before any score is calculated.
    for item in manifest["files"]:
        name = item.get("file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".nc"):
            raise ValueError("Unsafe evaluation file inventory")
        path = data_dir / name
        if path.is_symlink() or not path.resolve().is_relative_to(data_dir.resolve()):
            raise ValueError("Evaluation file resolves outside its archive")
        if not path.is_file() or path.stat().st_size != item.get("bytes"):
            raise ValueError(f"Missing or altered evaluation file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError(f"Checksum mismatch: {name}")
    day_fields = {}
    x_ref = y_ref = None
    for item in manifest["files"]:
        with xr.open_dataset(data_dir / item["file"]) as ds:
            x, y = np.asarray(ds.cols.values), np.asarray(ds.rows.values)
            if x_ref is not None and (not np.array_equal(x, x_ref) or not np.array_equal(y, y_ref)):
                raise ValueError("Evaluation chunks do not share one grid")
            x_ref, y_ref = x, y
            for index, value in enumerate(ds.time.values):
                day = dt.date.fromisoformat(np.datetime_as_string(value, unit="D"))
                if day in day_fields:
                    raise ValueError("Duplicate evaluation date")
                field = np.asarray(ds.IceConc.isel(time=index).squeeze().values, dtype=np.float32)
                if field.shape != (len(y), len(x)):
                    raise ValueError("Evaluation field/grid shape mismatch")
                field[(~np.isfinite(field)) | (field < 0) | (field > 1)] = np.nan
                day_fields[day] = field
    if x_ref is None:
        raise ValueError("No evaluation scenes")
    route_grids = {}
    for route in routes:
        lon, lat = route_points(route["coordinates"])
        route_grids[route["id"]] = grid_indices(lon, lat, x_ref, y_ref)
    scores = {route["id"]: [] for route in routes}
    for day in sorted(day_fields):
        previous, older = day - dt.timedelta(days=1), day - dt.timedelta(days=2)
        if previous not in day_fields or older not in day_fields:
            continue
        baseline = day_fields[previous]
        prediction = np.clip(np.einsum("...k,k->...", features(baseline, day_fields[older], day), weights), 0, 1)
        prediction[~np.isfinite(baseline) | ~np.isfinite(day_fields[older])] = np.nan
        for route_id, (rows, cols, inside) in route_grids.items():
            scores[route_id].append({"date": day.isoformat(), **score_day(prediction, baseline, day_fields[day], rows, cols, inside)})
    summaries = {route_id: aggregate(days) for route_id, days in scores.items()}
    passed = all(summary["scored_days"] >= min_days and summary["coverage_fraction"] >= min_coverage
                 for summary in summaries.values())
    return {"schema_version": 1, "scope": "research_only", "quality_gate": "passed" if passed else "insufficient_evidence",
            "operational_decision_ready": False, "horizon_days": 1, "comparator": "previous-day persistence",
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "routes_sha256": hashlib.sha256(route_bytes).hexdigest(),
            "dataset_id": manifest.get("dataset_id"),
            "thresholds": {"min_days": min_days, "min_coverage": min_coverage},
            "routes": [{"id": route["id"], "summary": summaries[route["id"]], "days": scores[route["id"]]}
                       for route in routes],
            "limitations": "Fixed research transects only; adjacent samples are correlated. No independent vessel outcomes, land/iceberg hazards, uncertainty calibration, or safety authorization."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-days", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=.5)
    args = parser.parse_args()
    report = evaluate(args.data_dir, args.model, args.routes, min_days=args.min_days,
                      min_coverage=args.min_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"quality_gate": report["quality_gate"], "output": str(args.output),
                      "routes": [{"id": route["id"], **route["summary"]} for route in report["routes"]]}, indent=2))
    return 0 if report["quality_gate"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
