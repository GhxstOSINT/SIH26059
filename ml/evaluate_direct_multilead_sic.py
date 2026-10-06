"""Fit direct-horizon SIC regressions on one archive and score a frozen later archive.

Unlike recursive rollout, each horizon predicts directly from the last two observed
fields. This is an offline research experiment, not an operational forecast service.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    from .evaluate_sic import add_weighted_deltas, g02202_cell_geometry
    from .identity import model_artifact_sha256
    from .train_sic import grids, inventory, verify_g02202_manifest
except ImportError:  # Direct execution as `python ml/evaluate_direct_multilead_sic.py`.
    from evaluate_sic import add_weighted_deltas, g02202_cell_geometry
    from identity import model_artifact_sha256
    from train_sic import grids, inventory, verify_g02202_manifest


REGULARIZATION = np.diag((0.0, 0.02, 0.02, 0.02, 0.02))


def direct_fingerprint(models: dict) -> str:
    payload = {"model_id": "sic-direct-horizon-regression-research-v1",
               "coefficients_by_lead": {key: value["coefficients"] for key, value in sorted(models.items())}}
    blend_weights = {key: value["persistence_blend_alpha"]
                     for key, value in sorted(models.items())
                     if "persistence_blend_alpha" in value}
    if blend_weights:
        payload["persistence_blend_alpha_by_lead"] = blend_weights
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def feature_matrix(latest: np.ndarray, previous: np.ndarray, target_day: dt.date) -> np.ndarray:
    phase = 2 * math.pi * (target_day.timetuple().tm_yday - 1) / 365.2425
    return np.stack((np.ones_like(latest), latest, previous,
                     np.full_like(latest, math.sin(phase)),
                     np.full_like(latest, math.cos(phase))), axis=-1)


def apply_persistence_blend(persistence: np.ndarray, direct_prediction: np.ndarray,
                            alpha: float) -> np.ndarray:
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Persistence-blend alpha must be between zero and one")
    return np.clip(persistence + alpha * (direct_prediction - persistence), 0, 1)


def load_fields(directory: Path):
    paths = sorted(directory.rglob("*.nc"))
    if not paths:
        raise ValueError(f"No NetCDF files in {directory}")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    verify_g02202_manifest(directory, paths, manifest)
    if not manifest or manifest.get("dataset_id") != "G02202":
        raise ValueError(f"Expected a checksummed G02202 archive at {directory}")
    return lambda: grids(inventory([str(p) for p in paths])), manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def fit_models(fields, leads: tuple[int, ...], cutoff: dt.date):
    accum = {lead: (np.zeros((5, 5)), np.zeros(5), 0, None, None) for lead in leads}
    history = collections.deque(maxlen=max(leads) + 2)
    for record in fields:
        day, grid, ocean = record
        history.append(record)
        for lead in leads:
            if len(history) <= lead + 1:
                continue
            origin = history[-lead - 1]
            older = history[-lead - 2] if len(history) > lead + 1 else None
            if older is None or day > cutoff:
                continue
            if (day - origin[0]).days != lead or (origin[0] - older[0]).days != 1:
                continue
            if grid.shape != origin[1].shape or grid.shape != older[1].shape:
                continue
            valid = np.isfinite(grid) & np.isfinite(origin[1]) & np.isfinite(older[1])
            if ocean is not None:
                valid &= ocean
            if not valid.any():
                continue
            x = feature_matrix(origin[1], older[1], day)[valid].astype(np.float64)
            y = grid[valid].astype(np.float64)
            gram, rhs, count, first, last = accum[lead]
            gram += x.T @ x
            rhs += x.T @ y
            accum[lead] = (gram, rhs, count + len(y), first or day, day)
    models = {}
    for lead, (gram, rhs, count, first, last) in accum.items():
        if count < 10_000:
            raise ValueError(f"Insufficient training samples for {lead}-day model: {count}")
        models[str(lead)] = {
            "lead_days": lead,
            "coefficients": np.linalg.solve(gram + REGULARIZATION, rhs).tolist(),
            "training_samples": count,
            "first_target_date": first.isoformat(),
            "last_target_date": last.isoformat(),
            "training_cutoff": cutoff.isoformat(),
        }
    return models


def score(fields, models, area, cutoff: dt.date, latitude=None):
    leads = tuple(sorted(int(key) for key in models))
    if latitude is not None and np.shape(latitude) != np.shape(area):
        raise ValueError("Latitude and ground-area grids must have identical shapes")
    def empty():
        return {"model_sq": 0.0, "persist_sq": 0.0, "model_abs": 0.0,
                "persist_abs": 0.0, "area_sum": 0.0, "samples": 0,
                "days": 0, "first": None, "last": None}
    totals = {lead: {"overall": empty(), "by_target_month": {},
                     "by_target_year": {}, "by_latitude_band": {}}
              for lead in leads}
    band_masks = None if latitude is None else {
        "south_of_80S": latitude <= -80,
        "80S_to_70S": (latitude > -80) & (latitude <= -70),
        "70S_to_60S": (latitude > -70) & (latitude <= -60),
        "north_of_60S": latitude > -60,
    }

    def add_metrics(item, model_delta, persistence_delta, weights, day):
        item["model_sq"] += float(np.sum(weights * model_delta ** 2))
        item["persist_sq"] += float(np.sum(weights * persistence_delta ** 2))
        item["model_abs"] += float(np.sum(weights * np.abs(model_delta)))
        item["persist_abs"] += float(np.sum(weights * np.abs(persistence_delta)))
        item["area_sum"] += float(weights.sum())
        item["samples"] += len(model_delta)
        item["days"] += 1
        item["first"] = item["first"] or day.isoformat()
        item["last"] = day.isoformat()

    history = collections.deque(maxlen=max(leads) + 2)
    for record in fields:
        day, grid, ocean = record
        history.append(record)
        for lead in leads:
            if len(history) <= lead + 1:
                continue
            origin, older = history[-lead - 1], history[-lead - 2]
            if day <= cutoff or (day - origin[0]).days != lead or (origin[0] - older[0]).days != 1:
                continue
            if grid.shape != origin[1].shape or grid.shape != older[1].shape or grid.shape != area.shape:
                continue
            valid = np.isfinite(grid) & np.isfinite(origin[1]) & np.isfinite(older[1])
            if ocean is not None:
                valid &= ocean
            if not valid.any():
                continue
            coefs = np.asarray(models[str(lead)]["coefficients"], dtype=np.float64)
            direct_prediction = np.clip(
                np.einsum("...k,k->...", feature_matrix(origin[1], older[1], day), coefs), 0, 1)
            alpha = models[str(lead)].get("persistence_blend_alpha")
            pred = (direct_prediction if alpha is None
                    else apply_persistence_blend(origin[1], direct_prediction, float(alpha)))
            weights = area[valid].astype(np.float64)
            md = pred[valid] - grid[valid]
            pd = origin[1][valid] - grid[valid]
            groups = totals[lead]
            add_metrics(groups["overall"], md, pd, weights, day)
            month = groups["by_target_month"].setdefault(f"{day.month:02d}", empty())
            year = groups["by_target_year"].setdefault(str(day.year), empty())
            add_metrics(month, md, pd, weights, day)
            add_metrics(year, md, pd, weights, day)
            if latitude is not None:
                for name, band_mask in band_masks.items():
                    band_valid = valid & band_mask
                    if band_valid.any():
                        band_metrics = groups["by_latitude_band"].setdefault(name, empty())
                        band_weights = area[band_valid].astype(np.float64, copy=False)
                        add_metrics(band_metrics, pred[band_valid] - grid[band_valid],
                                    origin[1][band_valid] - grid[band_valid], band_weights, day)
    results = {}
    latitude_names = ("south_of_80S", "80S_to_70S", "70S_to_60S", "north_of_60S")
    def summarize(v):
        if not v["area_sum"]:
            return None
        return {
            "target_days": v["days"], "valid_cell_samples": v["samples"],
            "evaluated_area_km2_samples": v["area_sum"],
            "period": {"first_target_date": v["first"], "last_target_date": v["last"]},
            "area_weighted_rmse": {"direct_model": math.sqrt(v["model_sq"] / v["area_sum"]),
                                   "persistence": math.sqrt(v["persist_sq"] / v["area_sum"])},
            "area_weighted_mae": {"direct_model": v["model_abs"] / v["area_sum"],
                                  "persistence": v["persist_abs"] / v["area_sum"]},
        }
    for lead, groups in totals.items():
        overall = summarize(groups["overall"])
        if overall is None:
            raise ValueError(f"No valid post-cutoff targets for {lead}-day model")
        results[str(lead)] = {
            **overall,
            "by_target_month": {key: summarize(value) for key, value in sorted(groups["by_target_month"].items())},
            "by_target_year": {key: summarize(value) for key, value in sorted(groups["by_target_year"].items())},
            "by_latitude_band": {
                name: (summarize(groups["by_latitude_band"].get(name, empty()))
                       or {"status": "no_valid_target_cells"})
                for name in latitude_names
            } if latitude is not None else {"status": "latitude_unavailable"},
        }
    return results


def validate_model_artifact(artifact: dict, train_manifest_hash: str,
                            cutoff: dt.date, leads: tuple[int, ...]) -> None:
    if artifact.get("model_id") != "sic-direct-horizon-regression-research-v1":
        raise ValueError("Supplied model artifact has an unsupported model_id")
    if artifact.get("training_cutoff") != cutoff.isoformat():
        raise ValueError("Model training cutoff does not match the requested evaluation cutoff")
    if artifact.get("training_source_manifest_sha256") != train_manifest_hash:
        raise ValueError("Model was not trained from the currently verified training manifest")
    lead_models = artifact.get("lead_models")
    if not isinstance(lead_models, dict) or set(lead_models) != {str(lead) for lead in leads}:
        raise ValueError("Model lead set does not match the requested evaluation leads")
    if artifact.get("model_fingerprint") != direct_fingerprint(lead_models):
        raise ValueError("Model coefficient fingerprint does not match its lead coefficients")
    if artifact.get("artifact_sha256") != model_artifact_sha256(artifact):
        raise ValueError("Model artifact SHA-256 does not match its contents")


def run(train_dir: Path, test_dir: Path, cutoff: dt.date, leads: tuple[int, ...],
        model_input: dict | None = None):
    train_fields, train_manifest, train_manifest_hash = load_fields(train_dir)
    test_fields, test_manifest, test_manifest_hash = load_fields(test_dir)
    if train_manifest.get("end_date") and train_manifest["end_date"] < cutoff.isoformat():
        raise ValueError("Training manifest ends before requested cutoff")
    if model_input is None:
        models = fit_models(train_fields(), leads, cutoff)
        if any(item["last_target_date"] != cutoff.isoformat() for item in models.values()):
            raise ValueError("Training archive has no usable target data through the requested cutoff")
    else:
        validate_model_artifact(model_input, train_manifest_hash, cutoff, leads)
        models = model_input["lead_models"]
    test_iter = iter(test_fields())
    first_record = next(test_iter, None)
    first = first_record[1] if first_record and first_record[0] > cutoff else None
    if first is None:
        first = next((g for d, g, _ in test_iter if d > cutoff), None)
    if first is None:
        raise ValueError("No post-cutoff grid in evaluation archive")
    area, latitude = g02202_cell_geometry(test_dir, first.shape)
    replay_fields = test_fields()
    scores = score(replay_fields, models, area, cutoff, latitude)
    artifact = model_input if model_input is not None else {
        "model_id": "sic-direct-horizon-regression-research-v1",
        "target": "daily sea-ice concentration fraction, Antarctic G02202 grid",
        "features": ["intercept", "origin_day_sic", "previous_observed_day_sic", "target_day_sin", "target_day_cos"],
        "training_cutoff": cutoff.isoformat(), "lead_models": models,
        "training_source_manifest_sha256": train_manifest_hash,
        "training_dataset_id": train_manifest.get("dataset_id"),
        "model_fingerprint": direct_fingerprint(models),
        "limitations": ["same-product hindcast", "correlated daily and spatial samples",
                        "not route-scale validation", "not calibrated uncertainty", "not for navigation"],
    }
    report = {
        "status": "frozen_direct_horizon_research_diagnostic",
        "method": "Separate horizon-specific ridge regressions; each forecast uses only the two observed SIC fields at its origin, with no recursive predicted inputs.",
        "training_cutoff": cutoff.isoformat(), "training_source_manifest_sha256": train_manifest_hash,
        "evaluation_source_manifest_sha256": test_manifest_hash,
        "model_fingerprint": artifact["model_fingerprint"],
        "evaluation_scope": "G02202 targets strictly after the frozen training cutoff; source QA/ocean masks applied.",
        "area_weight_method": "Projected pixel area divided by PROJ ellipsoidal areal_scale at each cell centre.",
        "by_lead": scores,
        "caveat": "Offline same-product pan-Antarctic diagnostics only. Do not connect to route decisions or represent as operational forecasts.",
    }
    artifact["artifact_sha256"] = model_artifact_sha256(artifact)
    return artifact, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--cutoff", type=dt.date.fromisoformat, default=dt.date(2016, 12, 31))
    parser.add_argument("--leads", default="1,3,5,7")
    parser.add_argument("--model-output", type=Path, default=Path("models/sic_direct_multilead.json"))
    parser.add_argument("--model-input", type=Path,
                        help="Reuse a fingerprint- and provenance-verified trained artifact instead of fitting it again")
    parser.add_argument("--report-output", type=Path, default=Path("outputs/sic-direct-multilead-holdout.json"))
    args = parser.parse_args()
    try:
        leads = tuple(sorted(set(int(x) for x in args.leads.split(","))))
        if not leads or leads[0] < 1 or leads[-1] > 30:
            raise ValueError("leads must be unique integers in [1,30]")
        model_input = (json.loads(args.model_input.read_text(encoding="utf-8"))
                       if args.model_input else None)
        artifact, report = run(args.training_dir, args.evaluation_dir, args.cutoff, leads, model_input)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for path, data in ((args.model_output, artifact), (args.report_output, report)):
        atomic_json_write(path, data)
    print(json.dumps({"status": report["status"], "model": str(args.model_output),
                      "report": str(args.report_output), "by_lead": report["by_lead"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
