"""Calibrate persistence blends on a pre-holdout period for direct SIC horizons.

Fits base regressions on an early period, selects one area-weighted MAE-optimal
blend per lead on a later calibration period, then evaluates only on a separate
post-calibration archive. No operational-use claim is made.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

try:
    from .evaluate_direct_multilead_sic import (
        atomic_json_write, direct_fingerprint, feature_matrix, fit_models,
        g02202_cell_geometry, load_fields, run, score,
    )
    from .identity import model_artifact_sha256
except ImportError:
    from evaluate_direct_multilead_sic import (
        atomic_json_write, direct_fingerprint, feature_matrix, fit_models,
        g02202_cell_geometry, load_fields, run, score,
    )
    from identity import model_artifact_sha256


def calibrate_blends(fields, models: dict, area: np.ndarray,
                     start: dt.date, end: dt.date, bins: int = 10001) -> dict[str, float]:
    """Select per-lead convex persistence weights by weighted-median L1 minimization.

    For each valid cell, absolute error of ``persistence + alpha * (model -
    persistence)`` is a weighted absolute distance from its zero-error alpha.
    The area-times-absolute-delta weighted median therefore minimizes pooled
    area-weighted MAE. A fixed histogram approximates the median reproducibly.
    """
    if start > end or bins < 3:
        raise ValueError("Invalid calibration period or histogram resolution")
    leads = tuple(sorted(int(key) for key in models))
    histograms = {lead: np.zeros(bins, dtype=np.float64) for lead in leads}
    history = collections.deque(maxlen=max(leads) + 2)
    for day, grid, ocean in fields:
        history.append((day, grid, ocean))
        if not start <= day <= end:
            continue
        for lead in leads:
            if len(history) <= lead + 1:
                continue
            origin, older = history[-lead - 1], history[-lead - 2]
            if (day - origin[0]).days != lead or (origin[0] - older[0]).days != 1:
                continue
            if grid.shape != origin[1].shape or grid.shape != older[1].shape or grid.shape != area.shape:
                continue
            valid = np.isfinite(grid) & np.isfinite(origin[1]) & np.isfinite(older[1])
            if ocean is not None:
                valid &= ocean
            if not valid.any():
                continue
            coefs = np.asarray(models[str(lead)]["coefficients"], dtype=np.float64)
            prediction = np.clip(np.einsum("...k,k->...",
                                           feature_matrix(origin[1], older[1], day), coefs), 0, 1)
            delta = prediction[valid] - origin[1][valid]
            adjustable = np.abs(delta) > 1e-10
            if not adjustable.any():
                continue
            target_delta = grid[valid][adjustable] - origin[1][valid][adjustable]
            ratios = np.clip(target_delta / delta[adjustable], 0.0, 1.0)
            weights = area[valid][adjustable].astype(np.float64) * np.abs(delta[adjustable])
            indices = np.minimum((ratios * (bins - 1)).astype(np.int64), bins - 1)
            histograms[lead] += np.bincount(indices, weights=weights, minlength=bins)
    alphas = {}
    for lead, histogram in histograms.items():
        total = float(histogram.sum())
        if total <= 0:
            alpha = 0.0
        else:
            index = int(np.searchsorted(np.cumsum(histogram), total / 2.0, side="left"))
            if index == 0:
                alpha = 0.0
            elif index == bins - 1:
                alpha = 1.0
            else:
                alpha = index / (bins - 1)
        alphas[str(lead)] = float(alpha)
    return alphas


def build_calibrated_model(training_dir: Path, fit_cutoff: dt.date,
                           calibration_start: dt.date, calibration_end: dt.date,
                           leads: tuple[int, ...], bins: int = 10001):
    fields, manifest, manifest_hash = load_fields(training_dir)
    raw_models = fit_models(fields(), leads, fit_cutoff)
    if any(model["last_target_date"] != fit_cutoff.isoformat() for model in raw_models.values()):
        raise ValueError("Fit archive has no usable targets through the requested cutoff")
    first_field = next(fields(), None)
    if first_field is None:
        raise ValueError("Training archive contains no readable daily fields")
    area, latitude = g02202_cell_geometry(training_dir, first_field[1].shape)
    alphas = calibrate_blends(fields(), raw_models, area, calibration_start, calibration_end, bins)
    blended = {}
    for lead, model in raw_models.items():
        alpha = alphas[lead]
        blended[lead] = {
            **model,
            "base_direct_coefficients": model["coefficients"],
            "coefficients": model["coefficients"],
            "persistence_blend_alpha": alpha,
        }
    artifact = {
        "model_id": "sic-direct-horizon-regression-research-v1",
        "target": "daily sea-ice concentration fraction, Antarctic G02202 grid",
        "features": ["intercept", "origin_day_sic", "previous_observed_day_sic", "target_day_sin", "target_day_cos"],
        "training_cutoff": fit_cutoff.isoformat(),
        "training_dataset_id": manifest.get("dataset_id"),
        "training_source_manifest_sha256": manifest_hash,
        "calibration": {
            "period_start": calibration_start.isoformat(),
            "period_end": calibration_end.isoformat(),
            "method": "per-lead area-weighted absolute-error minimization over convex blends of direct prediction and persistence; 10,001-bin weighted-median approximation",
            "alpha_by_lead": alphas,
            "warning": "Calibration-period metrics are used for parameter selection and are not an independent validation score.",
        },
        "lead_models": blended,
        "model_fingerprint": direct_fingerprint(blended),
        "limitations": ["same-product hindcast", "correlated daily and spatial samples",
                        "not route-scale validation", "not calibrated uncertainty", "not for navigation"],
    }
    artifact["artifact_sha256"] = model_artifact_sha256(artifact)
    calibration_inputs = (record for record in fields() if record[0] <= calibration_end)
    calibration_report = score(calibration_inputs, blended, area, fit_cutoff, latitude)
    return artifact, calibration_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--fit-cutoff", type=dt.date.fromisoformat, default=dt.date(2014, 12, 31))
    parser.add_argument("--calibration-start", type=dt.date.fromisoformat, default=dt.date(2015, 1, 1))
    parser.add_argument("--calibration-end", type=dt.date.fromisoformat, default=dt.date(2016, 12, 31))
    parser.add_argument("--leads", default="1,3,5,7")
    parser.add_argument("--histogram-bins", type=int, default=10001)
    parser.add_argument("--model-output", type=Path, default=Path("models/sic_direct_multilead_blended.json"))
    parser.add_argument("--report-output", type=Path, default=Path("outputs/sic-direct-multilead-blended-holdout.json"))
    args = parser.parse_args()
    try:
        leads = tuple(sorted(set(int(value) for value in args.leads.split(","))))
        if not leads or leads[0] < 1 or leads[-1] > 30:
            raise ValueError("leads must be unique integers in [1,30]")
        if args.calibration_start <= args.fit_cutoff or args.calibration_end < args.calibration_start:
            raise ValueError("Calibration dates must follow the fit cutoff and form an inclusive valid period")
        model, calibration_scores = build_calibrated_model(
            args.training_dir, args.fit_cutoff, args.calibration_start,
            args.calibration_end, leads, args.histogram_bins)
        final_model, report = run(args.training_dir, args.evaluation_dir,
                                  args.fit_cutoff, leads, model)
        report["method"] = ("Direct horizon-specific SIC regressions fit through the fit cutoff, "
                            "then blended with persistence using per-lead area-weighted MAE "
                            "calibration on a disjoint historical window.")
        report["calibration"] = {
            **final_model["calibration"],
            "calibration_period_scores": calibration_scores,
        }
        report["evaluation_scope"] = (
            "G02202 targets strictly after the frozen 2015-2016 calibration period; "
            "source QA/ocean masks applied. Training cutoff: " + args.fit_cutoff.isoformat() + ".")
        report["calibration_warning"] = (
            "Calibration-period scores are in-sample to alpha selection; use only the later "
            "frozen holdout to assess generalization. All scores are pan-Antarctic research "
            "diagnostics, not route-safety evidence.")
        atomic_json_write(args.model_output, final_model)
        atomic_json_write(args.report_output, report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "model": str(args.model_output),
                      "report": str(args.report_output),
                      "alpha_by_lead": final_model["calibration"]["alpha_by_lead"],
                      "holdout": {lead: values["area_weighted_mae"]
                                  for lead, values in report["by_lead"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
