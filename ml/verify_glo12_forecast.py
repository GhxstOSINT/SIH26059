"""Match one prospectively captured GLO12 forecast to a later OSI-SAF scene.

Produces a source-bound research verification sample. It does not calibrate
uncertainty from one run, assess a vessel route, or authorize navigation.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from ml.archive_glo12_forecast import inventory_forecast
from ml.inventory_copernicus_sic import inventory as inventory_observation
from ml.research_transect import load_research_transect, sample_transect


def match_run(manifest_path: Path, observation_path: Path,
              baseline_manifest_path: Path | None = None,
              research_transect_path: Path | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Regular forecast manifest is required")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(recorded, dict) or not recorded.get("eligible_for_prospective_verification"):
        raise ValueError("Forecast was not captured before its valid date")
    source_name = recorded.get("source_file")
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise ValueError("Forecast manifest source path is invalid")
    forecast_path = manifest_path.parent / source_name
    expected = inventory_forecast(
        forecast_path,
        captured_at=datetime.fromisoformat(recorded["captured_at_utc"]),
        provider_last_modified_at=(datetime.fromisoformat(recorded["provider_last_modified_at_utc"])
                                   if recorded.get("provider_last_modified_at_utc") else None),
    )
    if expected != recorded:
        raise ValueError("Forecast manifest does not match the original file")
    target_day = date.fromisoformat(recorded["valid_date"])
    observed = inventory_observation(observation_path, target_day, target_day)
    baseline_record = None
    baseline_path = None
    if baseline_manifest_path is not None:
        if not baseline_manifest_path.is_file() or baseline_manifest_path.is_symlink():
            raise ValueError("Regular baseline capture manifest is required")
        baseline_record = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
        baseline_name = baseline_record.get("source_file")
        if not isinstance(baseline_name, str) or Path(baseline_name).name != baseline_name:
            raise ValueError("Baseline observation path is invalid")
        baseline_path = baseline_manifest_path.parent / baseline_name
        baseline_day = date.fromisoformat(baseline_record["period"]["start_date"])
        if (baseline_record["period"]["end_date"] != baseline_day.isoformat()
                or baseline_day >= date.fromisoformat(recorded["bulletin_date"])):
            raise ValueError("Baseline must be a prior one-day observation")
        if datetime.fromisoformat(baseline_record["captured_at_utc"]) >= datetime.fromisoformat(recorded["forecast_base_time_utc"]):
            raise ValueError("Baseline was captured after the forecast bulletin time")
        baseline_inventory = inventory_observation(baseline_path, baseline_day, baseline_day)
        for field in ("source_file", "source_bytes", "source_sha256", "dataset_id", "dataset_version", "period"):
            if baseline_record.get(field) != baseline_inventory[field]:
                raise ValueError("Baseline capture provenance changed")
    with xr.open_dataset(forecast_path) as forecast_ds, xr.open_dataset(observation_path) as observation_ds:
        target = observation_ds.isel(time=0)
        forecast = forecast_ds.siconc.isel(time=0).interp(
            latitude=target.latitude, longitude=target.longitude, method="linear")
        predicted = np.asarray(forecast.values, dtype=np.float32)
        truth = np.asarray(target.ice_conc.values, dtype=np.float32) / 100
        uncertainty = np.asarray(target.total_uncertainty.values, dtype=np.float32) / 100
        flags = np.asarray(target.status_flag.values)
        valid = (np.isfinite(predicted) & np.isfinite(truth) & np.isfinite(uncertainty)
                 & (predicted >= 0) & (predicted <= 1)
                 & (truth >= 0) & (truth <= 1)
                 & (uncertainty >= 0) & (uncertainty <= 1)
                 & np.isfinite(flags) & (flags == 0))
        baseline_values = None
        if baseline_path is not None:
            with xr.open_dataset(baseline_path) as baseline_ds:
                baseline_target = baseline_ds.isel(time=0)
                baseline_field = baseline_target.ice_conc.interp(
                    latitude=target.latitude, longitude=target.longitude, method="linear")
                baseline_flags = baseline_target.status_flag.interp(
                    latitude=target.latitude, longitude=target.longitude, method="nearest")
                baseline_values = np.asarray(baseline_field.values, dtype=np.float32) / 100
                baseline_status = np.asarray(baseline_flags.values)
                baseline_values = np.where(
                    np.isfinite(baseline_status) & (baseline_status == 0),
                    baseline_values, np.nan)
                valid &= (np.isfinite(baseline_values) & (baseline_values >= 0)
                          & (baseline_values <= 1))
        if not np.any(valid):
            raise ValueError("No nominal OSI-SAF cells match the forecast grid")
        forecast_values = predicted[valid].astype(np.float32)
        observed_values = truth[valid].astype(np.float32)
        observation_uncertainty = uncertainty[valid].astype(np.float32)
        deltas = forecast_values.astype(np.float64) - observed_values.astype(np.float64)
        persistence = baseline_values[valid].astype(np.float32) if baseline_values is not None else None
        report = {
            "schema_version": 1,
            "scope": "single_run_research_verification_sample",
            "bulletin_date": recorded["bulletin_date"],
            "valid_date": recorded["valid_date"],
            "lead_days": recorded["lead_days"],
            "forecast_source_sha256": recorded["source_sha256"],
            "forecast_captured_at_utc": recorded["captured_at_utc"],
            "observation_dataset_id": observed["dataset_id"],
            "observation_source_sha256": observed["source_sha256"],
            "grid_alignment": "linear interpolation of GLO12 SIC to nominal OSI-SAF geographic cells; no extrapolation",
            "quality_mask": "finite 0–1 SIC and uncertainty; OSI-SAF status_flag == 0 (nominal retrieval)",
            "matched_cells": int(valid.sum()),
            "observation_grid_cells": int(valid.size),
            "matched_fraction": float(valid.mean()),
            "mae_fraction": float(np.mean(np.abs(deltas))),
            "rmse_fraction": float(np.sqrt(np.mean(deltas ** 2))),
            "mean_bias_fraction": float(np.mean(deltas)),
            "mean_observation_uncertainty_fraction": float(np.mean(observation_uncertainty)),
            "as_issued_persistence_comparison": (
                {"baseline_observation_date": baseline_record["period"]["start_date"],
                 "baseline_source_sha256": baseline_record["source_sha256"],
                 "baseline_captured_at_utc": baseline_record["captured_at_utc"],
                 "persistence_mae_fraction": float(np.mean(np.abs(persistence - observed_values))),
                 "persistence_rmse_fraction": float(np.sqrt(np.mean((persistence - observed_values) ** 2))),
                 "common_cells": int(valid.sum())}
                if persistence is not None else None),
            "uncertainty_calibrated": False,
            "navigation_clearance": False,
            "limitations": [
                "A single run and spatially correlated cells cannot establish forecast skill or calibrated uncertainty.",
                "OSI-SAF retrieval uncertainty is not forecast uncertainty.",
                "Independent vessel outcomes and route-scale safety assessment are not included.",
            ],
        }
        if research_transect_path is not None:
            route, route_sha = load_research_transect(research_transect_path)
            route_samples = {}
            report["research_transect"] = sample_transect(
                predicted, truth, uncertainty, flags,
                np.asarray(target.latitude.values), np.asarray(target.longitude.values),
                route, route_sha, bulletin_date=date.fromisoformat(recorded["bulletin_date"]),
                persistence=baseline_values, sample_arrays=route_samples)
    samples = {"forecast_sic_fraction": forecast_values,
               "observed_sic_fraction": observed_values,
               "observation_uncertainty_fraction": observation_uncertainty}
    if persistence is not None:
        samples["persistence_sic_fraction"] = persistence
    if research_transect_path is not None:
        samples.update(route_samples)
    return report, samples


def write_match(report: dict, samples: dict[str, np.ndarray], output: Path) -> tuple[Path, Path]:
    """Write a source-bound sample pair without replacing existing evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_path = output.with_suffix(".npz")
    if output.exists() or sample_path.exists():
        raise ValueError("Existing verification evidence cannot be overwritten")
    sample_temp = sample_path.with_suffix(".npz.tmp")
    # The samples are written first and bound by digest into the report.
    with sample_temp.open("wb") as target:
        np.savez_compressed(target, **samples)
    sample_temp.replace(sample_path)
    digest = hashlib.sha256()
    with sample_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    bound_report = {**report, "samples_file": sample_path.name, "samples_sha256": digest.hexdigest()}
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(bound_report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output, sample_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast_manifest", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--research-transect", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, samples = match_run(args.forecast_manifest, args.observation,
                                args.baseline_manifest, args.research_transect)
    output, sample_path = write_match(report, samples, args.output)
    print(json.dumps({"report": str(output), "samples": str(sample_path),
                      "matched_cells": report["matched_cells"], "uncertainty_calibrated": False}, indent=2))


if __name__ == "__main__":
    main()
