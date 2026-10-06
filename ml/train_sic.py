"""Train a small, reproducible Antarctic SIC gridded lag-regression baseline.

Input: daily Antarctic G02202-like NetCDF files, chronologically named with a
YYYYMMDD token. This is a research baseline; it is not a navigation forecast.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import xarray as xr
try:
    from .identity import model_artifact_sha256, model_fingerprint
except ImportError:  # Support direct execution as `python ml/train_sic.py`.
    from identity import model_artifact_sha256, model_fingerprint

VARIABLES = ("cdr_seaice_conc", "seaice_conc_cdr", "sea_ice_concentration", "ice_conc", "sic", "IceConc")
QA_VARIABLES = ("cdr_seaice_conc_qa_flag", "qa_flag", "qa_of_cdr_seaice_conc")
EXCLUDED_QA_BITS = 8 | 16 | 64  # no TB input, invalid ice/ocean mask, temporal interpolation
REGULARIZATION_DIAGONAL = (0.0, 0.02, 0.02, 0.02, 0.02)


def verify_g02202_manifest(data_dir: Path, paths: list[str | Path], manifest: dict | None) -> None:
    """Refuse to train/evaluate G02202 data whose declared archive is partial or altered."""
    if manifest is None:
        return
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a JSON object.")
    if manifest.get("dataset_id") != "G02202":
        return
    if not manifest.get("complete") or not isinstance(manifest.get("files"), list):
        raise ValueError("G02202 manifest must declare a complete annual-file inventory before training/evaluation.")
    inventory_by_name = {}
    for item in manifest["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise ValueError("G02202 manifest contains an invalid annual-file record.")
        name, digest = item["file"], item.get("sha256")
        if (Path(name).name != name or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())):
            raise ValueError(f"G02202 manifest has an unsafe filename or missing SHA-256 for {name}.")
        if name in inventory_by_name:
            raise ValueError(f"G02202 manifest lists {name} more than once.")
        inventory_by_name[name] = item

    root = data_dir.resolve()
    actual_paths = [Path(path).resolve() for path in paths]
    for path in actual_paths:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"G02202 inventory file resolves outside the dataset directory: {path.name}.") from exc
    actual_names = {path.name for path in actual_paths}
    if actual_names != set(inventory_by_name):
        missing = sorted(set(inventory_by_name) - actual_names)
        unlisted = sorted(actual_names - set(inventory_by_name))
        raise ValueError(f"G02202 files do not match the complete manifest (missing={missing[:3]}, unlisted={unlisted[:3]}).")
    for path in actual_paths:
        item = inventory_by_name[path.name]
        expected_size = item.get("bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"G02202 file size does not match its manifest entry: {path.name}.")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(2 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"].lower():
            raise ValueError(f"G02202 SHA-256 does not match its manifest entry: {path.name}.")


def variable_name(ds: xr.Dataset, path: str) -> str:
    name = next((candidate for candidate in VARIABLES if candidate in ds.data_vars), None)
    if name is None:
        raise ValueError(f"No recognized SIC variable in {path}; found {list(ds.data_vars)}")
    return name


def inventory(paths: list[str]) -> list[tuple[str, str, list[tuple[dt.date, int | None]], np.ndarray | None, str | None]]:
    """Index daily or time-stacked NetCDF files without loading grid values."""
    groups = []
    for path in paths:
        with xr.open_dataset(path, decode_cf=True) as ds:
            name = variable_name(ds, path)
            ocean_mask = None
            for mask_name in ("landmask", "ocean_mask", "sea_mask"):
                if mask_name in ds.variables and ds[mask_name].ndim == 2:
                    mask_values = np.asarray(ds[mask_name].values, dtype=np.float32)
                    if mask_name == "landmask" and "time" in ds[name].dims:
                        first_sic = np.asarray(ds[name].isel(time=0).squeeze().values)
                        candidates = [np.isfinite(mask_values) & (mask_values > 0.5),
                                      np.isfinite(mask_values) & (mask_values <= 0.5),
                                      ~np.isfinite(mask_values)]
                        ocean_mask = max(candidates, key=lambda candidate: int(np.count_nonzero(candidate & np.isfinite(first_sic))))
                    else:
                        ocean_mask = np.isfinite(mask_values) & (mask_values > 0.5)
                    break
            qa_name = next((candidate for candidate in QA_VARIABLES if candidate in ds.data_vars), None)
            if "time" in ds[name].dims:
                records = [(dt.date.fromisoformat(str(value)[:10]), index)
                           for index, value in enumerate(ds.time.values)]
            else:
                match = re.search(r"(19|20)\d{6}", os.path.basename(path))
                if not match:
                    print(f"Skipping file without a daily time coordinate: {path}", file=sys.stderr)
                    continue
                records = [(dt.datetime.strptime(match.group(0), "%Y%m%d").date(), None)]
        groups.append((path, name, records, ocean_mask, qa_name))
    groups.sort(key=lambda group: group[2][0][0])
    return groups


def grids(groups):
    """Stream daily 2-D concentration arrays with conservative QA/leakage masking."""
    for path, name, records, ocean_mask, qa_name in groups:
        try:
            with xr.open_dataset(path, decode_cf=True) as ds:
                variable = ds[name]
                for date, index in records:
                    sample = variable.isel(time=index) if index is not None else variable
                    values = np.asarray(sample.squeeze().values, dtype=np.float32)
                    if values.ndim != 2:
                        raise ValueError(f"Expected one 2-D daily grid in {path}, got {values.shape}")
                    values[(values < 0) | (values > 100)] = np.nan
                    finite = np.isfinite(values)
                    if finite.any() and float(values[finite].max()) > 1.01:
                        values /= 100.0
                    if qa_name:
                        qa = ds[qa_name]
                        if "time" in qa.dims:
                            qa = qa.isel(time=index if index is not None else 0)
                        quality = np.asarray(qa.squeeze().values, dtype=np.uint8)
                        if quality.shape != values.shape:
                            raise ValueError(f"QA field {qa_name} shape {quality.shape} does not match SIC {values.shape} in {path}")
                        values[(quality & EXCLUDED_QA_BITS) != 0] = np.nan
                    yield date, values.copy(), ocean_mask
        except (ValueError, OSError, KeyError) as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)


def features(a: np.ndarray, b: np.ndarray, date: dt.date) -> np.ndarray:
    phase = 2 * math.pi * (date.timetuple().tm_yday - 1) / 365.2425
    return np.stack((np.ones_like(a), a, b, np.full_like(a, math.sin(phase)),
                     np.full_like(a, math.cos(phase))), axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing quality-controlled daily NetCDF files")
    parser.add_argument("--additional-data-dir", action="append", default=[],
                        help="Additional directory from the same gridded product; may be repeated")
    parser.add_argument("--output", default="models/sic_baseline.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-start", type=dt.date.fromisoformat,
                        help="Explicit first validation target date; useful for frozen annual holdouts")
    parser.add_argument("--start-date", type=dt.date.fromisoformat, help="Optional first date to include from the input archive")
    parser.add_argument("--end-date", type=dt.date.fromisoformat, help="Optional last date to include from the input archive")
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("validation-fraction must be between 0.05 and 0.4")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("start-date must not be after end-date")

    source_directories = [Path(args.data_dir), *(Path(value) for value in args.additional_data_dir)]
    if len({path.resolve() for path in source_directories}) != len(source_directories):
        parser.error("Source directories must be distinct.")
    paths: list[str] = []
    source_manifests: list[dict | None] = []
    source_manifest_hashes: list[str | None] = []
    for source_dir in source_directories:
        directory_paths = glob.glob(os.path.join(str(source_dir), "**", "*.nc"), recursive=True)
        source_manifest_path = source_dir / "manifest.json"
        source_manifest = None
        if source_manifest_path.exists():
            try:
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                parser.error(f"Invalid source manifest {source_manifest_path}: {exc}")
        try:
            verify_g02202_manifest(source_dir, directory_paths, source_manifest)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        paths.extend(directory_paths)
        source_manifests.append(source_manifest)
        source_manifest_hashes.append(hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
                                      if source_manifest_path.is_file() else None)
    if len(source_directories) > 1:
        if any(manifest is None or manifest.get("dataset_id") != "G02202"
               or manifest.get("hemisphere") != "Southern" for manifest in source_manifests):
            parser.error("Combining directories requires complete Southern G02202 manifests for every source.")
        reference_grid = None
        for path in paths:
            with xr.open_dataset(path, decode_cf=False) as ds:
                if not {"x", "y", "crs"}.issubset(ds.variables):
                    parser.error(f"Combined G02202 source lacks x/y/CRS metadata: {path}")
                grid = (np.asarray(ds.x.values), np.asarray(ds.y.values),
                        ds.crs.attrs.get("crs_wkt") or ds.crs.attrs.get("spatial_ref"))
                if not grid[2]:
                    parser.error(f"Combined G02202 source lacks CRS definition: {path}")
                if reference_grid is None:
                    reference_grid = grid
                elif (not np.array_equal(grid[0], reference_grid[0])
                      or not np.array_equal(grid[1], reference_grid[1])
                      or grid[2] != reference_grid[2]):
                    parser.error(f"Combined G02202 grids/CRS do not match: {path}")
    groups = inventory(paths)
    if args.start_date or args.end_date:
        filtered_groups = []
        for path, name, records, ocean_mask, qa_name in groups:
            selected = [(day, index) for day, index in records
                        if (args.start_date is None or day >= args.start_date)
                        and (args.end_date is None or day <= args.end_date)]
            if selected:
                filtered_groups.append((path, name, selected, ocean_mask, qa_name))
        groups = filtered_groups
    dates = [date for _, _, records, _, _ in groups for date, _ in records]
    dates.sort()
    if len(dates) != len(set(dates)):
        parser.error("Source files contain duplicate daily timestamps.")
    if len(dates) < 100:
        parser.error(f"Need at least 100 daily fields; found {len(dates)}")
    if args.validation_start:
        cutoff = args.validation_start
        if not dates[2] < cutoff <= dates[-1]:
            parser.error("--validation-start must leave at least three earlier fields and one validation field.")
    else:
        split_at = max(2, min(len(dates) - 1, int(len(dates) * (1 - args.validation_fraction))))
        cutoff = dates[split_at]
    gram = np.zeros((5, 5), dtype=np.float64)
    rhs = np.zeros(5, dtype=np.float64)
    climatology_sum = climatology_count = None
    train_n = 0
    previous_older = previous = None
    train_first = train_last = None
    for date, current, ocean_mask in grids(groups):
        if previous is not None and previous_older is not None and (date - previous[0]).days == 1 and (previous[0] - previous_older[0]).days == 1 and current.shape == previous[1].shape == previous_older[1].shape:
            if date < cutoff:
                x = features(previous[1], previous_older[1], date)
                target = current
                valid = np.isfinite(target) & np.isfinite(previous[1]) & np.isfinite(previous_older[1])
                if ocean_mask is not None:
                    valid &= ocean_mask
                if climatology_sum is None:
                    climatology_sum = np.zeros((12, *current.shape), dtype=np.float64)
                    climatology_count = np.zeros((12, *current.shape), dtype=np.uint32)
                xv = x[valid].astype(np.float64)
                yv = target[valid].astype(np.float64)
                gram += xv.T @ xv
                rhs += xv.T @ yv
                month = date.month - 1
                climatology_sum[month][valid] += yv
                climatology_count[month][valid] += 1
                train_n += len(yv)
                train_first = train_first or date
                train_last = date
        if previous is not None and (date - previous[0]).days != 1:
            previous_older = None
        else:
            previous_older = previous
        previous = (date, current)

    if train_n < 10_000:
        parser.error("Insufficient contiguous valid grid samples; check files, variables, and temporal coverage")
    regularizer = np.diag(REGULARIZATION_DIAGONAL)
    weights = np.linalg.solve(gram + regularizer, rhs)
    model_sq = persist_sq = climatology_sq = model_abs = 0.0
    count = 0
    climatology_count_validation = 0
    validation_first = validation_last = None
    previous_older = previous = None
    for date, current, ocean_mask in grids(groups):
        if previous is not None and previous_older is not None and date >= cutoff and (date - previous[0]).days == 1 and (previous[0] - previous_older[0]).days == 1 and current.shape == previous[1].shape == previous_older[1].shape:
            x = features(previous[1], previous_older[1], date)
            valid = np.isfinite(current) & np.isfinite(previous[1]) & np.isfinite(previous_older[1])
            if ocean_mask is not None:
                valid &= ocean_mask
            pred = np.clip(np.einsum("...k,k->...", x, weights), 0, 1)
            diff = pred[valid] - current[valid]
            base_diff = previous[1][valid] - current[valid]
            model_sq += float(diff @ diff)
            persist_sq += float(base_diff @ base_diff)
            model_abs += float(np.abs(diff).sum())
            month = date.month - 1
            climatology_valid = valid & (climatology_count[month] > 0)
            climate = climatology_sum[month][climatology_valid] / climatology_count[month][climatology_valid]
            climate_diff = climate - current[climatology_valid]
            climatology_sq += float(climate_diff @ climate_diff)
            climatology_count_validation += len(climate_diff)
            count += len(diff)
            validation_first = validation_first or date
            validation_last = date
        if previous is not None and (date - previous[0]).days != 1:
            previous_older = None
        else:
            previous_older = previous
        previous = (date, current)
    if count == 0:
        parser.error("No contiguous validation samples after the chronological split")

    source_manifest = source_manifests[0]
    if source_manifest and source_manifest.get("dataset_id") == "G02202":
        year_scope = f"{dates[0].year}-{dates[-1].year}"
        validation_scope = f"Chronological holdout within the NOAA/NSIDC G02202 v6 input archive ({year_scope}). This is one product and is not an independent cross-product evaluation."
    elif source_manifest and source_manifest.get("paper_doi"):
        validation_scope = "Chronological holdout from the test dataset released with the Antarctic ConvLSTM paper. It is held out from this baseline fit but is not an independent external-product evaluation."
    else:
        source_label = (source_manifest or {}).get("dataset_id") or "the supplied input files"
        validation_scope = f"Chronological holdout within {source_label}. This single-source split is not independent cross-product validation."
    artifact = {
        "model_id": "sic-lag-regression-v1",
        "target": "daily sea-ice concentration fraction, Antarctic grid",
        "features": ["intercept", "previous_day_sic", "two_days_prior_sic", "day_of_year_sin", "day_of_year_cos"],
        "coefficients": weights.tolist(),
        "training": {"files": len(groups), "daily_fields": len(dates), "samples": train_n,
                     "selected_period_start": dates[0].isoformat(), "selected_period_end": dates[-1].isoformat(),
                     "first_train_target_date": train_first.isoformat() if train_first else None,
                     "last_train_target_date": train_last.isoformat() if train_last else None,
                     "validation_start": validation_first.isoformat() if validation_first else None,
                     "validation_end": validation_last.isoformat() if validation_last else None},
        "validation": {"grid_cell_samples": count, "mae": model_abs / count,
                       "rmse": math.sqrt(model_sq / count), "persistence_rmse": math.sqrt(persist_sq / count),
                       "monthly_climatology_rmse": (math.sqrt(climatology_sq / climatology_count_validation)
                                                     if climatology_count_validation else None),
                       "monthly_climatology_samples": climatology_count_validation},
        "validation_scope": validation_scope,
        "training_configuration": {
            "validation_fraction_requested": args.validation_fraction,
            "validation_start_requested": args.validation_start.isoformat() if args.validation_start else None,
            "chronological_split_cutoff_date": cutoff.isoformat(),
            "split_rule": "Sorted available daily timestamps; fit only on targets before cutoff; validate at or after cutoff.",
            "ridge_regularization_diagonal": list(REGULARIZATION_DIAGONAL),
            "seasonal_phase_denominator_days": 365.2425,
            "prediction_clipping": [0.0, 1.0],
            "minimum_daily_fields": 100,
            "minimum_fit_grid_samples": 10000,
        },
        "reproducibility": {
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_manifest_sha256": source_manifest_hashes[0],
            "source_manifest_sha256s": source_manifest_hashes,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "xarray": xr.__version__,
            },
            "numeric_reproducibility": "Floating-point solver results may vary slightly across BLAS/platforms; compare metrics within a documented tolerance and verify the model fingerprint separately.",
        },
        "data_source": "NetCDF input files; upstream provenance is not independently verified by the trainer.",
        "input_file_names": [os.path.basename(group[0]) for group in groups],
        "source_manifest": source_manifest,
        "source_manifests": source_manifests,
        "source_directories": [str(path) for path in source_directories],
        "validation_masking": "Finite SIC cells; static ocean mask applied when provided. Recognized QA masks exclude no-input (bit 8), invalid ice/ocean mask (bit 16), and temporal-interpolation (bit 64) cells from inputs and targets.",
        "warning": "Research baseline only. Not calibrated or approved for navigation or operational decisions.",
    }
    artifact["model_fingerprint"] = model_fingerprint(artifact)
    artifact["artifact_sha256"] = model_artifact_sha256(artifact)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(artifact, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(json.dumps({"artifact": args.output, "validation": artifact["validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
