"""Research-only NSIDC-0116 SIC-advection hindcast against persistence."""
from __future__ import annotations

import argparse
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import xarray as xr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sic(dataset: xr.Dataset, day: date) -> tuple[np.ndarray, np.ndarray]:
    matching = np.flatnonzero(dataset.time.values.astype("datetime64[D]") == np.datetime64(day))
    if matching.size != 1:
        raise ValueError(f"SIC input must contain exactly one field for {day}; found {matching.size}.")
    index = int(matching[0])
    values = np.asarray(dataset["cdr_seaice_conc"].isel(time=index).values, dtype=np.float32)
    qa = np.asarray(dataset["cdr_seaice_conc_qa_flag"].isel(time=index).values, dtype=np.uint8)
    temporal = np.asarray(dataset["cdr_seaice_conc_interp_temporal_flag"].isel(time=index).values, dtype=np.uint8)
    valid = np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    valid &= (qa & (8 | 16 | 32 | 64)) == 0
    valid &= temporal == 0
    values[~valid] = np.nan
    return values, valid


def backtrace_sic(base: np.ndarray, displacement_x_m: np.ndarray,
                  displacement_y_m: np.ndarray, motion_support: np.ndarray,
                  pixel_size_x_m: float, pixel_size_y_m: float,
                  minimum_support: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Backtrace SIC from destination pixels to source pixels on a north-up grid."""
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise RuntimeError("NSIDC advection requires scipy from requirements-motion-research.txt") from exc
    arrays = [np.asarray(array) for array in
              (base, displacement_x_m, displacement_y_m, motion_support)]
    if any(array.shape != arrays[0].shape for array in arrays[1:]) or arrays[0].ndim != 2:
        raise ValueError("SIC, displacement, and support fields must be matching 2-D grids.")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("Pixel sizes must be positive metre distances.")
    base, dx, dy, support = arrays
    rows, cols = np.indices(base.shape, dtype=np.float32)
    source_rows = rows + dy.astype(np.float32) / np.float32(pixel_size_y_m)
    source_cols = cols - dx.astype(np.float32) / np.float32(pixel_size_x_m)
    sample_coordinates = np.stack([source_rows, source_cols])
    valid_base = np.isfinite(base)
    advected = map_coordinates(np.where(valid_base, base, 0.0).astype(np.float32),
                               sample_coordinates, order=1, mode="constant", cval=0.0,
                               prefilter=False)
    sampled_support = map_coordinates(valid_base.astype(np.float32), sample_coordinates,
                                      order=1, mode="constant", cval=0.0, prefilter=False)
    valid = (np.isfinite(dx) & np.isfinite(dy) & np.isfinite(support)
             & (support >= minimum_support) & (sampled_support >= minimum_support))
    advected[~valid] = np.nan
    return advected, valid


def _manifest_entries(manifest: dict, start: date, end: date) -> dict[str, dict]:
    if manifest.get("dataset_id") != "NSIDC-0116" or manifest.get("complete") is not True:
        raise ValueError("Motion manifest must be a complete NSIDC-0116 acquisition manifest.")
    try:
        archive_start = date.fromisoformat(manifest["requested_period"]["start"])
        archive_end = date.fromisoformat(manifest["requested_period"]["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Motion manifest has no valid requested_period.") from exc
    if archive_start > start - timedelta(days=1) or archive_end < end - timedelta(days=1):
        raise ValueError("Motion manifest does not cover the one-day-lag motion dates for this experiment.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Motion manifest contains no file inventory.")
    return {entry["file"]: entry for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("file"), str)}


def motion_archive_entry(entries: dict[str, dict], motion_day: date) -> dict:
    matching = []
    for entry in entries.values():
        try:
            first = date.fromisoformat(entry["time_start"])
            last = date.fromisoformat(entry["time_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if first <= motion_day <= last:
            matching.append(entry)
    if len(matching) != 1:
        raise ValueError(f"Expected one NSIDC file covering {motion_day}; found {len(matching)}.")
    return matching[0]


def load_aligned_motion(path: Path, motion_entry: dict, sic_file_hash: str,
                        motion_day: date, expected_shape: tuple[int, int],
                        expected_x: np.ndarray, expected_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, decode_cf=True) as dataset:
        attrs = dataset.attrs
        if attrs.get("motion_date") != motion_day.isoformat():
            raise ValueError(f"Aligned motion date metadata mismatch in {path.name}.")
        if attrs.get("source_motion_file") != motion_entry.get("file"):
            raise ValueError(f"Aligned motion source file mismatch in {path.name}.")
        if attrs.get("source_motion_sha256") != motion_entry.get("sha256"):
            raise ValueError(f"Aligned motion source checksum mismatch in {path.name}.")
        if attrs.get("sic_reference_sha256") != sic_file_hash:
            raise ValueError(f"Aligned motion SIC-reference checksum mismatch in {path.name}.")
        if tuple(dataset.sizes.get(axis, -1) for axis in ("y", "x")) != expected_shape:
            raise ValueError(f"Aligned motion shape mismatch in {path.name}.")
        if not np.array_equal(dataset.x.values, expected_x) or not np.array_equal(dataset.y.values, expected_y):
            raise ValueError(f"Aligned motion x/y coordinates mismatch in {path.name}.")
        required = {"ice_motion_dx_24h", "ice_motion_dy_24h", "ice_motion_support_fraction"}
        if not required.issubset(dataset.data_vars):
            raise ValueError(f"Aligned motion file lacks required fields in {path.name}.")
        if dataset.attrs.get("decision_status") != "research_preprocessing_only":
            raise ValueError(f"Aligned motion file lacks the expected research-only marker in {path.name}.")
        return tuple(np.asarray(dataset[name].values, dtype=np.float32) for name in
                     ("ice_motion_dx_24h", "ice_motion_dy_24h", "ice_motion_support_fraction"))


def score(entries: list[tuple[date, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
          coefficient: float) -> dict:
    persistence_error = []
    blend_error = []
    per_date = []
    for issue, base, truth, advected, valid in entries:
        blend = np.clip(base + coefficient * (advected - base), 0.0, 1.0)
        p_error = truth[valid] - base[valid]
        b_error = truth[valid] - blend[valid]
        persistence_error.extend(p_error.tolist())
        blend_error.extend(b_error.tolist())
        if p_error.size:
            per_date.append({"issue_date": issue.isoformat(), "valid_cells": int(p_error.size),
                             "persistence_mae": float(np.mean(np.abs(p_error))),
                             "blend_mae": float(np.mean(np.abs(b_error)))})
    if not persistence_error:
        raise ValueError("No common valid cells were available to score this period.")
    p = np.asarray(persistence_error, dtype=np.float64)
    b = np.asarray(blend_error, dtype=np.float64)
    return {"n_cell_days": int(p.size), "evaluated_days": len(per_date),
            "persistence_mae": float(np.mean(np.abs(p))),
            "blend_mae": float(np.mean(np.abs(b))),
            "persistence_rmse": float(np.sqrt(np.mean(p ** 2))),
            "blend_rmse": float(np.sqrt(np.mean(b ** 2))),
            "mae_reduction_percent": float(100.0 * (1.0 - np.mean(np.abs(b)) / np.mean(np.abs(p)))) if np.mean(np.abs(p)) > 0 else None,
            "rmse_reduction_percent": float(100.0 * (1.0 - np.sqrt(np.mean(b ** 2)) / np.sqrt(np.mean(p ** 2)))) if np.sqrt(np.mean(p ** 2)) > 0 else None,
            "per_date": per_date}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True,
                        help="Verified NSIDC acquisition directory containing manifest.json and source files.")
    parser.add_argument("--aligned-dir", type=Path, required=True,
                        help="Directory of date-named aligned motion NetCDFs (YYYY-MM-DD.nc).")
    parser.add_argument("--sic-file", type=Path, required=True,
                        help="One checksummed G02202 annual file containing all issue and target dates.")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--fit-through", type=date.fromisoformat,
                        help="Last issue date used to select the convex motion blend coefficient.")
    parser.add_argument("--fixed-lambda", type=float,
                        help="Score a coefficient selected on an earlier, independent period; do not refit.")
    parser.add_argument("--lambda-report", type=Path,
                        help="Use the selected coefficient from a dated earlier hindcast report; record its SHA-256.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not be after --end")
    if args.fixed_lambda is not None and not 0.0 <= args.fixed_lambda <= 1.0:
        parser.error("--fixed-lambda must be in [0, 1]")
    if args.fixed_lambda is not None and args.fit_through is not None:
        parser.error("Use either --fixed-lambda or --fit-through, not both")
    if args.lambda_report is not None and (args.fixed_lambda is not None or args.fit_through is not None):
        parser.error("Use --lambda-report without --fixed-lambda or --fit-through")
    coefficient_source = None
    if args.lambda_report is not None:
        prior_report = json.loads(args.lambda_report.read_text(encoding="utf-8"))
        if prior_report.get("experiment") != "one-day lagged NSIDC-0116 motion-advection SIC hindcast":
            raise ValueError("Coefficient report is not an NSIDC-0116 advection hindcast.")
        try:
            prior_end = date.fromisoformat(prior_report["date_range"]["end_issue"])
            prior_lambda = float(prior_report["selected_lambda"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Coefficient report lacks a valid date range or selected lambda.") from exc
        if prior_end >= args.start or not np.isfinite(prior_lambda) or not 0.0 <= prior_lambda <= 1.0:
            raise ValueError("Coefficient report must end before the evaluation period and select lambda in [0, 1].")
        args.fixed_lambda = prior_lambda
        coefficient_source = {"file": args.lambda_report.name,
                              "sha256": sha256_file(args.lambda_report),
                              "source_last_issue": prior_end.isoformat()}
    manifest_path = args.motion_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_entries = _manifest_entries(manifest, args.start, args.end)
    source_manifest_hash = sha256_file(manifest_path)
    aligned_manifest_path = args.aligned_dir / "aligned-manifest.json"
    aligned_manifest = json.loads(aligned_manifest_path.read_text(encoding="utf-8"))
    if (aligned_manifest.get("dataset_id") != "SouthernPassage-NSIDC0116-aligned"
            or aligned_manifest.get("complete") is not True):
        raise ValueError("Aligned manifest must describe a complete Southern Passage NSIDC-0116 alignment.")
    sic_file_hash = sha256_file(args.sic_file)
    if aligned_manifest.get("sic_reference_sha256") != sic_file_hash:
        raise ValueError("Aligned manifest was created against a different SIC reference file.")
    if aligned_manifest.get("source_manifest_sha256") != source_manifest_hash:
        raise ValueError("Aligned manifest was created from a different motion acquisition manifest.")
    aligned_period = aligned_manifest.get("requested_period", {})
    if (aligned_period.get("start") is None or aligned_period.get("end") is None
            or date.fromisoformat(aligned_period["start"]) > args.start - timedelta(days=1)
            or date.fromisoformat(aligned_period["end"]) < args.end - timedelta(days=1)):
        raise ValueError("Aligned manifest does not cover all required one-day-lag motion dates.")
    aligned_entries = aligned_manifest.get("files")
    if not isinstance(aligned_entries, list):
        raise ValueError("Aligned manifest contains no file inventory.")
    aligned_by_date = {}
    for item in aligned_entries:
        if not isinstance(item, dict) or not isinstance(item.get("date"), str):
            continue
        if item["date"] in aligned_by_date:
            raise ValueError(f"Aligned manifest has a duplicate date: {item['date']}")
        aligned_by_date[item["date"]] = item
    with xr.open_dataset(args.sic_file, decode_cf=True) as sic:
        if not {"time", "x", "y"}.issubset(sic.coords):
            raise ValueError("SIC file must provide time/x/y coordinates.")
        pixel_x = float(np.median(np.abs(np.diff(sic.x.values))))
        pixel_y = float(np.median(np.abs(np.diff(sic.y.values))))
        x = np.asarray(sic.x.values).copy()
        y = np.asarray(sic.y.values).copy()
        shape = (int(sic.sizes["y"]), int(sic.sizes["x"]))
        if args.end + timedelta(days=1) > date.fromisoformat(str(sic.time.values[-1])[:10]):
            raise ValueError("SIC file does not contain the final next-day target date.")
        issues = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
        entries = []
        coverage = []
        verified_source_hashes: dict[str, str] = {}
        for issue in issues:
            motion_day = issue - timedelta(days=1)
            source_entry = motion_archive_entry(manifest_entries, motion_day)
            source_path = args.motion_dir / source_entry["file"]
            if source_path.name != source_entry["file"]:
                raise ValueError(f"Unsafe source motion filename in manifest: {source_entry['file']}")
            if source_entry["file"] not in verified_source_hashes:
                if sha256_file(source_path) != source_entry.get("sha256"):
                    raise ValueError(f"Source motion archive file failed manifest verification: {source_entry['file']}")
                verified_source_hashes[source_entry["file"]] = source_entry["sha256"]
            if verified_source_hashes[source_entry["file"]] != source_entry.get("sha256"):
                raise ValueError(f"Source motion archive file failed manifest verification: {source_entry['file']}")
            base, base_valid = read_sic(sic, issue)
            truth, truth_valid = read_sic(sic, issue + timedelta(days=1))
            aligned_entry = aligned_by_date.get(motion_day.isoformat())
            if not aligned_entry or aligned_entry.get("file") != f"{motion_day.isoformat()}.nc":
                raise ValueError(f"Aligned manifest has no expected file for motion date {motion_day}.")
            aligned_path = args.aligned_dir / aligned_entry["file"]
            if (not aligned_path.is_file()
                    or aligned_path.stat().st_size != aligned_entry.get("bytes")
                    or sha256_file(aligned_path) != aligned_entry.get("sha256")):
                raise ValueError(f"Aligned field failed its manifest integrity check: {aligned_path.name}")
            dx, dy, support = load_aligned_motion(aligned_path, source_entry, sic_file_hash,
                                                   motion_day, shape, x, y)
            advected, sampling_valid = backtrace_sic(base, dx, dy, support, pixel_x, pixel_y)
            valid = truth_valid & base_valid & sampling_valid & np.isfinite(advected)
            entries.append((issue, base, truth, advected, valid))
            coverage.append({"issue_date": issue.isoformat(), "motion_date": motion_day.isoformat(),
                             "valid_common_cells": int(valid.sum()), "grid_cells": int(valid.size),
                             "valid_fraction": float(valid.mean()),
                             "motion_source_file": source_entry["file"],
                             "motion_source_sha256": source_entry["sha256"],
                             "aligned_file_sha256": sha256_file(aligned_path)})
    split = args.fit_through or (args.start + timedelta(days=max(0, (args.end - args.start).days // 2)))
    if args.fixed_lambda is not None:
        selected_lambda = args.fixed_lambda
        fit_entries = []
        test_entries = entries
        fit_rule = ("fixed coefficient from earlier dated report; no fitting on evaluation dates"
                    if coefficient_source else
                    "fixed coefficient supplied by caller; no fitting on evaluation dates")
    else:
        fit_entries = [entry for entry in entries if entry[0] <= split]
        test_entries = [entry for entry in entries if entry[0] > split]
        if not fit_entries or not test_entries:
            parser.error("Chronological coefficient-fit and holdout dates must both be nonempty.")
        candidates = np.linspace(0.0, 1.0, 21)
        fit_scores = [score(fit_entries, float(candidate))["blend_mae"] for candidate in candidates]
        selected_lambda = float(candidates[int(np.argmin(fit_scores))])
        fit_rule = f"convex blend selected by minimum pooled fit-period MAE through {split.isoformat()}"
    fit_report = score(fit_entries, selected_lambda) if fit_entries else None
    test_report = score(test_entries, selected_lambda)
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    report = {
        "experiment": "one-day lagged NSIDC-0116 motion-advection SIC hindcast",
        "decision_status": "research_only_not_navigation_or_operational_forecast",
        "dataset": "NOAA/NSIDC G02202 v6 SIC + NSIDC-0116 v4 daily sea-ice motion",
        "sic_file": args.sic_file.name,
        "sic_file_sha256": sic_file_hash,
        "motion_manifest_sha256": source_manifest_hash,
        "aligned_manifest_sha256": sha256_file(aligned_manifest_path),
        "date_range": {"start_issue": args.start.isoformat(), "end_issue": args.end.isoformat(),
                       "last_target": (args.end + timedelta(days=1)).isoformat()},
        "motion_timing_assumption": "For issue date D, use aligned NSIDC motion record D-1. The date label and daily product description do not establish publication/availability time or strict causal availability; this is retrospective research, not operational forecast validation.",
        "quality_and_sampling": "G02202 QA excludes no-input, invalid-mask, spatial interpolation, and temporal interpolation cells. Motion fill/error values are screened upstream; the aligned support field must be at least 0.75. Scores use common valid SIC/motion cells only.",
        "aggregation": "Pooled equally across common-valid grid-cell samples; these scores are not ground-area weighted, route-weighted, or adjusted for spatial/temporal correlation.",
        "blend_formula": "clip(persistence + lambda * (lagged_motion_advected_SIC - persistence), 0, 1)",
        "coefficient_selection": fit_rule,
        "coefficient_source_report": coefficient_source,
        "selected_lambda": selected_lambda,
        "fit_metrics": fit_report,
        "holdout_metrics": test_report,
        "coverage_by_issue_date": coverage,
        "limitations": [
            "Same-product retrospective hindcast; not independent sensor validation.",
            "Grid-cell samples are spatially and temporally correlated; sample count is not an independent-trial count.",
            "Lagged ice motion may not represent the next-day displacement; no uncertainty calibration is provided.",
            "Sea-ice concentration and motion do not establish vessel passability, encounter risk, or route safety.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_sha256": sha256_file(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"selected_lambda": selected_lambda,
                      "holdout_mae_reduction_percent": test_report["mae_reduction_percent"],
                      "holdout_rmse_reduction_percent": test_report["rmse_reduction_percent"],
                      "holdout_cell_days": test_report["n_cell_days"],
                      "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NSIDC advection evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
