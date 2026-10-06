"""Summarize released Ice-kNN-South verification metrics by forecast lead."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import netCDF4
import numpy as np

DOI = "10.5281/zenodo.13144175"
MODELS = {
    "Persistence_1979-2024": "persistence",
    "Climatology_1979-2024": "climatology",
    "Ice-kNN-DCT=18_10_0.2_1979-2024": "Ice-kNN-South",
}
LEADS = (1, 7, 30, 60, 90)
PERIOD_KEY = "2023-2024"


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def summarize_arrays(verification: dict[str, dict[str, np.ndarray]], leads: tuple[int, ...] = LEADS) -> dict[str, object]:
    """Summarize archive arrays; lead indices are one-based days from initialization."""
    summary: dict[str, object] = {}
    reference_count = None
    for folder, label in MODELS.items():
        if folder not in verification:
            raise ValueError(f"Missing archived verification for {folder}")
        fields = verification[folder]
        iiee = np.asarray(fields["iiee"], dtype=np.float64)
        sie_pred = np.asarray(fields["sie_pred"], dtype=np.float64)
        sie_obs = np.asarray(fields["sie_obs"], dtype=np.float64)
        if iiee.ndim != 2 or sie_pred.shape != iiee.shape or sie_obs.shape != iiee.shape:
            raise ValueError(f"Invalid IIEE/SIE error dimensions for {folder}")
        sie_error = np.abs(sie_pred - sie_obs)
        if reference_count is None:
            reference_count = iiee.shape[0]
        elif iiee.shape[0] != reference_count:
            raise ValueError("Model verification arrays have different initialization counts")
        metrics = {}
        for lead in leads:
            if not 1 <= lead <= iiee.shape[1]:
                raise ValueError(f"Lead day {lead} exceeds available archive horizon")
            edge_values = iiee[:, lead - 1]
            area_values = sie_error[:, lead - 1]
            edge_values = edge_values[np.isfinite(edge_values)]
            area_values = area_values[np.isfinite(area_values)]
            metrics[str(lead)] = {
                "iiee_sample_count": int(len(edge_values)),
                "sie_error_sample_count": int(len(area_values)),
                "mean_iiee_million_km2": float(edge_values.mean()) if len(edge_values) else None,
                "mean_absolute_sie_error_million_km2": float(area_values.mean()) if len(area_values) else None,
            }
        summary[label] = metrics
    return {"initialization_count": int(reference_count or 0), "lead_days": list(leads), "models": summary}


def read_netcdf_member(archive: zipfile.ZipFile, member: str) -> dict[str, np.ndarray]:
    dataset = netCDF4.Dataset(member, mode="r", memory=archive.read(member))
    try:
        return {name.lower(): np.ma.filled(np.ma.asarray(dataset.variables[name][:]), np.nan)
                for name in ("IIEE", "SIE_pred", "SIE_obs")}
    finally:
        dataset.close()


def build_report(archive_path: Path) -> dict[str, object]:
    manifest_path = archive_path.with_name("manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("The Zenodo download manifest is missing or invalid.") from exc
    if (manifest.get("doi") != DOI or manifest.get("file") != archive_path.name
            or manifest.get("license") != "CC-BY-4.0"):
        raise ValueError("The archive manifest does not identify the expected CC-BY-4.0 Zenodo artifact.")
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError("Archive SHA-256 does not match its acquisition manifest.")

    with zipfile.ZipFile(archive_path) as archive:
        target_member = f"Ice-kNN-DCT=18_10_0.2_1979-2024/target_time_{PERIOD_KEY}.nc"
        time_dataset = netCDF4.Dataset(target_member, mode="r", memory=archive.read(target_member))
        try:
            input_times = [decode_text(item) for item in time_dataset.variables["input_time"][:]]
            target_times = time_dataset.variables["target_time"][:]
            target_dates = [decode_text(item) for item in target_times.reshape(-1)]
        finally:
            time_dataset.close()
        arrays = {folder: read_netcdf_member(archive, f"{folder}/verification_{PERIOD_KEY}.nc")
                  for folder in MODELS}
    stats = summarize_arrays(arrays)
    return {
        "source": "Lin et al. Ice-kNN-South released verification sample",
        "doi": DOI,
        "archive_sha256": digest,
        "evaluation_scope": "Authors' archived 2023-2024 verification statistics; this report summarizes aggregate outputs and does not rerun the model or independently reconstruct its SIC fields.",
        "forecast_initialization_dates": {"first": input_times[0], "last": input_times[-1]},
        "verified_target_dates": {"first": min(target_dates), "last": max(target_dates)},
        "units_note": "IIEE and absolute SIE error are reported in million km^2, matching the archive verification script's area scaling.",
        **stats,
        "warning": "Pan-Antarctic archived verification only. No regional route-scale SIC forecasts, uncertainty, vessel hazards, or navigation advice are included.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="model_verification_specific_year.zip acquired by ml/fetch_iceknn_verification.py")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = build_report(args.archive)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
