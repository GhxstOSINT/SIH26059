"""Fail-closed status for the separately archived, as-issued GLO12 bulletins."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from ml.archive_glo12_forecast import inventory_forecast


def forecast_archive_status(root: Path) -> dict:
    result = {
        "source": "Copernicus Marine GLO12 original forecast files",
        "archive_state": "unavailable",
        "latest_bulletin_date": None,
        "verified_leads": [],
        "rejected_files": 0,
        "forecast_skill_verified": False,
        "uncertainty_calibrated": False,
        "operator_decision_ready": False,
        "navigation_clearance": False,
        "reason": "No verified as-issued forecast files are available.",
    }
    if not root.is_dir() or root.is_symlink():
        return result
    verified = []
    for manifest_path in sorted(root.glob("*.json")):
        try:
            if manifest_path.is_symlink():
                raise ValueError("Symlinked manifest")
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_name = recorded.get("source_file")
            if not isinstance(source_name, str) or Path(source_name).name != source_name:
                raise ValueError("Invalid source name")
            source = root / source_name
            expected = inventory_forecast(
                source,
                captured_at=datetime.fromisoformat(recorded["captured_at_utc"]),
                provider_last_modified_at=(
                    datetime.fromisoformat(recorded["provider_last_modified_at_utc"])
                    if recorded.get("provider_last_modified_at_utc") else None
                ),
            )
            if recorded != expected or not recorded["eligible_for_prospective_verification"]:
                raise ValueError("Manifest or capture provenance mismatch")
            verified.append(recorded)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            result["rejected_files"] += 1
    if not verified:
        if result["rejected_files"]:
            result["reason"] = "Archive files failed provenance or integrity checks."
        return result
    latest = max(item["bulletin_date"] for item in verified)
    result.update({
        "archive_state": "verified_original_files",
        "latest_bulletin_date": latest,
        "verified_leads": sorted({item["lead_days"] for item in verified if item["bulletin_date"] == latest}),
        "reason": "Forecast files are authenticated locally; future observations and independent calibration are pending.",
    })
    return result
