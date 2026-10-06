"""Fail-closed verification of the initial, research-only SIC model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.identity import model_artifact_sha256, model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config" / "initial-model.json"


def _verified_json(root: Path, entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Release entry must be a relative path inside the project")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release entry resolves outside the project")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["file_sha256"]:
        raise ValueError(f"Release file digest mismatch: {relative.as_posix()}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Release entry is not a JSON object")
    return path, payload


def verify_initial_model(
    *, root: Path = ROOT, manifest_path: Path = DEFAULT_MANIFEST,
    runtime_model_path: Path | None = None,
) -> dict[str, Any]:
    """Verify model bytes, embedded identity, evidence binding and runtime selection."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("scope") != "research_only"
            or manifest.get("forecast_horizon_days") != 1
            or manifest.get("source_dataset_id") != "G02202"
            or manifest.get("operational_decision_ready") is not False):
        raise ValueError("Invalid initial-model release scope")
    model_path, model = _verified_json(root, manifest["artifact"])
    if runtime_model_path is not None and runtime_model_path.resolve() != model_path:
        raise ValueError("Runtime model differs from the selected initial artifact")
    if (model.get("model_id") != manifest["selection"]
            or model.get("model_fingerprint") != manifest["artifact"]["model_fingerprint"]
            or model.get("model_fingerprint") != model_fingerprint(model)
            or model.get("artifact_sha256") != manifest["artifact"]["artifact_sha256"]
            or model.get("artifact_sha256") != model_artifact_sha256(model)
            or (model.get("source_manifest") or {}).get("dataset_id") != "G02202"):
        raise ValueError("Selected model identity or provenance mismatch")
    _, holdout = _verified_json(root, manifest["evidence"]["post_training_holdout"])
    if (holdout.get("model_id") != model["model_id"]
            or holdout.get("model_fingerprint") != model["model_fingerprint"]
            or (holdout.get("period") or {}).get("first_target_date") != "2017-01-03"
            or (holdout.get("period") or {}).get("last_target_date") != "2024-12-31"):
        raise ValueError("Frozen holdout does not match the selected model")
    _, comparison = _verified_json(root, manifest["evidence"]["candidate_comparison"])
    if (comparison.get("decision") != "retain_current_model"
            or (comparison.get("models", {}).get("current") or {}).get("fingerprint") != model["model_fingerprint"]
            or comparison.get("candidate_change_percent", {}).get("rmse_reduction_vs_current", 100) >= 1):
        raise ValueError("Candidate decision does not support the current selection")
    return {
        "verified": True,
        "scope": "research_only",
        "model_id": model["model_id"],
        "model_fingerprint": model["model_fingerprint"],
        "artifact_sha256": model["artifact_sha256"],
        "forecast_horizon_days": 1,
        "source_dataset_id": "G02202",
        "candidate_decision": comparison["decision"],
        "operational_decision_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = verify_initial_model(root=args.root, manifest_path=args.manifest)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "reason": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
