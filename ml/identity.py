"""Stable identity helpers for research model artifacts and evaluation reports."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def model_fingerprint(model: dict[str, Any]) -> str:
    """Fingerprint the model family and exact coefficient vector (not metadata)."""
    identity = {"model_id": model.get("model_id"),
                "coefficients": [float(value) for value in model.get("coefficients", [])]}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_artifact_sha256(model: dict[str, Any]) -> str:
    """Digest the complete model artifact, including its provenance and evidence."""
    artifact = {key: value for key, value in model.items() if key != "artifact_sha256"}
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
