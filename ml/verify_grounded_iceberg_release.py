"""Verify local integrity of the pinned grounded-iceberg release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, load_checkpoint: bool = True) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for asset in manifest["assets"]:
        path = root / asset["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing release asset: {path}")
        actual_hash = sha256_file(path)
        if actual_hash.lower() != asset["sha256"].lower():
            raise ValueError(f"SHA-256 mismatch for {path.name}: {actual_hash}")
        results.append({"file": path.name, "bytes": path.stat().st_size, "sha256": actual_hash})

    checkpoint_summary = None
    if load_checkpoint:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Checkpoint verification requires PyTorch") from exc
        checkpoint_path = root / "resunet_v1.0.0.pth"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("Checkpoint did not load as a non-empty state dictionary")
        checkpoint_summary = {
            "tensor_count": len(state),
            "parameter_count": sum(value.numel() for value in state.values() if hasattr(value, "numel")),
            "first_keys": list(state)[:6],
        }

    smoke = manifest.get("inference_smoke_test")
    smoke_output_check = None
    if isinstance(smoke, dict) and smoke.get("completed") is True:
        output_path = root / smoke["output_file"]
        if output_path.is_file():
            actual_hash = sha256_file(output_path)
            if actual_hash.lower() != smoke["output_sha256"].lower():
                raise ValueError(f"SHA-256 mismatch for smoke-test output {output_path.name}: {actual_hash}")
            if output_path.stat().st_size != smoke["output_bytes"]:
                raise ValueError(f"Byte-size mismatch for smoke-test output {output_path.name}")
            smoke_output_check = {
                "file": smoke["output_file"],
                "bytes": output_path.stat().st_size,
                "sha256": actual_hash,
            }

    return {
        "release": manifest["release_tag"],
        "asset_checks": results,
        "checkpoint_safe_load": checkpoint_summary,
        "smoke_test_output_integrity": smoke_output_check,
        "scope": "Artifact integrity/loadability only; this does not validate segmentation accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("work/datasets/grounded-iceberg-demo"),
        help="Directory containing manifest.json and release assets",
    )
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Check file hashes without importing/loading PyTorch",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root, load_checkpoint=not args.skip_checkpoint_load), indent=2))


if __name__ == "__main__":
    main()
