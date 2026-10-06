"""Fetch the small CC-BY-4.0 Ice-kNN-South published verification archive.

This is an archived forecast-result sample, not the model's training or
inference implementation. The much larger 2.56 GB full verification archive
is intentionally not downloaded by this helper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

DOI = "10.5281/zenodo.13144175"
RECORD_ID = "13144175"
FILE_NAME = "model_verification_specific_year.zip"
FILE_SIZE = 4_498_865
FILE_MD5 = "b967db4ffb87c0fc3609e865286a7ba5"
FILE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files/{FILE_NAME}/content"
CHUNK_SIZE = 1024 * 1024


def verify(path: Path) -> tuple[str, str]:
    md5, sha256 = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    if size != FILE_SIZE or md5.hexdigest() != FILE_MD5:
        raise ValueError(f"Published verification failed for {path.name}; expected {FILE_SIZE} bytes and MD5 {FILE_MD5}.")
    return sha256.hexdigest(), md5.hexdigest()


def acquire(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / FILE_NAME
    if target.exists():
        sha256, md5 = verify(target)
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=FILE_NAME + ".", suffix=".part", dir=output_dir)
        temporary = Path(temporary_name)
        md5, sha256, size = hashlib.md5(usedforsecurity=False), hashlib.sha256(), 0
        try:
            request = urllib.request.Request(FILE_URL, headers={"User-Agent": "SouthernPassage-research/0.1"})
            with os.fdopen(descriptor, "wb") as sink:
                with urllib.request.urlopen(request, timeout=180) as response:
                    while chunk := response.read(CHUNK_SIZE):
                        size += len(chunk)
                        if size > FILE_SIZE:
                            raise ValueError("Zenodo response exceeded the published file size; refusing the response.")
                        sink.write(chunk)
                        md5.update(chunk)
                        sha256.update(chunk)
                sink.flush()
                os.fsync(sink.fileno())
            if size != FILE_SIZE or md5.hexdigest() != FILE_MD5:
                raise ValueError("Zenodo archive size or published MD5 checksum did not match.")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        sha256, md5 = sha256.hexdigest(), md5.hexdigest()

    manifest = {
        "dataset": "Ice-kNN-South published verification sample",
        "doi": DOI,
        "zenodo_record": f"https://zenodo.org/records/{RECORD_ID}",
        "license": "CC-BY-4.0",
        "file": FILE_NAME,
        "bytes": FILE_SIZE,
        "md5": md5,
        "sha256": sha256,
        "accessed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Published verification outputs for an archived year; not weights, training code, or live forecasts.",
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Dedicated directory for this small published archive")
    args = parser.parse_args()
    print(json.dumps(acquire(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
