"""Build a checksummed, date-complete aligned NSIDC-0116 archive."""
from __future__ import annotations

import argparse
from datetime import date, timedelta, datetime, timezone
import json
from pathlib import Path
import tempfile

if __package__:
    from .align_nsidc0116 import align_nsidc0116_file, sha256_file
else:  # Support direct execution from the repository root.
    from align_nsidc0116 import align_nsidc0116_file, sha256_file


def _source_entry(entries: list[dict], day: date) -> dict:
    matches = []
    for entry in entries:
        try:
            first = date.fromisoformat(entry["time_start"])
            last = date.fromisoformat(entry["time_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if first <= day <= last:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest file covering {day}; found {len(matches)}.")
    return matches[0]


def align_archive(motion_dir: Path, sic_reference: Path, output_dir: Path,
                  start: date, end: date) -> dict:
    motion_dir = Path(motion_dir).resolve()
    sic_reference = Path(sic_reference).resolve()
    output_dir = Path(output_dir).resolve()
    if start > end:
        raise ValueError("start must not be after end.")
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite existing aligned archive: {output_dir}")
    source_manifest_path = motion_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_id") != "NSIDC-0116" or source_manifest.get("complete") is not True:
        raise ValueError("Source manifest must describe a complete NSIDC-0116 acquisition.")
    try:
        acquired_start = date.fromisoformat(source_manifest["requested_period"]["start"])
        acquired_end = date.fromisoformat(source_manifest["requested_period"]["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Source manifest has no valid requested period.") from exc
    if acquired_start > start or acquired_end < end:
        raise ValueError("Source manifest does not cover every requested motion date.")
    source_entries = source_manifest.get("files")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("Source manifest contains no file inventory.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    sic_hash = sha256_file(sic_reference)
    source_hashes: dict[str, str] = {}
    aligned_entries = []
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.staging-",
                                     dir=output_dir.parent) as temp_name:
        staging = Path(temp_name)
        for offset in range((end - start).days + 1):
            day = start + timedelta(days=offset)
            source = _source_entry(source_entries, day)
            filename = source.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Source manifest contains an unsafe filename.")
            source_path = motion_dir / filename
            if filename not in source_hashes:
                actual_hash = sha256_file(source_path)
                if actual_hash != source.get("sha256"):
                    raise ValueError(f"Source file failed manifest verification: {filename}")
                source_hashes[filename] = actual_hash
            target = staging / f"{day.isoformat()}.nc"
            aligned = align_nsidc0116_file(source_path, sic_reference, day,
                                            motion_sha256=source_hashes[filename],
                                            sic_reference_sha256=sic_hash)
            aligned.to_netcdf(target)
            aligned_entries.append({"date": day.isoformat(), "file": target.name,
                                    "bytes": target.stat().st_size,
                                    "sha256": sha256_file(target),
                                    "source_motion_file": filename,
                                    "source_motion_sha256": source_hashes[filename]})
        manifest = {
            "dataset_id": "SouthernPassage-NSIDC0116-aligned",
            "source_dataset_id": "NSIDC-0116",
            "source_dataset_version": source_manifest.get("dataset_version"),
            "source_doi": source_manifest.get("doi"),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "sic_reference_file": sic_reference.name,
            "sic_reference_sha256": sic_hash,
            "requested_period": {"start": start.isoformat(), "end": end.isoformat()},
            "complete": True,
            "completeness_note": "Every requested date was aligned and checksummed. This does not assert that each cell has valid motion; inspect support_fraction for each field.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": aligned_entries,
        }
        (staging / "aligned-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    return {"complete": True, "date_count": len(aligned_entries),
            "aligned_manifest": str(output_dir / "aligned-manifest.json"),
            "sic_reference_sha256": sic_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_dir", type=Path)
    parser.add_argument("sic_reference_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    try:
        result = align_archive(args.motion_dir, args.sic_reference_file,
                               args.output_dir, args.start, args.end)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        parser.exit(1, f"NSIDC archive alignment failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
