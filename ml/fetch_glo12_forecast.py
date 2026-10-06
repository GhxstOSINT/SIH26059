"""Capture original GLO12 forecast files for 1/3/5/7-day verification.

Requires the official ``copernicusmarine`` package and an existing Copernicus
Marine login. The provider's native files, not replaceable historical subsets,
are saved with their bulletin dates and checksummed manifests.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re

from ml.archive_glo12_forecast import DATASET_ID, DATASET_VERSION, inventory_forecast


DEFAULT_LEADS = (1, 3, 5, 7)


def expected_names(run_date: date, leads: tuple[int, ...]) -> set[str]:
    from datetime import timedelta

    if not leads or any(day < 1 or day > 10 for day in leads) or len(set(leads)) != len(leads):
        raise ValueError("Choose distinct forecast leads between 1 and 10 days")
    return {
        f"glo12_rg_1d-m_{(run_date + timedelta(days=lead)):%Y%m%d}-"
        f"{(run_date + timedelta(days=lead)):%Y%m%d}_2D_fcst_R{run_date:%Y%m%d}.nc"
        for lead in leads
    }


def fetch_run(run_date: date, output_dir: Path, leads: tuple[int, ...] = DEFAULT_LEADS) -> list[dict]:
    import copernicusmarine

    names = expected_names(run_date, leads)
    if run_date > datetime.now(timezone.utc).date():
        raise ValueError("Cannot fetch a bulletin from a future date")
    if output_dir.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    # A complete local bulletin is immutable evidence. Revalidate it without
    # requiring provider authentication on every status/retry invocation.
    local = []
    for name in sorted(names):
        source = output_dir / name
        manifest_path = output_dir / f"{source.stem}.manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Forecast manifest may not be a symlink")
        if not manifest_path.is_file():
            break
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = inventory_forecast(
            source,
            captured_at=datetime.fromisoformat(existing["captured_at_utc"]),
            provider_last_modified_at=(datetime.fromisoformat(existing["provider_last_modified_at_utc"])
                                       if existing.get("provider_last_modified_at_utc") else None),
        )
        if existing != expected:
            raise ValueError("Existing forecast manifest conflicts with the file; refusing to replace evidence")
        local.append(existing)
    if len(local) == len(names):
        return local
    alternatives = "|".join(re.escape(name) for name in sorted(names))
    response = copernicusmarine.get(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        regex=rf".*(?:{alternatives})$",
        output_directory=output_dir,
        no_directories=True,
        skip_existing=True,
        disable_progress_bar=True,
    )
    returned = {entry.filename: entry for entry in response.files}
    if set(returned) != names:
        raise ValueError(f"Provider returned {len(returned)} of {len(names)} expected original forecast files")
    manifests = []
    root = output_dir.resolve()
    for name in sorted(names):
        entry = returned[name]
        source = Path(entry.file_path).resolve()
        if source.parent != root or not source.is_file():
            raise ValueError("Downloaded forecast file is outside the requested archive")
        manifest_path = root / f"{source.stem}.manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Forecast manifest may not be a symlink")
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                expected = inventory_forecast(
                    source,
                    captured_at=datetime.fromisoformat(existing["captured_at_utc"]),
                    provider_last_modified_at=(datetime.fromisoformat(existing["provider_last_modified_at_utc"])
                                               if existing.get("provider_last_modified_at_utc") else None),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Existing forecast manifest is invalid") from exc
            if existing != expected:
                raise ValueError("Existing forecast manifest conflicts with the file; refusing to replace evidence")
            manifests.append(existing)
            continue
        report = inventory_forecast(
            source,
            provider_last_modified_at=datetime.fromisoformat(entry.last_modified_datetime),
        )
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        manifests.append(report)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date())
    parser.add_argument("--leads", type=int, nargs="+", default=DEFAULT_LEADS)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("work/datasets/copernicus-glo12-forecast"))
    args = parser.parse_args()
    reports = fetch_run(args.run_date, args.output_dir, tuple(args.leads))
    print(json.dumps({"bulletin_date": args.run_date.isoformat(),
                      "archived_leads": [report["lead_days"] for report in reports],
                      "prospectively_captured": all(report["eligible_for_prospective_verification"]
                                                    for report in reports),
                      "uncertainty_calibrated": False}, indent=2))


if __name__ == "__main__":
    main()
