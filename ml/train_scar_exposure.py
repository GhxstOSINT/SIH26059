"""Fit a bounded historical iceberg-sighting climatology from the SCAR archive.

This is an observation-conditional research model, not a current hazard forecast.
The first release uses NPI rows only; AAD's different size protocol is reserved
for a separate harmonization and external-transfer study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SOURCE_DOI = "https://doi.org/10.21334/npolar.2021.e4b9a604"
NPI_NAME = "NPI_Iceberg_database_1977-2010_version_2022-05-20.csv"
INNER_NAME = "SCAR Iceberg database 2023-05-22.zip"
TRAIN_CUTOFF = 2000
LAT_STEP = 5
LON_STEP = 15
CELL_PRIOR = 30
PARENT_PRIOR = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npi(path: Path) -> tuple[list[dict], Counter]:
    rejected = Counter()
    records = []
    with zipfile.ZipFile(path) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read(INNER_NAME))) as inner:
            raw = inner.read(NPI_NAME).decode("utf-8-sig", errors="replace")
    for row in csv.reader(io.StringIO(raw), delimiter=";"):
        if row and row[0] == "ID":
            continue
        if len(row) < 20:
            rejected["short_row"] += 1
            continue
        try:
            date = datetime.strptime(row[4].strip(), "%d.%m.%Y %H:%M")
            lat, lon = float(row[5]), float(row[6])
            total = int(row[9])
            if not (-80 <= lat <= -40 and -180 <= lon <= 180 and total >= 0):
                raise ValueError("outside supported range")
            cruise = row[1].strip()
            if not cruise:
                raise ValueError("missing cruise")
        except (ValueError, IndexError):
            rejected["invalid_fields"] += 1
            continue
        season = (date.month - 1) // 3
        lat_bin = math.floor(lat / LAT_STEP) * LAT_STEP
        lon_bin = math.floor((lon + 180) / LON_STEP) * LON_STEP - 180
        lon_bin = min(lon_bin, 165)
        records.append(
            {
                "year": date.year,
                "cruise": cruise,
                "parent": (lat_bin, season),
                "cell": (lat_bin, lon_bin, season),
                "positive": int(total > 0),
            }
        )
    return records, rejected


def fit(records: list[dict]) -> dict:
    global_counts = [0, 0]
    parents = defaultdict(lambda: [0, 0])
    cells = defaultdict(lambda: [0, 0])
    for record in records:
        y = record["positive"]
        global_counts[0] += y
        global_counts[1] += 1
        parents[record["parent"]][0] += y
        parents[record["parent"]][1] += 1
        cells[record["cell"]][0] += y
        cells[record["cell"]][1] += 1
    global_rate = (global_counts[0] + 0.5) / (global_counts[1] + 1)
    return {"global": global_rate, "parents": dict(parents), "cells": dict(cells)}


def predict(model: dict, record: dict, use_cell: bool = True) -> tuple[float, int]:
    parent_yes, parent_n = model["parents"].get(record["parent"], (0, 0))
    parent_rate = (parent_yes + PARENT_PRIOR * model["global"]) / (parent_n + PARENT_PRIOR)
    if not use_cell:
        return parent_rate, parent_n
    cell_yes, cell_n = model["cells"].get(record["cell"], (0, 0))
    rate = (cell_yes + CELL_PRIOR * parent_rate) / (cell_n + CELL_PRIOR)
    return rate, cell_n


def score(records: list[dict], model: dict, mode: str) -> dict:
    if not records:
        return {"observations": 0, "brier": None, "log_loss": None}
    brier = log_loss = 0.0
    supported = 0
    for record in records:
        y = record["positive"]
        if mode == "global":
            p = model["global"]
        else:
            p, n = predict(model, record, use_cell=(mode == "cell"))
            supported += int(n >= 10)
        p = min(max(p, 1e-9), 1 - 1e-9)
        brier += (p - y) ** 2
        log_loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return {
        "observations": len(records),
        "observed_positive_fraction": sum(r["positive"] for r in records) / len(records),
        "brier": brier / len(records),
        "log_loss": log_loss / len(records),
        "at_least_10_training_observations_fraction": supported / len(records) if mode != "global" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    rows, rejected = load_npi(args.archive)
    test = [r for r in rows if r["year"] >= TRAIN_CUTOFF]
    holdout_cruises = {r["cruise"] for r in test}
    train = [r for r in rows if r["year"] < TRAIN_CUTOFF and r["cruise"] not in holdout_cruises]
    if not train or not test:
        raise SystemExit("Need both pre-2000 training and 2000+ holdout observations")
    model = fit(train)
    result = {
        "model": "scar-npi-observation-conditional-climatology-v1",
        "decision_status": "research_only",
        "source": {"doi": SOURCE_DOI, "archive_sha256": sha256(args.archive), "file": NPI_NAME},
        "method": {
            "target": "at least one iceberg recorded during a ship observation",
            "train_years": f"{min(r['year'] for r in train)}-{TRAIN_CUTOFF - 1}",
            "holdout_years": f"{TRAIN_CUTOFF}-{max(r['year'] for r in test)}",
            "latitude_bin_degrees": LAT_STEP,
            "longitude_bin_degrees": LON_STEP,
            "calendar_season_months": 3,
            "cell_prior_observations": CELL_PRIOR,
            "parent_prior_observations": PARENT_PRIOR,
            "training_rows": len(train),
            "holdout_rows": len(test),
            "training_cruises": len({r['cruise'] for r in train}),
            "holdout_cruises": len({r['cruise'] for r in test}),
            "cruise_ids_in_both_periods": len({r['cruise'] for r in train} & holdout_cruises),
            "earlier_rows_excluded_for_cruise_disjoint_holdout": sum(
                r["year"] < TRAIN_CUTOFF and r["cruise"] in holdout_cruises for r in rows
            ),
            "rejected_rows": dict(rejected),
            "aad_files": "not fitted; separate protocols and size classes require harmonization",
        },
        "holdout": {
            "global_rate_baseline": score(test, model, "global"),
            "latitude_season_baseline": score(test, model, "parent"),
            "spatial_season_climatology": score(test, model, "cell"),
        },
        "limitations": [
            "Historical ship-observation conditional rate, not current iceberg presence or encounter probability for a vessel.",
            "Ship routes, observation range, visibility, and observation frequency vary; row counts are not a uniform spatial survey.",
            "Nearby observations and repeat voyages are correlated; this holdout is chronological but not fully independent by route.",
            "No current meteorology, currents, ice motion, satellite detections, or vessel-specific constraints are used.",
            "No estimate may be displayed as navigation clearance or a live iceberg forecast.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": result["model"],
        "decision_status": result["decision_status"],
        "source": result["source"],
        "method": result["method"],
        "global_rate": model["global"],
        "parent_counts": {",".join(map(str, key)): value for key, value in sorted(model["parents"].items())},
        "cell_counts": {",".join(map(str, key)): value for key, value in sorted(model["cells"].items())},
        "limitations": result["limitations"],
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    result["artifact_sha256"] = sha256(args.model_output)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model_output": str(args.model_output), "holdout": result["holdout"]}, indent=2))


if __name__ == "__main__":
    main()
