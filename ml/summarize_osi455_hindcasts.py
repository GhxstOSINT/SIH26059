"""Summarize frozen-coefficient OSI-455 hindcast outputs by year."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(fit: dict, evaluations: list[dict]) -> dict:
    if not evaluations:
        raise ValueError("At least one independent evaluation output is required")
    coefficient = fit.get("selected_lambda")
    if not isinstance(coefficient, (int, float)):
        raise ValueError("Fit output must contain a selected_lambda")
    fit_period = fit.get("period", {})
    fit_year = int(fit_period.get("start", "0000")[:4])
    rows = []
    seen_years: set[int] = set()
    for result in evaluations:
        if result.get("selected_lambda") != coefficient or result.get("train") is not None:
            raise ValueError("Evaluation outputs must use the frozen fit coefficient without refitting")
        period = result.get("period", {})
        year = int(period.get("start", "0000")[:4])
        if year <= fit_year or int(period.get("end", "0000")[:4]) != year or year in seen_years:
            raise ValueError("Evaluation periods must be unique full years later than the fit period")
        metrics = result.get("holdout", {}).get("pooled", {})
        count = int(metrics.get("n_cells", 0))
        if count <= 0:
            raise ValueError(f"Evaluation year {year} has no scored cells")
        seen_years.add(year)
        row = {"year": year, "issue_period": period, "n_correlated_cell_days": count,
               "persistence_mae": float(metrics["persistence_mae"]),
               "advection_mae": float(metrics["advection_mae"]),
               "persistence_rmse": float(metrics["persistence_rmse"]),
               "advection_rmse": float(metrics["advection_rmse"])}
        row["mae_reduction_percent"] = 100 * (1 - row["advection_mae"] / row["persistence_mae"])
        row["rmse_reduction_percent"] = 100 * (1 - row["advection_rmse"] / row["persistence_rmse"])
        row["motion_coverage"] = result.get("motion_coverage")
        row["geographic_strata"] = result.get("holdout", {}).get("geographic_strata")
        row["provenance"] = result.get("provenance", {})
        rows.append(row)
    rows.sort(key=lambda row: row["year"])
    total = sum(row["n_correlated_cell_days"] for row in rows)
    pooled = {
        "n_correlated_cell_days": total,
        "persistence_mae": sum(row["n_correlated_cell_days"] * row["persistence_mae"] for row in rows) / total,
        "advection_mae": sum(row["n_correlated_cell_days"] * row["advection_mae"] for row in rows) / total,
        "persistence_rmse": (sum(row["n_correlated_cell_days"] * row["persistence_rmse"] ** 2 for row in rows) / total) ** 0.5,
        "advection_rmse": (sum(row["n_correlated_cell_days"] * row["advection_rmse"] ** 2 for row in rows) / total) ** 0.5,
        "equal_year_mean_mae_reduction_percent": sum(row["mae_reduction_percent"] for row in rows) / len(rows),
        "equal_year_mean_rmse_reduction_percent": sum(row["rmse_reduction_percent"] for row in rows) / len(rows),
    }
    pooled["mae_reduction_percent"] = 100 * (1 - pooled["advection_mae"] / pooled["persistence_mae"])
    pooled["rmse_reduction_percent"] = 100 * (1 - pooled["advection_rmse"] / pooled["persistence_rmse"])
    return {
        "experiment": "OSI-455 one-day SIC advection with a frozen chronological-fit coefficient and date-label timing convention",
        "fit": {"year": fit_year, "selected_lambda": coefficient,
                "fit_output_provenance": fit.get("provenance", {}),
                "split": fit.get("protocol", {}).get("split")},
        "evaluations": rows,
        "combined": pooled,
        "interpretation": [
            "Grid-cell-day observations are spatially and temporally correlated, not independent trials.",
            "Full-year date ranges do not guarantee full-season motion coverage; consult strict-quality coverage by month.",
            "The motion product is 75 km and precedes the experiment's assumed 00Z date label by 12 hours; the daily SIC field's real-time availability is unverified.",
            "This evaluates SIC concentration only, not route clearance, ship safety, iceberg drift, or encounter risk.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit = json.loads(args.fit_output.read_text(encoding="utf-8"))
    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in args.evaluation_output]
    result = summarize(fit, evaluations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["combined"], indent=2))


if __name__ == "__main__":
    main()
