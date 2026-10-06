import pytest

from ml.summarize_osi455_hindcasts import summarize


def result(year, count, persistence_mae, advection_mae, lam=0.1):
    return {
        "period": {"start": f"{year}-09-01", "end": f"{year}-09-30"},
        "selected_lambda": lam,
        "train": None,
        "motion_coverage": {"days_with_strict_motion": year},
        "provenance": {"motion_manifest_sha256": f"hash-{year}"},
        "holdout": {"geographic_strata": {"longitude_000_090E": {"n_cells": count}},
                    "pooled": {"n_cells": count,
                                "persistence_mae": persistence_mae,
                                "advection_mae": advection_mae,
                                "persistence_rmse": persistence_mae * 2,
                                "advection_rmse": advection_mae * 2}},
    }


def test_summarize_uses_frozen_coefficient_and_cell_weighted_aggregate():
    fit = {"selected_lambda": 0.1, "period": {"start": "2017-09-01"},
           "protocol": {"split": "fit before holdout"}}
    summary = summarize(fit, [result(2019, 1, 0.2, 0.1), result(2018, 3, 0.4, 0.3)])
    assert [row["year"] for row in summary["evaluations"]] == [2018, 2019]
    assert summary["combined"]["n_correlated_cell_days"] == 4
    assert summary["combined"]["persistence_mae"] == pytest.approx(0.35)
    assert summary["combined"]["advection_mae"] == pytest.approx(0.25)
    assert summary["combined"]["mae_reduction_percent"] == pytest.approx(100 * (1 - 0.25 / 0.35))
    assert summary["evaluations"][0]["motion_coverage"]["days_with_strict_motion"] == 2018
    assert summary["evaluations"][0]["geographic_strata"]["longitude_000_090E"]["n_cells"] == 3


def test_summarize_rejects_refitting_or_reused_fit_year():
    fit = {"selected_lambda": 0.1, "period": {"start": "2017-09-01"}}
    with pytest.raises(ValueError, match="frozen fit coefficient"):
        summarize(fit, [result(2018, 1, 0.2, 0.1, lam=0.2)])
    same_year = result(2017, 1, 0.2, 0.1)
    with pytest.raises(ValueError, match="later than the fit period"):
        summarize(fit, [same_year])
