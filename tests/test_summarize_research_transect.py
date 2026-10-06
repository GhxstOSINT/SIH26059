from datetime import date, timedelta

import numpy as np

from ml.research_transect import load_research_transect
from ml.summarize_research_transect import summarize
from ml.verify_glo12_forecast import write_match


def _report(directory, bulletin, *, lead=1, bias=0.02, coverage=1.0):
    route, route_sha = load_research_transect()
    target = bulletin + timedelta(days=lead)
    common = round(10 * coverage)
    report = {
        "scope": "single_run_research_verification_sample",
        "bulletin_date": bulletin.isoformat(), "valid_date": target.isoformat(),
        "lead_days": lead, "matched_cells": common,
        "forecast_source_sha256": "a" * 64, "observation_source_sha256": "b" * 64,
        "uncertainty_calibrated": False, "navigation_clearance": False,
        "research_transect": {
            "route_id": route["id"], "route_config_sha256": route_sha,
            "evaluation_status": "single_bulletin_research_sample",
            "navigation_clearance": False, "unique_observation_cells": 10,
            "common_valid_cells": common,
            "forecast_mae_fraction": abs(bias), "forecast_rmse_fraction": abs(bias),
            "forecast_bias_fraction": bias, "observed_mean_sic_fraction": 0.4,
            "forecast_mean_sic_fraction": 0.4 + bias,
            "ice_edge_disagreement_fraction": 0.0,
            "persistence_mae_fraction": 0.06,
        },
    }
    samples = {"forecast_sic_fraction": np.full(common, 0.4 + bias, dtype=np.float32),
               "observed_sic_fraction": np.full(common, 0.4, dtype=np.float32),
               "observation_uncertainty_fraction": np.full(common, 0.1, dtype=np.float32),
               "transect_forecast_sic_fraction": np.full(common, 0.4 + bias, dtype=np.float32),
               "transect_observed_sic_fraction": np.full(common, 0.4, dtype=np.float32),
               "transect_observation_uncertainty_fraction": np.full(common, 0.1, dtype=np.float32),
               "transect_persistence_sic_fraction": np.full(common, 0.46, dtype=np.float32),
               "persistence_sic_fraction": np.full(common, 0.46, dtype=np.float32)}
    output = directory / f"glo12-{bulletin.isoformat()}-lead-{lead}-peninsula-weddell.json"
    write_match(report, samples, output)
    return output


def test_empty_research_evidence_stays_pending(tmp_path):
    report = summarize(tmp_path)
    assert report["verified_case_count"] == 0
    assert report["uncertainty_calibrated"] is False
    assert all(lead["route_mean_sic_interval"] is None for lead in report["leads"])


def test_chronological_bulletin_interval_and_paired_baseline(tmp_path):
    start = date(2026, 10, 4)
    for index in range(40):
        _report(tmp_path, start + timedelta(days=index), bias=0.02 if index < 30 else 0.01)
    result = summarize(tmp_path)
    lead = result["leads"][0]
    assert result["verified_case_count"] == 40
    assert lead["evaluable_bulletins"] == 40
    assert lead["paired_persistence_bulletins"] == 40
    assert lead["route_mean_sic_interval"]["fit_bulletins"] == 30
    assert lead["route_mean_sic_interval"]["holdout_bulletins"] == 10
    assert lead["route_mean_sic_interval"]["observed_holdout_coverage"] == 1.0
    assert result["navigation_clearance"] is False
    assert result["uncertainty_calibrated"] is False


def test_prefreeze_is_excluded_and_corrupt_report_rejected(tmp_path):
    early = _report(tmp_path, date(2026, 10, 3))
    valid = _report(tmp_path, date(2026, 10, 4), coverage=0.4)
    report = summarize(tmp_path)
    assert report["excluded_pre_freeze_reports"] == 1
    assert report["leads"][0]["low_coverage_bulletins"] == 1
    with valid.with_suffix(".npz").open("ab") as source:
        source.write(b"tamper")
    report = summarize(tmp_path)
    assert report["rejected_reports"] == 1
    assert report["verified_case_count"] == 0
