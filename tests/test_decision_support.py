from datetime import datetime, timezone

import numpy as np

from service.decision_support import evidence_gate, fixed_path_counterfactuals, ice_edge_fragility, vessel_screen


def exposure(line=1.0, band=1.0):
    result = {"observation_timestamp": "2024-06-01T12:00:00Z",
              "sampling": {"valid_sample_fraction": line,
                           "context_band_valid_sample_fraction": band,
                           "corridor_width_assessed": True},
              "context_band_sic_summary_fraction": {"maximum": 0.4}}
    result["evidence_gate"] = evidence_gate(result, datetime(2024, 6, 1, 18, tzinfo=timezone.utc))
    return result


def test_gate_abstains_even_with_full_coverage():
    gate = exposure()["evidence_gate"]
    assert gate["observation_display_status"] == "descriptive_only"
    assert gate["decision_status"] == "abstain"
    assert gate["confidence_probability"] is None
    assert "forecast_uncertainty_not_calibrated" in gate["abstention_reasons"]


def test_gate_identifies_incomplete_context():
    gate = exposure(band=0.5)["evidence_gate"]
    assert gate["observation_display_status"] == "insufficient_coverage"
    assert "context_band_observation_coverage_below_research_display_threshold" in gate["abstention_reasons"]


def test_fragility_marks_flip_and_missing_bins():
    report = ice_edge_fragility([
        {"start_km": 0, "end_km": 5, "mean_sic_fraction": 0.12, "data_coverage_fraction": 1},
        {"start_km": 5, "end_km": 10, "mean_sic_fraction": 0.18, "data_coverage_fraction": 1},
        {"start_km": 10, "end_km": 15, "mean_sic_fraction": None, "data_coverage_fraction": 0},
    ])
    assert report["fragile_segment_count"] == 2
    assert report["unassessable_segment_count"] == 1
    assert report["segments"][0]["flips_with_plus_5_points"]
    assert report["segments"][1]["flips_with_minus_5_points"]


def test_vessel_research_screen_never_claims_clearance():
    result = vessel_screen(exposure(), max_sic=0.5, min_coverage=0.9)
    assert result["profile_verification"] == "user_supplied_unverified"
    assert result["navigation_clearance"] is False
    assert result["decision_status"] == "abstain"
    assert result["status"] == "within_entered_observation_limits_only"
    assert "observed_sic_exceeds_entered_limit" in vessel_screen(
        exposure(), max_sic=0.3, min_coverage=0.9)["findings"]


def test_fixed_route_counterfactuals_share_one_objective_and_do_not_claim_rerouting():
    field = np.array([[0.1, 0.1, 0.1], [0.7, 0.7, 0.7]], dtype=float)
    candidates = [
        {"candidate_id": "north", "path_cells": [(0, 0), (0, 1), (0, 2)]},
        {"candidate_id": "south", "path_cells": [(1, 0), (1, 1), (1, 2)]},
    ]
    result = fixed_path_counterfactuals(field, candidates, 1000, 1000)
    assert [row["sic_change_fraction"] for row in result["scenarios"]] == [-0.05, 0.0, 0.05]
    assert all(row["best_fixed_candidate_id"] == "north" for row in result["scenarios"])
    assert result["navigation_suitability"] == "not_assessed"
    assert "not re-routed" in result["warning"]
