"""Descriptive route evidence gates; never a navigation-clearance engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from service.route_planner import route_metrics


# Research display threshold only. It is not a maritime acceptance criterion.
MIN_DESCRIPTIVE_COVERAGE = 0.90
ICE_EDGE_REFERENCE = 0.15


def evidence_gate(exposure: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    sampling = exposure["sampling"]
    line = float(sampling["valid_sample_fraction"])
    band = float(sampling["context_band_valid_sample_fraction"])
    has_band = bool(sampling["corridor_width_assessed"])
    reasons = []
    if line < MIN_DESCRIPTIVE_COVERAGE:
        reasons.append("centerline_observation_coverage_below_research_display_threshold")
    if has_band and band < MIN_DESCRIPTIVE_COVERAGE:
        reasons.append("context_band_observation_coverage_below_research_display_threshold")
    timestamp = datetime.fromisoformat(exposure["observation_timestamp"].replace("Z", "+00:00"))
    reference = now or datetime.now(timezone.utc)
    age_hours = max(0.0, (reference - timestamp).total_seconds() / 3600)
    if age_hours > 24:
        reasons.append("observation_is_historical_or_stale")
    reasons.extend(("cross_sensor_agreement_not_evaluated", "forecast_uncertainty_not_calibrated",
                    "vessel_and_iceberg_safety_not_assessed"))
    return {
        "decision_status": "abstain",
        "observation_display_status": "descriptive_only" if line >= MIN_DESCRIPTIVE_COVERAGE and (not has_band or band >= MIN_DESCRIPTIVE_COVERAGE) else "insufficient_coverage",
        "confidence_probability": None,
        "observation_age_hours_at_request": round(age_hours, 2),
        "centerline_coverage_fraction": line,
        "context_band_coverage_fraction": band if has_band else None,
        "research_display_coverage_threshold": MIN_DESCRIPTIVE_COVERAGE,
        "abstention_reasons": reasons,
        "meaning": "Observation completeness is not forecast confidence or navigational safety.",
    }


def ice_edge_fragility(profile: list[dict[str, Any]]) -> dict[str, Any]:
    """Show which descriptive 5-km bins change 15% ice-edge class under perturbation."""
    segments = []
    for item in profile:
        value = item.get("mean_sic_fraction")
        coverage = float(item["data_coverage_fraction"])
        if value is None or coverage < MIN_DESCRIPTIVE_COVERAGE:
            state = "unassessable"
            switches_at = None
        else:
            value = float(value)
            state = "ice_present" if value >= ICE_EDGE_REFERENCE else "below_ice_edge_reference"
            switches_at = round(ICE_EDGE_REFERENCE - value, 4)
        segments.append({"start_km": item["start_km"], "end_km": item["end_km"],
                         "coverage_fraction": coverage, "observed_class": state,
                         "sic_change_to_class_flip": switches_at,
                         "flips_with_plus_5_points": switches_at is not None and 0 < switches_at <= 0.05,
                         "flips_with_minus_5_points": switches_at is not None and -0.05 <= switches_at < 0})
    return {"analysis_type": "observed_ice_edge_threshold_sensitivity",
            "ice_edge_reference_fraction": ICE_EDGE_REFERENCE,
            "perturbation_fraction": 0.05,
            "segments": segments,
            "fragile_segment_count": sum(s["flips_with_plus_5_points"] or s["flips_with_minus_5_points"] for s in segments),
            "unassessable_segment_count": sum(s["observed_class"] == "unassessable" for s in segments),
            "warning": "A hypothetical 5-percentage-point SIC change, not a calibrated forecast interval or route recommendation."}


def vessel_screen(exposure: dict[str, Any], *, max_sic: float, min_coverage: float) -> dict[str, Any]:
    """Compare observations with caller-entered limits, without authenticating a PWOM."""
    gate = exposure["evidence_gate"]
    summary = exposure.get("context_band_sic_summary_fraction") or exposure.get("sic_summary_fraction")
    coverage = gate["context_band_coverage_fraction"] if gate["context_band_coverage_fraction"] is not None else gate["centerline_coverage_fraction"]
    findings = []
    if coverage < min_coverage:
        findings.append("below_entered_minimum_observation_coverage")
    if summary is None:
        findings.append("no_valid_ice_observation")
    elif float(summary["maximum"]) > max_sic:
        findings.append("observed_sic_exceeds_entered_limit")
    return {"status": "screen_incomplete" if findings else "within_entered_observation_limits_only",
            "observed_max_sic_fraction": None if summary is None else float(summary["maximum"]),
            "observed_coverage_fraction": coverage,
            "findings": findings,
            "profile_verification": "user_supplied_unverified",
            "decision_status": gate["decision_status"],
            "navigation_clearance": False,
            "warning": "Entered vessel limits are not an authenticated Polar Water Operational Manual. This excludes ice thickness, pressure, bergs, weather, bathymetry and maritime approval."}


def fixed_path_counterfactuals(field: np.ndarray, candidates: list[dict[str, Any]],
                               pixel_x_m: float, pixel_y_m: float) -> dict[str, Any]:
    """Re-score the same candidate paths under a bounded, uniform SIC change."""
    scenarios = []
    for change in (-0.05, 0.0, 0.05):
        adjusted = np.where(np.isfinite(field), np.clip(field + change, 0, 1), np.nan)
        options = []
        for candidate in candidates:
            metrics = route_metrics(adjusted, candidate["path_cells"], pixel_x_m, pixel_y_m,
                                    concentration_weight=4.0, expanded_cells=0)
            options.append({"candidate_id": candidate["candidate_id"],
                            "common_objective_equivalent_km": metrics["objective"]["projected_distance_equivalent_km"]})
        ranked = sorted(options, key=lambda option: (option["common_objective_equivalent_km"], option["candidate_id"]))
        scenarios.append({"sic_change_fraction": change,
                          "best_fixed_candidate_id": ranked[0]["candidate_id"],
                          "margin_to_next_equivalent_km": ranked[1]["common_objective_equivalent_km"] - ranked[0]["common_objective_equivalent_km"] if len(ranked) > 1 else None,
                          "fixed_candidate_scores": options})
    baseline = scenarios[1]["best_fixed_candidate_id"]
    return {"analysis_type": "fixed_alternative_observed_sic_sensitivity",
            "common_objective": "sum(projected edge km * (1 + 4 * midpoint SIC fraction squared))",
            "baseline_best_fixed_candidate_id": baseline,
            "preferred_fixed_alternative_changes": any(item["best_fixed_candidate_id"] != baseline for item in scenarios),
            "scenarios": scenarios,
            "navigation_suitability": "not_assessed",
            "warning": "Only the three existing observed-grid paths are re-scored; paths are not re-routed. Uniform ±5-point SIC is hypothetical, not a forecast, probability, vessel cost, or navigation recommendation."}
