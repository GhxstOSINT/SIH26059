import datetime as dt

import numpy as np

from ml.evaluate_direct_multilead_sic import (
    direct_fingerprint, fit_models, score, validate_model_artifact,
)
from ml.identity import model_artifact_sha256


def _fields(count=14):
    start = dt.date(2020, 1, 1)
    return [(start + dt.timedelta(days=i),
             np.full((100, 100), 0.25 + 0.01 * i, dtype=np.float32), None)
            for i in range(count)]


def test_direct_lead_fit_uses_only_targets_at_or_before_cutoff():
    fields = _fields()
    cutoff = fields[7][0]
    models = fit_models(fields, (1, 3), cutoff)

    assert models["1"]["first_target_date"] == "2020-01-03"
    assert models["1"]["last_target_date"] == cutoff.isoformat()
    assert models["3"]["last_target_date"] == cutoff.isoformat()
    assert models["1"]["training_samples"] == 6 * 10_000


def test_direct_multilead_scores_frozen_targets_against_origin_persistence():
    fields = _fields()
    cutoff = fields[7][0]
    models = fit_models(fields[:9], (1, 3), cutoff)
    latitude = np.repeat(np.linspace(-85, -55, 100)[:, None], 100, axis=1)
    report = score(fields, models, np.ones((100, 100), dtype=np.float64), cutoff, latitude)

    assert report["1"]["period"]["first_target_date"] == "2020-01-09"
    assert report["1"]["target_days"] == 6
    assert report["3"]["target_days"] == 6
    assert report["3"]["valid_cell_samples"] == 6 * 10_000
    assert "area_weighted_mae" in report["3"]
    assert "01" in report["3"]["by_target_month"]
    assert set(report["3"]["by_latitude_band"]) == {
        "south_of_80S", "80S_to_70S", "70S_to_60S", "north_of_60S"}


def test_direct_model_fingerprint_binds_coefficients_for_each_lead():
    model = {"1": {"coefficients": [0, 1, 2]}, "3": {"coefficients": [0, 2, 3]}}
    before = direct_fingerprint(model)
    model["3"]["coefficients"][1] = 99
    assert direct_fingerprint(model) != before


def test_fit_and_score_never_bridge_a_missing_date():
    fields = _fields(5)
    second_start = dt.date(2020, 1, 10)
    fields.extend((second_start + dt.timedelta(days=i), grid.copy(), mask)
                  for i, (_, grid, mask) in enumerate(_fields(5)))
    cutoff = fields[-1][0]
    models = fit_models(fields, (1,), cutoff)

    assert models["1"]["training_samples"] == 6 * 10_000
    assert models["1"]["first_target_date"] == "2020-01-03"
    assert models["1"]["last_target_date"] == "2020-01-14"

    holdout = score(fields, models, np.ones((100, 100), dtype=np.float64), dt.date(2020, 1, 1))
    assert holdout["1"]["target_days"] == 6
    assert holdout["1"]["period"]["first_target_date"] == "2020-01-03"
    assert holdout["1"]["period"]["last_target_date"] == "2020-01-14"


def test_empty_latitude_bands_are_reported_explicitly():
    fields = _fields()
    cutoff = fields[7][0]
    models = fit_models(fields[:9], (1,), cutoff)
    latitude = np.full((100, 100), -85.0)
    report = score(fields, models, np.ones((100, 100)), cutoff, latitude)

    assert report["1"]["by_latitude_band"]["south_of_80S"]["target_days"] == 6
    assert report["1"]["by_latitude_band"]["north_of_60S"]["status"] == "no_valid_target_cells"


def test_reused_model_must_match_provenance_fingerprint_and_artifact_digest():
    cutoff = dt.date(2020, 1, 8)
    models = fit_models(_fields(), (1,), cutoff)
    artifact = {
        "model_id": "sic-direct-horizon-regression-research-v1",
        "training_cutoff": cutoff.isoformat(),
        "training_source_manifest_sha256": "abc123",
        "lead_models": models,
        "model_fingerprint": direct_fingerprint(models),
    }
    artifact["artifact_sha256"] = model_artifact_sha256(artifact)
    validate_model_artifact(artifact, "abc123", cutoff, (1,))

    tampered = {**artifact, "lead_models": {"1": {**models["1"], "coefficients": [0, 0, 0, 0, 0]}}}
    with np.testing.assert_raises_regex(ValueError, "fingerprint"):
        validate_model_artifact(tampered, "abc123", cutoff, (1,))
