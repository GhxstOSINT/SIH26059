import numpy as np
import pytest

from ml.analyze_iceknn_verification import MODELS, summarize_arrays


def test_summarize_arrays_compares_published_metrics_by_forecast_lead():
    verification = {}
    for index, folder in enumerate(MODELS):
        verification[folder] = {
            "iiee": np.array([[1.0 + index, 3.0], [np.nan, 5.0]]),
            "sie_pred": np.array([[4.0 + index, 5.0], [8.0, 9.0]]),
            "sie_obs": np.array([[3.0, 2.0], [7.0, 7.0]]),
        }

    report = summarize_arrays(verification, leads=(1, 2))
    model = report["models"]["Ice-kNN-South"]
    assert model["1"]["iiee_sample_count"] == 1
    assert model["1"]["mean_iiee_million_km2"] == 3.0
    assert model["2"]["mean_absolute_sie_error_million_km2"] == 2.5
    assert model["2"]["iiee_sample_count"] == 2


def test_summarize_arrays_rejects_missing_model_or_bad_shapes():
    with pytest.raises(ValueError, match="Missing archived verification"):
        summarize_arrays({}, leads=(1,))
    bad = {folder: {"iiee": np.zeros((2, 2)), "sie_pred": np.zeros((2, 2)),
                    "sie_obs": np.zeros((1, 2))} for folder in MODELS}
    with pytest.raises(ValueError, match="dimensions"):
        summarize_arrays(bad, leads=(1,))
