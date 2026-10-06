import numpy as np
import pytest

from ml.evaluate_copernicus_fusion import fit_ridge, predict, scores


def test_fit_recovers_change_without_future_target_feature():
    rng = np.random.default_rng(4)
    x = np.column_stack((rng.uniform(10, 80, 200), rng.normal(0, 5, 200)))
    delta = 1.5 + 0.4 * x[:, 1]
    model = fit_ridge(x, delta, penalty=0.01)
    assert np.mean(np.abs(predict(model, x) - (x[:, 0] + delta))) < 0.01


def test_predictions_respect_physical_bounds():
    x = np.column_stack((np.full(100, 99.0), np.zeros(100)))
    model = fit_ridge(x, np.full(100, 20.0))
    assert np.all(predict(model, x) == 100)


def test_scores_are_percentage_point_errors():
    result = scores(np.array([0.0, 100.0]), np.array([10.0, 80.0]))
    assert result["mae_percentage_points"] == 15
    assert result["rmse_percentage_points"] == pytest.approx(np.sqrt(250))


def test_insufficient_sample_is_rejected():
    with pytest.raises(ValueError):
        fit_ridge(np.ones((2, 2)), np.ones(2))
