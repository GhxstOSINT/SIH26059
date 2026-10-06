import datetime as dt

import numpy as np

from ml.calibrate_sic_persistence_blend import calibrate_blends
from ml.evaluate_direct_multilead_sic import apply_persistence_blend, direct_fingerprint


def test_weighted_absolute_error_calibration_finds_known_convex_blend():
    start = dt.date(2020, 1, 1)
    fields = [
        (start, np.full((100, 100), 0.1, dtype=np.float32), None),
        (start + dt.timedelta(days=1), np.full((100, 100), 0.2, dtype=np.float32), None),
        (start + dt.timedelta(days=2), np.full((100, 100), 0.5, dtype=np.float32), None),
    ]
    models = {"1": {"coefficients": [0.8, 0, 0, 0, 0]}}

    alpha = calibrate_blends(fields, models, np.ones((100, 100)),
                             start + dt.timedelta(days=2), start + dt.timedelta(days=2), bins=1001)

    assert alpha["1"] == 0.5


def test_persistence_blend_has_exact_endpoints_and_binds_alpha_in_fingerprint():
    persistence = np.array([[0.2, 0.9]])
    direct = np.array([[0.8, 0.1]])

    assert np.array_equal(apply_persistence_blend(persistence, direct, 0), persistence)
    assert np.allclose(apply_persistence_blend(persistence, direct, 1), direct)
    model = {"1": {"coefficients": [0, 1, 0, 0, 0], "persistence_blend_alpha": 0.4}}
    original = direct_fingerprint(model)
    model["1"]["persistence_blend_alpha"] = 0.7
    assert direct_fingerprint(model) != original
