import numpy as np
import pandas as pd
import pytest

from cardio.models import get_model_specs


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "age": rng.integers(365 * 20, 365 * 70, n),
            "gender": rng.integers(1, 3, n),
            "height": rng.integers(150, 200, n),
            "weight": rng.integers(50, 110, n),
            "ap_hi": rng.integers(90, 180, n),
            "ap_lo": rng.integers(60, 110, n),
            "cholesterol": rng.integers(1, 4, n),
            "gluc": rng.integers(1, 4, n),
            "smoke": rng.integers(0, 2, n),
            "alco": rng.integers(0, 2, n),
            "active": rng.integers(0, 2, n),
        }
    )
    # keep ap_lo < ap_hi so feature engineering behaves sensibly
    X["ap_lo"] = np.minimum(X["ap_lo"], X["ap_hi"] - 10)
    y = pd.Series(rng.integers(0, 2, n), name="cardio")
    return X, y


def test_every_model_spec_fits_and_predicts_proba(synthetic_data):
    X, y = synthetic_data
    specs = get_model_specs()
    assert len(specs) >= 8

    for name, spec in specs.items():
        # Use a tiny search so this test runs quickly.
        spec.n_iter = 2 if spec.n_iter is not None else spec.n_iter
        pipeline = spec.build_pipeline()
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2), f"{name} produced unexpected proba shape"
        preds = pipeline.predict(X)
        assert set(preds).issubset({0, 1}), f"{name} produced non-binary predictions"
