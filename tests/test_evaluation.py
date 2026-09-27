import numpy as np
import pandas as pd
import pytest

from cardio.evaluation import cv_comparison_table, tune_model
from cardio.models import get_model_specs


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 200
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
    X["ap_lo"] = np.minimum(X["ap_lo"], X["ap_hi"] - 10)
    y = pd.Series(rng.integers(0, 2, n), name="cardio")
    return X, y


def test_tune_model_with_no_hyperparameters_produces_usable_cv_results(synthetic_data):
    """The Dummy baseline has no param_distributions; tune_model takes a
    manual cross_validate path that must produce cv_results_ keys shaped
    like a real SearchCV object's (mean_test_*, std_test_*, split{i}_test_*),
    since cv_comparison_table and the notebook's per-fold box plot rely on
    that shape for every model, tuned or not."""
    spec = get_model_specs()["Dummy (stratified)"]
    search = tune_model(spec, *synthetic_data, cv_splits=3)

    assert search.best_index_ == 0
    assert search.cv_results_["mean_test_roc_auc"][0] == pytest.approx(
        np.mean(
            [search.cv_results_[f"split{i}_test_roc_auc"][0] for i in range(3)]
        )
    )
    assert hasattr(search.best_estimator_, "predict_proba")


def test_cv_comparison_table_includes_dummy_row(synthetic_data):
    X, y = synthetic_data
    specs = get_model_specs()
    searches = {
        name: tune_model(specs[name], X, y, cv_splits=3)
        for name in ["Dummy (stratified)", "Gaussian Naive Bayes"]
    }
    table = cv_comparison_table(searches)

    assert set(table.index) == {"Dummy (stratified)", "Gaussian Naive Bayes"}
    assert "roc_auc (mean)" in table.columns
