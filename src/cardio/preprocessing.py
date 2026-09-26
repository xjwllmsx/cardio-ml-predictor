"""Leak-free preprocessing shared by every model pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardio.config import BINARY_FEATURES, CONTINUOUS_FEATURES, ORDINAL_FEATURES
from cardio.features import make_feature_engineer


def build_preprocessor(scale=True):
    """Build a ColumnTransformer for the engineered feature set.

    Cholesterol/glucose are already ordered 1-3 and the binary columns are
    already 0/1, so both pass through unchanged. Continuous columns are
    standardized for distance- and gradient-based models (KNN, Logistic
    Regression, the MLP) and passed through as-is for tree-based models,
    which are scale-invariant.
    """
    continuous_transform = StandardScaler() if scale else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_transform, CONTINUOUS_FEATURES),
            ("ordinal", "passthrough", ORDINAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_pipeline(estimator, scale=True):
    """Assemble the full pipeline: feature engineering -> preprocess -> model."""
    return Pipeline(
        steps=[
            ("features", make_feature_engineer()),
            ("preprocess", build_preprocessor(scale=scale)),
            ("model", estimator),
        ]
    )
