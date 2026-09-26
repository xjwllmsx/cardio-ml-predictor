"""Domain-driven feature engineering, usable as the first step of a Pipeline."""

from sklearn.preprocessing import FunctionTransformer

DAYS_PER_YEAR = 365.25


def add_engineered_features(X):
    """Return a copy of X with clinically meaningful derived features.

    - age_years: age converted from days to years (replaces age)
    - gender: recoded from 1/2 to 0 (female) / 1 (male)
    - bmi: body mass index, weight (kg) / height (m)^2
    - pulse_pressure: systolic minus diastolic blood pressure
    - map: mean arterial pressure, (systolic + 2 * diastolic) / 3
    """
    X = X.copy()
    X["age_years"] = X.pop("age") / DAYS_PER_YEAR
    X["gender"] = (X["gender"] == 2).astype(int)
    X["bmi"] = X["weight"] / (X["height"] / 100) ** 2
    X["pulse_pressure"] = X["ap_hi"] - X["ap_lo"]
    X["map"] = (X["ap_hi"] + 2 * X["ap_lo"]) / 3
    return X


def make_feature_engineer():
    """Wrap add_engineered_features so it runs inside a Pipeline."""
    return FunctionTransformer(add_engineered_features, feature_names_out=_names_out)


def _names_out(transformer, input_features):
    names = [f for f in input_features if f != "age"]
    return [*names, "age_years", "bmi", "pulse_pressure", "map"]
