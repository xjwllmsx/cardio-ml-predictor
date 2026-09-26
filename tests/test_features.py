import pandas as pd

from cardio.features import add_engineered_features


def _sample_X():
    return pd.DataFrame(
        {
            "age": [365.25 * 40, 365.25 * 50],
            "gender": [1, 2],
            "height": [170, 160],
            "weight": [70, 64],
            "ap_hi": [120, 140],
            "ap_lo": [80, 90],
            "cholesterol": [1, 2],
            "gluc": [1, 1],
            "smoke": [0, 1],
            "alco": [0, 0],
            "active": [1, 0],
        }
    )


def test_age_years_conversion():
    out = add_engineered_features(_sample_X())
    assert out["age_years"].round(2).tolist() == [40.0, 50.0]
    assert "age" not in out.columns


def test_gender_recoded_to_binary():
    out = add_engineered_features(_sample_X())
    assert out["gender"].tolist() == [0, 1]


def test_bmi_pulse_pressure_map_known_values():
    out = add_engineered_features(_sample_X())

    expected_bmi = [70 / (1.70**2), 64 / (1.60**2)]
    assert out["bmi"].round(3).tolist() == [round(v, 3) for v in expected_bmi]

    assert out["pulse_pressure"].tolist() == [40, 50]

    expected_map = [(120 + 2 * 80) / 3, (140 + 2 * 90) / 3]
    assert out["map"].round(3).tolist() == [round(v, 3) for v in expected_map]


def test_no_nans_produced():
    out = add_engineered_features(_sample_X())
    assert not out.isna().any().any()
