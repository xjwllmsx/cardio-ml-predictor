import pandas as pd
import pytest

from cardio.data import clean_data


def _valid_row(**overrides):
    row = {
        "id": 1,
        "age": 20000,
        "gender": 1,
        "height": 165,
        "weight": 70,
        "ap_hi": 120,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "cardio": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def sample_df():
    rows = [
        _valid_row(id=1),
        _valid_row(id=2, ap_hi=130, ap_lo=85),
        _valid_row(id=3, ap_hi=15),  # implausible ap_hi
        _valid_row(id=4, ap_lo=250),  # implausible ap_lo
        _valid_row(id=5, height=40),  # implausible height
        _valid_row(id=6, weight=400),  # implausible weight
        _valid_row(id=7, ap_hi=80, ap_lo=100),  # ap_lo >= ap_hi
    ]
    # duplicate of row id=1's features (different id)
    dup = _valid_row(id=8)
    rows.append(dup)
    return pd.DataFrame(rows)


def test_clean_data_drops_each_bad_row_class(sample_df):
    cleaned, report = clean_data(sample_df)

    # Two valid, non-duplicate rows should survive: id 1 (or its dup) and id 2
    assert len(cleaned) == 2
    assert set(cleaned["ap_hi"]) <= {120, 130}

    total_row = report[report["Rule"] == "Total"].iloc[0]
    assert total_row["Rows Dropped"] == len(sample_df) - len(cleaned)


def test_clean_data_report_counts_known_violations(sample_df):
    _, report = clean_data(sample_df)
    counts = report.set_index("Rule")["Rows Dropped"]

    assert counts["Duplicate record"] == 1
    assert counts["ap_hi outside [80, 250]"] == 1
    assert counts["ap_lo outside [40, 160]"] == 1
    assert counts["height outside [120, 220]"] == 1
    assert counts["weight outside [30, 200]"] == 1
    assert counts["ap_lo >= ap_hi"] == 1


def test_clean_data_resets_index(sample_df):
    cleaned, _ = clean_data(sample_df)
    assert list(cleaned.index) == list(range(len(cleaned)))
