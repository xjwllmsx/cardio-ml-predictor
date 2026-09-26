"""Loading and rule-based cleaning of the cardiovascular disease dataset."""

import pandas as pd

from cardio.config import DATA_PATH, ID_COL, PLAUSIBLE_RANGES, TARGET_COL


def load_data(path=DATA_PATH):
    """Read the raw dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df):
    """Drop duplicate and physiologically implausible rows.

    Every rule is a fixed threshold rather than something learned from the
    data, so applying it before the train/test split does not leak
    information between the two sets.

    Returns the cleaned dataframe and a report dataframe listing how many
    rows each rule removed (rules are applied in order, so each row is
    counted once, under the first rule it failed).
    """
    report = []
    cleaned = df.copy()

    def apply_rule(name, keep_mask):
        nonlocal cleaned
        report.append({"Rule": name, "Rows Dropped": int((~keep_mask).sum())})
        cleaned = cleaned[keep_mask]

    feature_cols = [c for c in cleaned.columns if c != ID_COL]
    apply_rule("Duplicate record", ~cleaned.duplicated(subset=feature_cols))

    for col, (low, high) in PLAUSIBLE_RANGES.items():
        apply_rule(
            f"{col} outside [{low}, {high}]", cleaned[col].between(low, high)
        )

    apply_rule("ap_lo >= ap_hi", cleaned["ap_lo"] < cleaned["ap_hi"])

    report = pd.DataFrame(report)
    total_dropped = len(df) - len(cleaned)
    report.loc[len(report)] = {"Rule": "Total", "Rows Dropped": total_dropped}
    report["% of Original"] = report["Rows Dropped"] / len(df) * 100

    return cleaned.reset_index(drop=True), report


def split_features_target(df):
    """Separate the feature matrix from the target, dropping the ID column."""
    features = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore")
    return features, df[TARGET_COL]
