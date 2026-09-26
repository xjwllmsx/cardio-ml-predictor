"""Cross-validated model comparison, test-set evaluation, and plotting helpers."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from cardio.config import RANDOM_STATE

CV_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "neg_brier_score": "neg_brier_score",
    "recall": "recall",
    "f1": "f1",
    "accuracy": "accuracy",
}


def print_statement_header(header_str):
    """Print a header with dividers to section off print statements."""
    print("\n", "=" * 40)
    print(header_str)
    print("=" * 40)


def plot_cvd_distribution(data, ax=None, title="CVD Distribution"):
    """Plot the CVD distribution from a specified target series."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots()
    cardio_counts = data.value_counts()
    ax.bar(
        ["Healthy", "CVD"],
        cardio_counts.values,
        color=["#4ECDC4", "#FF6B6B"],
        edgecolor="black",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Count")
    for i, v in enumerate(cardio_counts.values):
        ax.text(
            i,
            v,
            f"{v:,}\n({v / len(data) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def tune_model(spec, X_train, y_train, cv_splits=5, n_jobs=-1):
    """Run a (randomized) CV search for a single ModelSpec and return the fitted search."""
    pipeline = spec.build_pipeline()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    if not spec.param_distributions:
        # Nothing to tune (e.g. the Dummy baseline): still cross-validate for
        # a comparable row in the comparison table.
        from sklearn.model_selection import cross_validate

        results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=CV_SCORING, return_estimator=True
        )
        pipeline.fit(X_train, y_train)

        cv_results = {
            f"mean_test_{k}": np.array([np.mean(v)])
            for k, v in results.items()
            if k.startswith("test_")
        }
        cv_results.update(
            {
                f"std_test_{k}": np.array([np.std(v)])
                for k, v in results.items()
                if k.startswith("test_")
            }
        )

        # A minimal stand-in for a fitted SearchCV object, so callers with no
        # hyperparameters to tune (e.g. the Dummy baseline) still produce a
        # comparable row in the CV comparison table.
        return SimpleNamespace(
            best_estimator_=pipeline,
            best_params_={},
            cv_results_=cv_results,
            best_index_=0,
        )

    n_iter = spec.n_iter
    if n_iter is None:
        from sklearn.model_selection import GridSearchCV

        search = GridSearchCV(
            pipeline,
            param_grid=spec.param_distributions,
            scoring=CV_SCORING,
            refit="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=spec.param_distributions,
            n_iter=n_iter,
            scoring=CV_SCORING,
            refit="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
    search.fit(X_train, y_train)
    return search


def cv_comparison_table(searches):
    """Build a mean +/- std CV metric table from a dict of {name: fitted search}."""
    rows = []
    for name, search in searches.items():
        idx = getattr(search, "best_index_", 0)
        row = {"Model": name}
        for metric in CV_SCORING:
            mean = search.cv_results_[f"mean_test_{metric}"][idx]
            std = search.cv_results_[f"std_test_{metric}"][idx]
            if metric == "neg_brier_score":
                mean, metric = -mean, "brier_score"
            row[f"{metric} (mean)"] = mean
            row[f"{metric} (std)"] = std
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute a dict of classification metrics with no printing/plotting side effects."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred),
        "Specificity": recall_score(y_true, y_pred, pos_label=0),
        "F1 Score": f1_score(y_true, y_pred),
        "F2 Score": fbeta_score(y_true, y_pred, beta=2),
    }
    if y_proba is not None:
        metrics["ROC AUC"] = roc_auc_score(y_true, y_proba)
        metrics["PR AUC (Average Precision)"] = average_precision_score(
            y_true, y_proba
        )
        metrics["Brier Score"] = brier_score_loss(y_true, y_proba)
        metrics["Log Loss"] = log_loss(y_true, y_proba)
    return metrics


def bootstrap_metric_ci(y_true, y_pred, y_proba, metric_fn, n_resamples=1000, seed=RANDOM_STATE):
    """95% bootstrap confidence interval for a metric function(y_true, y_pred_or_proba)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    values = y_proba if y_proba is not None else y_pred
    values = np.asarray(values)
    n = len(y_true)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        scores[i] = metric_fn(y_true[idx], values[idx])
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return float(lower), float(upper)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "CVD"])
    disp.plot(cmap="Blues")
    plt.title(title, pad=20)
    plt.show()
    return cm


def plot_roc_curves(models_probas, y_true, ax=None):
    """models_probas: dict of {name: predicted probability array}."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in models_probas.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    return ax


def plot_pr_curves(models_probas, y_true, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in models_probas.items():
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    return ax


def plot_calibration(models_probas, y_true, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in models_probas.items():
        CalibrationDisplay.from_predictions(y_true, proba, name=name, ax=ax, n_bins=10)
    ax.set_title("Calibration Curves")
    return ax


def subgroup_report(X_test, y_true, y_pred, y_proba, group_col, bins=None, labels=None):
    """Compute metrics separately for each level of a grouping column.

    If `bins` is given, group_col is cut into bands first (for continuous
    columns like age); otherwise each unique value of group_col is its own group.
    """
    if bins is not None:
        groups = pd.cut(X_test[group_col], bins=bins, labels=labels)
    else:
        groups = X_test[group_col]

    rows = []
    for level in sorted(groups.dropna().unique(), key=str):
        mask = (groups == level).to_numpy()
        if mask.sum() == 0:
            continue
        m = compute_metrics(
            np.asarray(y_true)[mask], np.asarray(y_pred)[mask], np.asarray(y_proba)[mask]
        )
        m["Group"] = level
        m["N"] = int(mask.sum())
        rows.append(m)
    return pd.DataFrame(rows).set_index("Group")
