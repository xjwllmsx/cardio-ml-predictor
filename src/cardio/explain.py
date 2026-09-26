"""Model interpretability helpers: permutation importance and SHAP."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from cardio.config import RANDOM_STATE


def permutation_importance_table(pipeline, X_test, y_test, n_repeats=10, scoring="roc_auc"):
    """Model-agnostic permutation importance on held-out data, as a sorted dataframe."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance (mean)": result.importances_mean,
            "Importance (std)": result.importances_std,
        }
    ).sort_values("Importance (mean)", ascending=False)
    return table.reset_index(drop=True)


def _transform_for_model(pipeline, X):
    """Run X through every step of the pipeline except the final model."""
    X_transformed = X
    for _, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)
    return X_transformed


def tree_shap_summary(pipeline, X_sample, max_display=15):
    """Beeswarm + bar SHAP summary plots for a tree-based model (TreeExplainer)."""
    import shap

    model = pipeline.named_steps["model"]
    X_transformed = _transform_for_model(pipeline, X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()

    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()

    return shap_values, X_transformed


def linear_shap_summary(pipeline, X_train_sample, X_sample, max_display=15):
    """SHAP summary for a linear model (LinearExplainer), for contrast with a tree model."""
    import shap

    model = pipeline.named_steps["model"]
    X_train_transformed = _transform_for_model(pipeline, X_train_sample)
    X_transformed = _transform_for_model(pipeline, X_sample)

    explainer = shap.LinearExplainer(model, X_train_transformed)
    shap_values = explainer(X_transformed)

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()

    return shap_values, X_transformed


def shap_waterfall_for_index(shap_values, index, max_display=12):
    """Waterfall plot explaining a single prediction (e.g. one TP or FN patient)."""
    import shap

    shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()
