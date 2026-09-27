"""Model definitions and hyperparameter search spaces for the comparison."""

from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from cardio.config import RANDOM_STATE
from cardio.preprocessing import build_pipeline


@dataclass
class ModelSpec:
    """A model plus how to build and tune its pipeline."""

    name: str
    estimator: Any
    param_distributions: dict
    scale: bool = True
    n_iter: int | None = None  # None = exhaustive grid search

    def build_pipeline(self):
        return build_pipeline(self.estimator, scale=self.scale)


def get_model_specs():
    """Return the ModelSpec for every model in the comparison, in report order."""
    specs = [
        ModelSpec(
            name="Dummy (stratified)",
            estimator=DummyClassifier(
                strategy="stratified", random_state=RANDOM_STATE
            ),
            param_distributions={},
        ),
        ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE
            ),
            param_distributions={
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["saga"],
            },
        ),
        ModelSpec(
            name="Gaussian Naive Bayes",
            estimator=GaussianNB(),
            param_distributions={
                "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3],
            },
        ),
        ModelSpec(
            name="K-Nearest Neighbors",
            estimator=KNeighborsClassifier(),
            param_distributions={
                "model__n_neighbors": list(range(5, 102, 4)),
                "model__weights": ["uniform", "distance"],
            },
            n_iter=20,
        ),
        ModelSpec(
            name="Decision Tree",
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            param_distributions={
                "model__max_depth": [3, 4, 5, 6, 8, 10, None],
                "model__min_samples_leaf": [1, 5, 10, 25, 50],
            },
            scale=False,
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            param_distributions={
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [6, 10, 14, None],
                "model__min_samples_leaf": [1, 5, 10, 25],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
            scale=False,
            n_iter=25,
        ),
        ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            param_distributions={
                "model__n_estimators": [200, 400, 600],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [3, 4, 5, 6],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "model__reg_lambda": [0.1, 1, 5, 10],
            },
            scale=False,
            n_iter=25,
        ),
        ModelSpec(
            name="Neural Network (MLP)",
            estimator=MLPClassifier(
                early_stopping=True, random_state=RANDOM_STATE, max_iter=500
            ),
            param_distributions={
                "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "model__learning_rate_init": [1e-3, 3e-3, 1e-2],
            },
            n_iter=15,
        ),
    ]
    return {spec.name: spec for spec in specs}


def build_stacking_pipeline(tuned_pipelines):
    """Build a stacking ensemble from already-tuned pipelines.

    tuned_pipelines: dict mapping model name -> fitted best_estimator_
    (a full features->preprocess->model Pipeline) for the models to combine.
    Each fitted pipeline is used as-is as a stacking base estimator, so the
    ensemble reuses each model's own tuned preprocessing (scaled vs. not).
    """
    estimators = [(name, pipe) for name, pipe in tuned_pipelines.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        stack_method="predict_proba",
        n_jobs=-1,
    )
