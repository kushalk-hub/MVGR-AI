"""Train the four deterministic classifiers and tune them.

Models
------
``logistic_regression``  linear baseline; coefficients are interpretable
``random_forest``        bagged trees; feature importances
``xgboost``              gradient-boosted trees; handles feature interactions
``svm``                  margin classifier, linear and RBF kernels

Every candidate is scored with **group-aware stratified** cross-validation:
``StratifiedGroupKFold`` keeps paraphrase families (``group_id``) inside a
single fold, so a family can never appear in both the training and the
validation part of a fold. The folds are seeded, so the search is repeatable.

Hyperparameters are searched with ``RandomizedSearchCV`` and the best
candidate is refitted on the full training split. Validation and test data are
never seen during the search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detection.dataset_loader import LABEL_AI, LABEL_HUMAN

SEED = 42


@dataclass
class ModelSpec:
    """One classifier: its estimator, its default parameters and its search space.

    ``probability_strategy`` is ``"native"`` when the estimator already
    produces calibrated probabilities, and ``"margin_calibrated"`` when the
    model is a margin classifier whose scores are turned into probabilities by
    a sigmoid calibrator fitted on the validation split.
    """

    name: str
    label: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)
    supports_proba: bool = True
    proba_note: str = ""
    probability_strategy: str = "native"

    def build(self, params: dict[str, Any] | None = None) -> Any:
        merged = dict(self.params)
        merged.update(params or {})
        return self.estimator(**merged)


def _specs(estimator_jobs: int = -1) -> dict[str, ModelSpec]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    specs: dict[str, ModelSpec] = {
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            label="Logistic Regression",
            estimator=LogisticRegression,
            params={
                "max_iter": 3000,
                # `liblinear` is the only solver that supports both L1 and L2
                # through `l1_ratio`. sklearn 1.9 deprecated the `penalty`
                # argument, so the penalty is tuned as
                # ``l1_ratio``: 0.0 is L2, 1.0 is L1.
                "solver": "liblinear",
                "l1_ratio": 0.0,
                "class_weight": "balanced",
                "random_state": SEED,
            },
            search_space={
                # The upper bound is deliberately capped: liblinear's per-fit
                # cost grows sharply with C on this ill-conditioned matrix
                # (C=100 with L1 takes minutes on CPU), and the study gains
                # nothing from it.
                "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "l1_ratio": [0.0, 1.0],
                "class_weight": [None, "balanced"],
            },
            supports_proba=True,
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            label="Random Forest",
            estimator=RandomForestClassifier,
            params={
                "n_estimators": 400,
                "random_state": SEED,
                "n_jobs": estimator_jobs,
                "class_weight": "balanced_subsample",
            },
            search_space={
                "n_estimators": [200, 300, 400, 600],
                "max_depth": [None, 8, 12, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
            },
            supports_proba=True,
        ),
        "svm": ModelSpec(
            name="svm",
            label="Support Vector Machine",
            estimator=SVC,
            params={
                # `probability=True` runs an internal five-fold cross-validation
                # on every fit, which multiplies the cost of the search by five.
                # The margin scores are calibrated afterwards on the validation
                # split instead (see MarginCalibratedModel).
                "random_state": SEED,
                "class_weight": "balanced",
                "cache_size": 2048,
            },
            search_space={
                "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                # Linear and RBF are the two kernels the study compares.
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            },
            supports_proba=True,
            probability_strategy="margin_calibrated",
        ),
    }

    try:  # XGBoost is an optional dependency.
        from xgboost import XGBClassifier

        specs["xgboost"] = ModelSpec(
            name="xgboost",
            label="XGBoost",
            estimator=XGBClassifier,
            params={
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "random_state": SEED,
                "n_jobs": estimator_jobs,
            },
            search_space={
                "n_estimators": [200, 300, 400, 600, 800],
                "max_depth": [3, 4, 5, 6, 8, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.5, 0.7, 0.8, 0.9, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
            },
            supports_proba=True,
        )
    except ImportError:  # pragma: no cover - exercised only without xgboost
        pass

    return specs


def model_specs(estimator_jobs: int = -1) -> dict[str, ModelSpec]:
    """Return the available model specifications.

    ``estimator_jobs`` is the number of threads the tree ensembles may use
    (``-1`` = every core).
    """
    return _specs(estimator_jobs)


@dataclass
class SearchResult:
    """The outcome of one hyperparameter search."""

    model_name: str
    label: str
    best_params: dict[str, Any]
    best_score: float
    cv_results: list[dict[str, Any]]
    n_candidates: int
    seconds: float
    supports_proba: bool = True
    cv_strategy: str = "stratified_group_kfold"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "label": self.label,
            "best_params": {k: _jsonable(v) for k, v in self.best_params.items()},
            "best_cv_score": round(float(self.best_score), 6),
            "n_candidates": int(self.n_candidates),
            "seconds": round(float(self.seconds), 3),
            "supports_proba": bool(self.supports_proba),
            "cv_strategy": self.cv_strategy,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def make_cv(groups: np.ndarray | None, n_splits: int, seed: int) -> Any:
    """Build a group-aware stratified splitter (or a stratified one)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if groups is not None and len(np.unique(groups)) >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def run_hyperparameter_search(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 3,
    n_iter: int = 10,
    seed: int = SEED,
    n_jobs: int = 1,
    verbose: int = 0,
) -> SearchResult:
    """Search the model's hyperparameter space with group-aware CV."""
    from sklearn.model_selection import RandomizedSearchCV

    cv = make_cv(groups, n_splits, seed)
    search = RandomizedSearchCV(
        estimator=spec.estimator(**spec.params),
        param_distributions=spec.search_space,
        n_iter=min(n_iter, _grid_size(spec.search_space)),
        cv=cv,
        scoring="roc_auc",
        random_state=seed,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        # A single invalid candidate is discarded rather than aborting the search.
        error_score=np.nan,
    )
    started = time.perf_counter()
    try:
        search.fit(X, y, groups=groups)
    except Exception as exc:  # noqa: BLE001 - one bad candidate must not kill the run
        # Fall back to the model's default parameters, which are always valid.
        estimator = spec.build()
        estimator.fit(X, y)
        seconds = time.perf_counter() - started
        return SearchResult(
            model_name=spec.name,
            label=spec.label,
            best_params={},
            best_score=float("nan"),
            cv_results=[
                {
                    "params": {},
                    "mean_roc_auc": float("nan"),
                    "std_roc_auc": float("nan"),
                    "note": f"search failed ({type(exc).__name__}: {exc}); "
                    "the default parameters were used",
                }
            ],
            n_candidates=0,
            seconds=seconds,
            supports_proba=spec.supports_proba,
            cv_strategy="default_parameters_fallback",
        )
    seconds = time.perf_counter() - started

    candidates: list[dict[str, Any]] = []
    for params, score in zip(
        search.cv_results_["params"], search.cv_results_["mean_test_score"], strict=True
    ):
        candidates.append(
            {
                "params": {k: _jsonable(v) for k, v in params.items()},
                "mean_roc_auc": round(float(score), 6),
                "std_roc_auc": round(
                    float(search.cv_results_["std_test_score"][len(candidates)]), 6
                ),
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["mean_roc_auc"] if np.isfinite(item["mean_roc_auc"]) else 1.0
        )
    )

    return SearchResult(
        model_name=spec.name,
        label=spec.label,
        best_params={k: _jsonable(v) for k, v in search.best_params_.items()},
        best_score=float(search.best_score_),
        cv_results=candidates,
        n_candidates=len(candidates),
        seconds=seconds,
        supports_proba=spec.supports_proba,
    )


def _grid_size(space: dict[str, Any]) -> int:
    size = 1
    for values in space.values():
        size *= max(1, len(values))
    return size


def fit_final(
    spec: ModelSpec, X: np.ndarray, y: np.ndarray, params: dict[str, Any] | None = None
) -> Any:
    """Fit the chosen model on the full training split.

    A margin classifier is wrapped so that its decision scores can be
    calibrated afterwards; the calibrator is fitted by the caller on the
    validation split.
    """
    estimator = spec.build(params)
    estimator.fit(X, y)
    if spec.probability_strategy == "margin_calibrated":
        return MarginCalibratedModel(
            estimator, classes=getattr(estimator, "classes_", None)
        )
    return estimator


def positive_probability(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Return P(label == AI) for a fitted estimator.

    Every model in this branch is configured to produce real probabilities
    (``SVC(probability=True)`` for the SVM). If a model cannot, this raises
    instead of returning a rescaled decision score: an uncalibrated margin is
    not a probability and must never be presented as one.
    """
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            f"{type(estimator).__name__} does not expose predict_proba; "
            "probabilities would be uncalibrated decision scores"
        )
    probabilities = np.asarray(estimator.predict_proba(X))
    classes = list(getattr(estimator, "classes_", [LABEL_HUMAN, LABEL_AI]))
    return probabilities[:, classes.index(LABEL_AI)]


class MarginCalibratedModel:
    """A margin classifier with a sigmoid calibrator fitted on a held-out set.

    ``SVC(probability=True)`` runs an internal five-fold cross-validation on
    every single fit, which multiplies the cost of a hyperparameter search by
    five. This wrapper fits the margin model once and calibrates its scores
    afterwards, on the validation split, with Platt scaling.

    The calibration uses the *validation* rows — the split that exists for
    model selection — so the test split stays untouched. The probabilities
    are therefore genuinely calibrated; no decision score is presented as a
    probability.
    """

    def __init__(self, base: Any, calibrator: Any | None = None, classes: Any = None):
        self.base = base
        self.calibrator = calibrator
        self._classes = np.asarray(classes) if classes is not None else None

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> MarginCalibratedModel:
        self.base.fit(X, y)
        self._classes = np.asarray(getattr(self.base, "classes_", np.array([0, 1])))
        return self

    def fit_calibrator(self, X: np.ndarray, y: np.ndarray) -> MarginCalibratedModel:
        """Fit the Platt scaler on held-out rows (the validation split)."""
        from sklearn.linear_model import LogisticRegression

        margins = np.asarray(self.base.decision_function(X)).ravel()
        self.calibrator = LogisticRegression(max_iter=1000).fit(
            margins.reshape(-1, 1), np.asarray(y)
        )
        return self

    # ------------------------------------------------------------------ #
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.base.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise ValueError(
                "the calibrator has not been fitted; call fit_calibrator() on the "
                "validation split before asking for probabilities"
            )
        margins = np.asarray(self.base.decision_function(X)).ravel().reshape(-1, 1)
        positive = self.calibrator.predict_proba(margins)[:, 1]
        return np.column_stack([1.0 - positive, positive])

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is None:
            raise AttributeError("classes_ is only available after fit")
        return self._classes

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"base": self.base, "calibrator": self.calibrator}

    def __getattr__(self, name: str) -> Any:
        # Forward anything else (feature_importances_, coef_, ...) to the base model.
        if name.startswith("_"):
            raise AttributeError(name)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


__all__ = [
    "MarginCalibratedModel",
    "ModelSpec",
    "SearchResult",
    "fit_final",
    "make_cv",
    "model_specs",
    "positive_probability",
    "run_hyperparameter_search",
]
