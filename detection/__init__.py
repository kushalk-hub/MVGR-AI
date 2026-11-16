"""Deterministic (statistical) AI-text detection branch.

This package owns the deterministic side of MVGR-AI: the stylometric,
writeprints, GLTR and watermark features are produced by the existing
``fingerprinting/``, ``gltr/``, ``watermark/`` and ``fusion/`` packages, and
this package turns them into a trained, independently evaluated detector.

It is deliberately independent of the graph branch (``graph_engine/`` and
``deterministic/`` graph analysis): nothing here reads or writes graph
artifacts, and the predictor interface only needs the numeric feature vector.

Layout
------
``dataset_loader``      load and audit ``10000.csv``; resolve the train/val/test split
``feature_alignment``  align features to sample ids; leakage-safe feature selection
``preprocessing``      train-only fitted imputation and scaling
``train_models``       Logistic Regression / Random Forest / XGBoost / SVM + tuning
``evaluate_models``    metrics, plots and result artifacts
``feature_ablation``   the E1-E9 feature-family ablation study
``robustness_evaluation`` paraphrase-robustness breakdown
``interpretability``   coefficients, importances, SHAP and error analysis
``predictor``          the reusable ``DeterministicDetector`` interface
"""

from __future__ import annotations

__all__ = [
    "DATASET_FILENAME",
    "LABEL_HUMAN",
    "LABEL_AI",
]

#: The primary dataset of the deterministic branch.
DATASET_FILENAME = "10000.csv"

#: Label convention used everywhere in this branch.
LABEL_HUMAN = 0
LABEL_AI = 1
