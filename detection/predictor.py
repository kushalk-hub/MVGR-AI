"""The reusable deterministic detection interface.

``DeterministicDetector`` is the public entry point of the deterministic
branch. It wraps a trained estimator together with the preprocessing that was
fitted on the training split, so a caller only has to supply the feature
vector:

.. code-block:: python

    detector = DeterministicDetector.load("models/deterministic/best_model.pkl")
    result = detector.predict_from_features(features)

The returned mapping is a plain dictionary and is JSON-serialisable:

.. code-block:: json

    {"prediction": "AI", "label": 1, "human_probability": 0.18,
     "ai_probability": 0.82, "model": "XGBoost"}

Only models that expose real, calibrated probabilities are accepted; an
uncalibrated decision score is never presented as a probability.

The interface is completely independent of the graph branch: it needs nothing
but the numeric feature vector and the model files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from detection.dataset_loader import LABEL_AI, LABEL_HUMAN

DEFAULT_MODEL_DIR = Path("models/deterministic")


@dataclass
class DetectorBundle:
    """Everything needed to turn a feature vector into a prediction."""

    estimator: Any
    preprocessor: Any
    feature_names: list[str]
    model_name: str
    model_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_label": self.model_label,
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "metadata": dict(self.metadata),
        }


class DeterministicDetector:
    """Load a trained detector and score feature vectors with it."""

    def __init__(self, bundle: DetectorBundle):
        self.bundle = bundle

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> DeterministicDetector:
        """Load a detector bundle from a pickle written by the trainer."""
        import pickle

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"model bundle not found: {target}")
        with open(target, "rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, DetectorBundle):
            raise TypeError(
                f"{target} does not contain a DetectorBundle "
                f"(found {type(bundle).__name__})"
            )
        return cls(bundle)

    @classmethod
    def from_parts(
        cls,
        estimator: Any,
        preprocessor: Any,
        feature_names: Iterable[str],
        *,
        model_name: str,
        model_label: str,
        metadata: dict[str, Any] | None = None,
    ) -> DeterministicDetector:
        return cls(
            DetectorBundle(
                estimator=estimator,
                preprocessor=preprocessor,
                feature_names=list(feature_names),
                model_name=model_name,
                model_label=model_label,
                metadata=dict(metadata or {}),
            )
        )

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> list[str]:
        return list(self.bundle.feature_names)

    @property
    def model_name(self) -> str:
        return self.bundle.model_name

    @property
    def model_label(self) -> str:
        return self.bundle.model_label

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.bundle.metadata)

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def _as_frame(self, features: Any) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        elif isinstance(features, dict):
            frame = pd.DataFrame([features])
        else:
            array = np.asarray(features, dtype=float)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.shape[1] != len(self.bundle.feature_names):
                raise ValueError(
                    f"expected {len(self.bundle.feature_names)} features, "
                    f"got {array.shape[1]}"
                )
            frame = pd.DataFrame(array, columns=self.bundle.feature_names)
        missing = [
            column for column in self.bundle.feature_names if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"missing {len(missing)} required feature column(s), "
                f"first: {missing[:5]}"
            )
        return frame

    def predict_from_features(self, features: Any) -> dict[str, Any]:
        """Predict one document from its feature vector."""
        return self.predict_batch(features)[0]

    def predict_batch(self, features: Any) -> list[dict[str, Any]]:
        """Predict a batch of documents from their feature vectors."""
        frame = self._as_frame(features)
        ordered = frame.reindex(columns=self.bundle.feature_names)
        matrix = self.bundle.preprocessor.transform(ordered)

        estimator = self.bundle.estimator
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                f"{type(estimator).__name__} does not expose predict_proba; "
                "probabilities would be uncalibrated decision scores"
            )
        probabilities = np.asarray(estimator.predict_proba(matrix))
        classes = list(getattr(estimator, "classes_", [LABEL_HUMAN, LABEL_AI]))
        ai_index = classes.index(LABEL_AI)
        human_index = classes.index(LABEL_HUMAN)
        ai_probability = probabilities[:, ai_index]
        human_probability = probabilities[:, human_index]
        labels = np.asarray(estimator.predict(matrix)).astype(int)

        results: list[dict[str, Any]] = []
        for index in range(len(labels)):
            label = int(labels[index])
            results.append(
                {
                    "prediction": "AI" if label == LABEL_AI else "Human",
                    "label": label,
                    "human_probability": round(float(human_probability[index]), 6),
                    "ai_probability": round(float(ai_probability[index]), 6),
                    "model": self.bundle.model_label,
                    "model_name": self.bundle.model_name,
                    "confidence": round(
                        float(max(human_probability[index], ai_probability[index])), 6
                    ),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def predict_from_text(
        self, text: str, *, config: str = "fingerprint_config_10k"
    ) -> dict[str, Any]:
        """Extract the features from raw text and predict.

        This is a convenience wrapper around the existing fingerprinting
        pipeline; it loads the reference GPT-2 model from disk and runs
        offline. Prefer :meth:`predict_from_features` when the features are
        already available.
        """
        from detection.text_features import extract_features_for_text

        features = extract_features_for_text(text, config_name=config)
        return self.predict_from_features(features)

    def describe(self) -> dict[str, Any]:
        """A JSON-serialisable description of the loaded detector."""
        return {
            "model_name": self.bundle.model_name,
            "model_label": self.bundle.model_label,
            "feature_count": len(self.bundle.feature_names),
            "label_convention": {"human": LABEL_HUMAN, "ai": LABEL_AI},
            "metadata": self.bundle.metadata,
        }


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_bundle(bundle: DetectorBundle, path: str | Path) -> str:
    """Write a detector bundle to disk."""
    import pickle

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return str(target)


def save_metadata(metadata: dict[str, Any], path: str | Path) -> str:
    """Write the model metadata sidecar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
        handle.write("\n")
    return str(target)


__all__ = [
    "DEFAULT_MODEL_DIR",
    "DeterministicDetector",
    "DetectorBundle",
    "save_bundle",
    "save_metadata",
]
