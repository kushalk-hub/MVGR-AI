"""Train-only fitted preprocessing for the deterministic detector.

Everything learned here — the median used to impute missing values and the
mean/scale used to standardise features — is fitted on the **training rows
only** and then applied unchanged to validation and test. Fitting on the full
table would let the test rows influence the model, so it is never done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    """A fitted imputer + scaler bound to an ordered feature list.

    ``median_values``, ``mean_values`` and ``scale_values`` are aligned with
    ``feature_names``. A feature that had no value anywhere in the training
    data is listed in ``dropped_columns``; it is never imputed with zero,
    which would fabricate a signal.
    """

    feature_names: list[str]
    median_values: np.ndarray
    mean_values: np.ndarray
    scale_values: np.ndarray
    dropped_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    fitted: bool = False
    scaled: bool = True

    # ------------------------------------------------------------------ #
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform a frame into a numeric matrix using the fitted statistics."""
        if not self.fitted:
            raise RuntimeError("Preprocessor.transform called before fit")
        dropped = set(self.dropped_columns)
        columns = [name for name in self.feature_names if name not in dropped]
        lookup = {name: index for index, name in enumerate(self.feature_names)}
        positions = np.asarray([lookup[name] for name in columns], dtype=int)

        matrix = frame.reindex(columns=columns).to_numpy(dtype=float)
        matrix = np.where(np.isfinite(matrix), matrix, np.nan)

        medians = np.asarray(self.median_values, dtype=float)[positions]
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix = np.where(missing, np.broadcast_to(medians, matrix.shape), matrix)

        if self.scaled:
            mean = np.asarray(self.mean_values, dtype=float)[positions]
            scale = np.asarray(self.scale_values, dtype=float)[positions]
            matrix = (matrix - mean) / scale
        return matrix

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "dropped_columns": list(self.dropped_columns),
            "constant_columns": list(self.constant_columns),
            "scaled": bool(self.scaled),
            "fitted": bool(self.fitted),
            "n_features": len(self.feature_names),
            "n_dropped": len(self.dropped_columns),
            "n_constant": len(self.constant_columns),
        }


def fit_preprocessor(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    *,
    scale: bool = True,
) -> Preprocessor:
    """Fit imputation and scaling on the training rows only."""
    if not feature_names:
        raise ValueError("no features to fit")

    matrix = train_frame.reindex(columns=list(feature_names)).to_numpy(dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)

    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(matrix, axis=0)
    dropped = [
        name for name, value in zip(feature_names, medians, strict=True)
        if not np.isfinite(value)
    ]
    dropped_set = set(dropped)

    kept_positions = [
        index for index, name in enumerate(feature_names) if name not in dropped_set
    ]
    kept_matrix = matrix[:, kept_positions]
    with np.errstate(invalid="ignore"):
        kept_medians = np.nanmedian(kept_matrix, axis=0)
        kept_means = np.nanmean(kept_matrix, axis=0)
        kept_stds = np.nanstd(kept_matrix, axis=0)

    # `kept_stds` only covers the columns that survive the missing-value check,
    # so it is zipped against the kept names, not against the full list.
    kept_names = [
        name for name in feature_names if name not in set(dropped)
    ]
    constant = [
        name
        for name, value in zip(kept_names, kept_stds, strict=True)
        if not np.isfinite(value) or value <= 0
    ]

    full_median = np.full(len(feature_names), np.nan, dtype=float)
    full_mean = np.zeros(len(feature_names), dtype=float)
    full_scale = np.ones(len(feature_names), dtype=float)
    for position, index in enumerate(kept_positions):
        full_median[index] = kept_medians[position]
        full_mean[index] = (
            kept_means[position] if np.isfinite(kept_means[position]) else 0.0
        )
        full_scale[index] = (
            kept_stds[position]
            if np.isfinite(kept_stds[position]) and kept_stds[position] > 0
            else 1.0
        )

    return Preprocessor(
        feature_names=list(feature_names),
        median_values=full_median,
        mean_values=full_mean,
        scale_values=full_scale,
        dropped_columns=dropped,
        constant_columns=constant,
        fitted=True,
        scaled=scale,
    )


__all__ = ["Preprocessor", "fit_preprocessor"]
