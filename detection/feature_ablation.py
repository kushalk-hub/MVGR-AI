"""Feature-family ablation study.

Nine feature sets are compared under one fixed methodology — the same split,
the same reference model, the same metrics, the same test rows:

======  =========================================  ================
Key     Feature set                               Expected size
======  =========================================  ================
``E1``  stylometric only                          140
``E2``  writeprints only                          301
``E3``  GLTR only                                 19
``E4``  watermark evidence only                   12
``E5``  combined selected features                424
``E6``  combined without GLTR                     405
``E7``  combined without writeprints              123
``E8``  combined without stylometric              284
``E9``  combined without watermark                424
======  =========================================  ================

The sizes above are the *shipped* 424-column reference set; the actual counts
used here are measured on the 10,000-row dataset and reported as such.

Two rules keep the study honest:

* Every column set is selected with the **training rows only** (missing-value
  threshold, constant columns and correlation filtering are all fitted on
  train), exactly like the main experiment.
* The watermark branch is *evidence*, not a predictive feature. It is included
  here as an explicitly flagged ablation so its contribution can be measured
  rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from detection.dataset_loader import ID_COLUMN
from detection.feature_alignment import (
    FAMILY_PREFIXES,
    family_columns,
    select_features_train_only,
)

#: The nine ablation experiments, in the order they are reported.
ABLATION_ORDER: tuple[str, ...] = (
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
)

ABLATION_NAMES: dict[str, str] = {
    "E1": "Stylo only",
    "E2": "Writeprints only",
    "E3": "GLTR only",
    "E4": "Watermark only",
    "E5": "Combined selected features",
    "E6": "Combined features without GLTR",
    "E7": "Combined features without Writeprints",
    "E8": "Combined features without Stylo",
    "E9": "Combined features without Watermark",
}

#: The family that each "without" experiment removes.
REMOVED_FAMILY: dict[str, str] = {
    "E6": "gltr",
    "E7": "writeprints",
    "E8": "stylometric",
    "E9": "watermark",
}


@dataclass
class AblationSet:
    """One feature set of the ablation study."""

    key: str
    name: str
    columns: list[str]
    families: list[str] = field(default_factory=list)
    is_combined: bool = True
    note: str = ""

    @property
    def size(self) -> int:
        return len(self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "features": self.size,
            "families": list(self.families),
            "is_combined": bool(self.is_combined),
            "note": self.note,
            "columns": list(self.columns),
        }


def build_ablation_sets(
    combined_columns: list[str],
    train_frame: pd.DataFrame,
    *,
    watermark_frame: pd.DataFrame | None = None,
    watermark_columns: list[str] | None = None,
    selection_config: dict[str, Any] | None = None,
) -> list[AblationSet]:
    """Build the nine feature sets.

    ``combined_columns`` is the train-only selected column list (experiment
    E5). The single-family sets are selected with the same train-only rules
    restricted to that family. The "without" sets remove one family from the
    combined list.
    """
    config = dict(selection_config or {})

    sets: list[AblationSet] = []

    # E1-E3: one family each, selected on the training rows only.
    for key, family in (("E1", "stylometric"), ("E2", "writeprints"), ("E3", "gltr")):
        available = family_columns(train_frame, family)
        if not available:
            sets.append(
                AblationSet(
                    key=key,
                    name=ABLATION_NAMES[key],
                    columns=[],
                    families=[family],
                    is_combined=False,
                    note="family absent from the feature matrix",
                )
            )
            continue
        columns, _ = select_features_train_only(
            train_frame[[ID_COLUMN, *available]], **config
        )
        sets.append(
            AblationSet(
                key=key,
                name=ABLATION_NAMES[key],
                columns=columns,
                families=[family],
                is_combined=False,
            )
        )

    # E4: the watermark evidence columns (numeric only).
    wm_columns: list[str] = []
    if watermark_frame is not None:
        candidates = watermark_columns
        if candidates is None:
            candidates = [
                column
                for column in watermark_frame.columns
                if column != ID_COLUMN
                and pd.api.types.is_numeric_dtype(watermark_frame[column])
            ]
        wm_columns = list(candidates)
    sets.append(
        AblationSet(
            key="E4",
            name=ABLATION_NAMES["E4"],
            columns=wm_columns,
            families=["watermark"],
            is_combined=False,
            note=(
                "watermark evidence; the key used at generation time is unknown, "
                "so the green list is arbitrary with respect to authorship"
            ),
        )
    )

    # E5: the combined train-only selected features.
    sets.append(
        AblationSet(
            key="E5",
            name=ABLATION_NAMES["E5"],
            columns=list(combined_columns),
            families=["stylometric", "writeprints", "gltr"],
            is_combined=True,
        )
    )

    # E6-E9: the combined set minus one family.
    for key in ("E6", "E7", "E8"):
        family = REMOVED_FAMILY[key]
        prefix = FAMILY_PREFIXES[family]
        columns = [column for column in combined_columns if not column.startswith(prefix)]
        sets.append(
            AblationSet(
                key=key,
                name=ABLATION_NAMES[key],
                columns=columns,
                families=[
                    other
                    for other in ("stylometric", "writeprints", "gltr")
                    if other != family
                ],
                is_combined=True,
                note=f"combined set with the {family} family removed",
            )
        )

    # E9: the combined set without the watermark family. The watermark branch is
    # not part of the fused feature matrix at all (it is evidence only), so E9
    # is identical to E5 by construction — which is itself the finding.
    sets.append(
        AblationSet(
            key="E9",
            name=ABLATION_NAMES["E9"],
            columns=list(combined_columns),
            families=["stylometric", "writeprints", "gltr"],
            is_combined=True,
            note=(
                "the watermark branch is evidence-only and never enters the feature "
                "matrix, so this set is identical to E5"
            ),
        )
    )
    return sets


def run_ablation(
    sets: list[AblationSet],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    spec: Any,
    params: dict[str, Any] | None = None,
    watermark_train: pd.DataFrame | None = None,
    watermark_test: pd.DataFrame | None = None,
    calibration_frame: pd.DataFrame | None = None,
    calibration_labels: np.ndarray | None = None,
    scale: bool = True,
) -> list[dict[str, Any]]:
    """Train and score the reference model on every ablation feature set."""
    from detection.evaluate_models import fit_and_score

    rows: list[dict[str, Any]] = []
    for ablation in sets:
        frame_train = train_frame
        frame_test = test_frame
        if ablation.key == "E4":
            if watermark_train is None or watermark_test is None:
                rows.append(
                    {
                        "key": ablation.key,
                        "name": ablation.name,
                        "features": 0,
                        "status": "skipped",
                        "reason": "no watermark table available",
                    }
                )
                continue
            frame_train = watermark_train
            frame_test = watermark_test
        if not ablation.columns:
            rows.append(
                {
                    "key": ablation.key,
                    "name": ablation.name,
                    "features": 0,
                    "status": "skipped",
                    "reason": "no usable features in this set",
                }
            )
            continue
        _, _, result = fit_and_score(
            spec,
            params,
            frame_train,
            frame_test,
            ablation.columns,
            y_train,
            y_test,
            split="test",
            scale=scale,
            calibration_frame=calibration_frame,
            calibration_labels=calibration_labels,
        )
        row = {
            "key": ablation.key,
            "name": ablation.name,
            "features": ablation.size,
            "status": "evaluated",
            "families": ",".join(ablation.families),
            "is_combined": ablation.is_combined,
            "note": ablation.note,
        }
        row.update({k: round(float(v), 6) for k, v in result.metrics.items()})
        rows.append(row)
    return rows


__all__ = [
    "ABLATION_NAMES",
    "ABLATION_ORDER",
    "AblationSet",
    "REMOVED_FAMILY",
    "build_ablation_sets",
    "run_ablation",
]
