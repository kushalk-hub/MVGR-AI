"""Tests for the deterministic detection branch.

These are unit tests: they use synthetic frames and the real ``10000.csv``
only where the file itself is the subject under test. Nothing here needs the
network, the GPT-2 model, or the extracted 10,000-row features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.dataset_loader import (  # noqa: E402
    LABEL_AI,
    LABEL_HUMAN,
    audit_dataset,
    load_bundle,
    load_dataset,
    split_integrity_report,
)
from detection.evaluate_models import compute_metrics, evaluate_model  # noqa: E402
from detection.feature_ablation import (  # noqa: E402
    ABLATION_ORDER,
    build_ablation_sets,
    run_ablation,
)
from detection.feature_alignment import (  # noqa: E402
    AlignmentError,
    align_features,
    coverage_report,
    family_breakdown,
    feature_family_of,
    select_features_train_only,
)
from detection.interpretability import (  # noqa: E402
    error_profile,
    logistic_coefficients,
    xgboost_importances,
)
from detection.predictor import DetectorBundle, DeterministicDetector  # noqa: E402
from detection.preprocessing import fit_preprocessor  # noqa: E402
from detection.robustness_evaluation import (  # noqa: E402
    evaluate_categories,
    robustness_table,
    summarise_robustness,
)
from detection.train_models import model_specs  # noqa: E402

DATASET = REPO_ROOT / "10000.csv"
requires_dataset = pytest.mark.skipif(
    not DATASET.exists(), reason="10000.csv is not present"
)


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
def _synthetic_features(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = [f"s{i:04d}" for i in range(n)]
    frame = pd.DataFrame({"sample_id": ids})
    frame["stylometric__vocab_total_words"] = rng.normal(100, 10, n)
    frame["stylometric__sent_count"] = rng.integers(1, 20, n).astype(float)
    frame["writeprints__wp_char_unigram_000"] = rng.normal(0.05, 0.01, n)
    frame["gltr__gltr_mean_prob"] = rng.uniform(0.01, 0.5, n)
    frame["gltr__gltr_perplexity"] = rng.uniform(10, 500, n)
    frame["constant__x"] = 1.0
    return frame


def _synthetic_dataset(n: int = 240, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([LABEL_HUMAN] * (n // 2) + [LABEL_AI] * (n - n // 2))
    rng.shuffle(labels)
    text_types = ["original", "paraphrased", "deep_paraphrased"]
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i:04d}" for i in range(n)],
            "label": labels.astype(str),
            "text_type": [text_types[i % len(text_types)] for i in range(n)],
            "source_dataset": "synthetic",
            "group_id": [f"g{i // 2}" for i in range(n)],
        }
    )
    return frame


# --------------------------------------------------------------------------- #
# dataset loader
# --------------------------------------------------------------------------- #
@requires_dataset
def test_dataset_has_the_expected_shape():
    frame = load_dataset()
    assert len(frame) == 10000
    assert frame["sample_id"].is_unique
    assert (frame["text"].str.strip() != "").all()
    assert set(frame["label"].astype(str).unique()) == {"0", "1"}


@requires_dataset
def test_audit_reports_every_requested_field():
    audit = audit_dataset()
    assert audit["rows"] == 10000
    assert audit["valid_records"] + audit["invalid_records"] == audit["rows"]
    assert audit["label_human"] + audit["label_ai"] == audit["rows"]
    assert audit["duplicate_sample_ids"] == 0
    assert audit["duplicate_texts_exact"] == 0
    assert audit["missing_values_by_column"]
    assert audit["source_csv_split_distribution"]
    assert audit["families"]["unique"] > 0


@requires_dataset
def test_resolved_split_is_leakage_free():
    bundle = load_bundle()
    report = bundle.split_report
    assert report["leakage_free"] is True
    assert report["families_spanning_multiple_splits"] == 0
    assert report["duplicate_texts_spanning_splits"] == 0
    counts = bundle.counts()
    assert set(counts) == {"train", "validation", "test"}
    assert sum(counts.values()) == 10000


@requires_dataset
def test_resolved_split_covers_every_row_exactly_once():
    bundle = load_bundle()
    assert bundle.frame["model_split"].notna().all()
    assert (bundle.frame["model_split"] != "").all()


def test_split_integrity_report_detects_a_broken_family():
    """A family whose two members land in different splits must be reported."""
    frame = _synthetic_dataset()
    # The fixture puts rows 2k and 2k+1 in the same family, so an alternating
    # assignment breaks exactly those families.
    frame["model_split"] = [
        "train" if index % 2 == 0 else "test" for index in range(len(frame))
    ]
    report = split_integrity_report(frame)
    assert report["leakage_free"] is False
    assert report["families_spanning_multiple_splits"] >= 1
    assert report["rows_in_families_spanning_splits"] >= 1


# --------------------------------------------------------------------------- #
# feature alignment
# --------------------------------------------------------------------------- #
def test_align_features_requires_full_coverage():
    features = _synthetic_features(20)
    dataset = _synthetic_dataset(20)
    aligned = align_features(features, dataset)
    assert len(aligned) == 20
    assert "gltr__gltr_mean_prob" in aligned.columns

    smaller = features.head(19)
    with pytest.raises(AlignmentError):
        align_features(smaller, dataset)


def test_coverage_report_counts_records_and_columns():
    features = _synthetic_features(20)
    dataset = _synthetic_dataset(20)
    report = coverage_report(features, dataset)
    assert report["dataset_records"] == 20
    assert report["covered_records"] == 20
    assert report["uncovered_records"] == 0
    assert report["feature_columns"] >= 4


def test_train_only_selection_ignores_validation_and_test_rows():
    """A column that is constant in train must be dropped, even if it varies later.

    This is exactly the leakage the train-only selection prevents: the
    decision is taken on the training rows alone, so the variation that only
    appears in the validation and test rows cannot save the column.
    """
    rng = np.random.default_rng(3)
    n = 120
    frame = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)]})
    frame["good"] = rng.normal(0, 1, n)
    frame["train_constant"] = 1.0
    frame.loc[60:, "train_constant"] = rng.normal(5, 1, n - 60)

    train = frame.head(60)
    columns, report = select_features_train_only(train)
    assert "good" in columns
    assert "train_constant" not in columns
    assert "train_constant" in report["dropped_constant"]
    assert report["fitted_on"] == "train_split_only"


def test_train_only_selection_drops_missing_and_correlated_columns():
    rng = np.random.default_rng(5)
    n = 100
    frame = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)]})
    frame["a"] = rng.normal(0, 1, n)
    frame["a_copy"] = frame["a"] * 1.0 + 1e-9  # perfectly correlated
    frame["mostly_missing"] = np.nan
    frame.loc[:9, "mostly_missing"] = 1.0

    columns, report = select_features_train_only(frame)
    assert "a" in columns
    assert "a_copy" not in columns
    assert "mostly_missing" not in columns
    assert "a_copy" in report["dropped_correlated"]


def test_feature_family_mapping():
    assert feature_family_of("gltr__gltr_mean_prob") == "gltr"
    assert feature_family_of("stylometric__sent_count") == "stylometric"
    assert feature_family_of("unknown_column") == "other"
    counts = family_breakdown(["gltr__a", "gltr__b", "writeprints__c"])
    assert counts["gltr"] == 2
    assert counts["writeprints"] == 1


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def test_preprocessor_imputes_with_the_training_median():
    frame = pd.DataFrame(
        {
            "a": [1.0, 3.0, 5.0, np.nan],
            "b": [2.0, 2.0, 2.0, 2.0],
        }
    )
    preprocessor = fit_preprocessor(frame, ["a", "b"], scale=False)
    matrix = preprocessor.transform(frame)
    # The training median of "a" is 3.0, so the missing cell becomes 3.0.
    assert matrix[3, 0] == pytest.approx(3.0)
    assert preprocessor.constant_columns == ["b"]


def test_preprocessor_scales_using_training_statistics_only():
    train = pd.DataFrame({"a": [0.0, 2.0]})
    other = pd.DataFrame({"a": [100.0]})
    preprocessor = fit_preprocessor(train, ["a"], scale=True)
    matrix = preprocessor.transform(other)
    # Training mean 1.0, std 1.0 -> 100.0 becomes 99.0
    assert matrix[0, 0] == pytest.approx(99.0)


def test_preprocessor_reports_columns_that_cannot_be_imputed():
    frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    preprocessor = fit_preprocessor(frame, ["a", "b"], scale=False)
    assert preprocessor.dropped_columns == ["a"]
    matrix = preprocessor.transform(frame)
    assert matrix.shape == (2, 1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_covers_the_full_metric_set():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.4, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
        "false_negative_rate",
        "specificity",
        "balanced_accuracy",
    ):
        assert key in metrics
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_model_produces_a_confusion_matrix():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(-1, 0.4, (60, 3)), rng.normal(1, 0.4, (60, 3))])
    y = np.array([0] * 60 + [1] * 60)
    estimator = LogisticRegression(max_iter=1000).fit(X, y)
    result = evaluate_model("lr", "Logistic Regression", estimator, X, y)
    assert len(result.confusion_matrix) == 2
    assert sum(sum(row) for row in result.confusion_matrix) == 120


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #
def test_ablation_builds_nine_sets_with_the_expected_families():
    features = _synthetic_features()
    train = features.copy()
    columns = [c for c in features.columns if c != "sample_id"]
    sets = build_ablation_sets(columns, train)
    assert [s.key for s in sets] == list(ABLATION_ORDER)
    by_key = {s.key: s for s in sets}
    assert by_key["E1"].families == ["stylometric"]
    assert by_key["E2"].families == ["writeprints"]
    assert by_key["E3"].families == ["gltr"]
    assert by_key["E6"].families == ["stylometric", "writeprints"]
    # E6 removes every GLTR column from the combined set.
    assert all(not c.startswith("gltr__") for c in by_key["E6"].columns)


def test_ablation_runs_and_reports_a_metric_per_set():
    features = _synthetic_features()
    n = len(features)
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    columns = [c for c in features.columns if c != "sample_id"]
    sets = build_ablation_sets(columns, features)
    spec = model_specs()["logistic_regression"]
    rows = run_ablation(
        sets, features, features, y, y, spec=spec, params={"C": 1.0, "penalty": "l2"}
    )
    assert len(rows) == len(sets)
    for row in rows:
        assert row["status"] in {"evaluated", "skipped"}
        if row["status"] == "evaluated":
            assert "accuracy" in row


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_category_evaluation_reports_every_category():
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "text_type": [
                "original",
                "original",
                "paraphrased",
                "paraphrased",
                "original",
                "original",
                "deep_paraphrased",
                "deep_paraphrased",
            ],
        }
    )
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.7, 0.2, 0.8, 0.9, 0.8, 0.3, 0.95])
    categories = evaluate_categories(metadata, y_true, y_pred, y_proba)
    assert len(categories) == 4
    frame = robustness_table(categories)
    assert not frame.empty
    summary = summarise_robustness(frame)
    assert "categories" in summary
    assert "robustness_claim_supported" in summary


# --------------------------------------------------------------------------- #
# interpretability
# --------------------------------------------------------------------------- #
def test_error_profile_separates_the_four_outcome_groups():
    features = _synthetic_features(40)
    columns = [c for c in features.columns if c != "sample_id"]
    y_true = np.array([0] * 20 + [1] * 20)
    y_pred = np.array([0] * 19 + [1] + [1] * 19 + [0])
    profile = error_profile(features, y_true, y_pred, columns)
    assert profile["group_counts"]["true_negative"] == 19
    assert profile["group_counts"]["false_positive"] == 1
    assert profile["group_counts"]["false_negative"] == 1
    assert profile["group_counts"]["true_positive"] == 19
    assert not profile["table"].empty


def test_logistic_coefficients_are_signed_and_aligned():
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])
    estimator = LogisticRegression().fit(X, y)
    coefficients = logistic_coefficients(estimator, ["feature"])
    assert coefficients.index[0] == "feature"
    assert coefficients.iloc[0] > 0


def test_xgboost_importances_align_to_feature_names():
    """A model fit on a numpy array keys features as f0..fn, not by name."""
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(-1, 0.4, (80, 3)), rng.normal(1, 0.4, (80, 3))])
    y = np.array([0] * 80 + [1] * 80)
    estimator = xgboost.XGBClassifier(n_estimators=20, max_depth=3).fit(X, y)
    names = ["alpha", "beta", "gamma"]
    importances = xgboost_importances(estimator, names)
    assert set(importances.index) == set(names)
    # The mapping must not collapse every feature to zero.
    assert importances.sum() > 0
    assert importances.iloc[0] >= importances.iloc[-1]


# --------------------------------------------------------------------------- #
# predictor
# --------------------------------------------------------------------------- #
def test_detector_round_trips_through_disk(tmp_path):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(-1, 0.4, (60, 3)), rng.normal(1, 0.4, (60, 3))])
    y = np.array([0] * 60 + [1] * 60)
    estimator = LogisticRegression(max_iter=1000).fit(X, y)
    frame = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    preprocessor = fit_preprocessor(frame, ["f1", "f2", "f3"], scale=True)

    bundle = DetectorBundle(
        estimator=estimator,
        preprocessor=preprocessor,
        feature_names=["f1", "f2", "f3"],
        model_name="logistic_regression",
        model_label="Logistic Regression",
        metadata={"dataset_file": "10000.csv"},
    )
    path = tmp_path / "best_model.pkl"
    from detection.predictor import save_bundle

    save_bundle(bundle, path)
    detector = DeterministicDetector.load(path)

    result = detector.predict_from_features(frame.head(1))
    assert set(result) >= {"prediction", "label", "human_probability", "ai_probability", "model"}
    assert result["prediction"] in {"AI", "Human"}
    assert result["label"] in {0, 1}
    assert 0.0 <= result["ai_probability"] <= 1.0
    assert 0.0 <= result["human_probability"] <= 1.0
    assert result["ai_probability"] + result["human_probability"] == pytest.approx(1.0)
    assert result["model"] == "Logistic Regression"


def test_detector_rejects_a_frame_with_missing_columns():
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    preprocessor = fit_preprocessor(pd.DataFrame({"f1": [0.0, 1.0]}), ["f1"], scale=False)
    detector = DeterministicDetector.from_parts(
        estimator,
        preprocessor,
        ["f1", "f2"],
        model_name="logistic_regression",
        model_label="Logistic Regression",
    )
    with pytest.raises(ValueError):
        detector.predict_from_features(pd.DataFrame({"f1": [0.5]}))


def test_detector_accepts_a_plain_dictionary():
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    preprocessor = fit_preprocessor(pd.DataFrame({"f1": [0.0, 1.0]}), ["f1"], scale=False)
    detector = DeterministicDetector.from_parts(
        estimator,
        preprocessor,
        ["f1"],
        model_name="logistic_regression",
        model_label="Logistic Regression",
    )
    result = detector.predict_from_features({"f1": 1.0})
    assert result["label"] == 1


def test_detector_rejects_a_model_without_probabilities():
    from sklearn.svm import SVC

    estimator = SVC(probability=False).fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    preprocessor = fit_preprocessor(pd.DataFrame({"f1": [0.0, 1.0]}), ["f1"], scale=False)
    detector = DeterministicDetector.from_parts(
        estimator,
        preprocessor,
        ["f1"],
        model_name="svm",
        model_label="Support Vector Machine",
    )
    with pytest.raises(ValueError):
        detector.predict_from_features({"f1": 1.0})


def test_margin_calibrated_model_produces_real_probabilities():
    from sklearn.svm import SVC

    from detection.train_models import MarginCalibratedModel

    rng = np.random.default_rng(9)
    X = np.vstack([rng.normal(-1, 0.4, (60, 2)), rng.normal(1, 0.4, (60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    base = SVC(kernel="linear").fit(X, y)
    model = MarginCalibratedModel(base, classes=base.classes_)
    model.fit_calibrator(X, y)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (120, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert list(model.classes_) == [0, 1]
    assert (model.predict(X) == y).mean() > 0.9


def test_margin_calibrated_model_refuses_uncalibrated_probabilities():
    from sklearn.svm import SVC

    from detection.train_models import MarginCalibratedModel

    base = SVC(kernel="linear").fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    model = MarginCalibratedModel(base, classes=base.classes_)
    with pytest.raises(ValueError):
        model.predict_proba(np.array([[0.5]]))


def test_writeprints_vocabulary_scope_limits_the_fit_to_the_training_rows(tmp_path):
    """The vocabulary is a learned step, so only the training rows may shape it."""
    from scripts import fingerprint_pipeline as fp

    frame = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(6)], "text": ["t"] * 6}
    )
    assignment = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "model_split": ["train", "train", "train", "test", "validation", "test"],
        }
    )
    split_file = tmp_path / "split_assignment.csv"
    assignment.to_csv(split_file, index=False)

    config = {
        "input": {"id_column": "sample_id"},
        "writeprints": {
            "vocabulary_scope": {
                "mode": "split_file",
                "split_file": str(split_file),
                "split_column": "model_split",
                "split_values": ["train"],
            }
        },
    }
    mask = fp._vocabulary_scope_mask(config, frame)
    assert mask == [True, True, True, False, False, False]

    # Without a scope every row may be used (the extraction-only default).
    plain = {"input": {"id_column": "sample_id"}, "writeprints": {}}
    assert fp._vocabulary_scope_mask(plain, frame) == [True] * 6

    # A missing split file must not silently narrow the fit.
    missing = dict(config)
    missing["writeprints"] = dict(config["writeprints"])
    missing["writeprints"]["vocabulary_scope"] = dict(
        config["writeprints"]["vocabulary_scope"], split_file=str(tmp_path / "nope.csv")
    )
    assert fp._vocabulary_scope_mask(missing, frame) == [True] * 6


def test_detection_package_does_not_depend_on_the_graph_branch():
    """The deterministic branch must stay independent of the graph branch."""
    import detection.dataset_loader
    import detection.evaluate_models
    import detection.feature_ablation
    import detection.feature_alignment
    import detection.interpretability
    import detection.predictor
    import detection.preprocessing
    import detection.robustness_evaluation
    import detection.train_models

    forbidden = ("graph_engine", "deterministic.feature_", "deterministic.pipeline")
    for module in (
        detection.dataset_loader,
        detection.evaluate_models,
        detection.feature_ablation,
        detection.feature_alignment,
        detection.interpretability,
        detection.predictor,
        detection.preprocessing,
        detection.robustness_evaluation,
        detection.train_models,
    ):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{module.__name__} references {token}"


def test_text_features_extract_the_documented_columns():
    """Raw text -> feature vector, using the existing fingerprinting pipeline."""
    pytest.importorskip("spacy")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from gltr.local_model import DEFAULT_MODEL_DIR, local_dir_ready

    ready, reason = local_dir_ready(DEFAULT_MODEL_DIR)
    if not ready:
        pytest.skip(f"no local GPT-2 at {DEFAULT_MODEL_DIR}: {reason}")

    from detection.text_features import extract_features_for_text

    features = extract_features_for_text(
        "The quick brown fox jumps over the lazy dog. Machine text tends to be smooth.",
        feature_names=["stylometric__sent_count", "gltr__gltr_token_count"],
    )
    assert set(features) == {"stylometric__sent_count", "gltr__gltr_token_count"}
    assert features["gltr__gltr_token_count"] > 0
    assert features["stylometric__sent_count"] >= 1
