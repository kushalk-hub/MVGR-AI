"""Module C tests — GLTR-style local token analysis.

The numeric core (causal shifting, probabilities, ranks, entropy, document
aggregation) is pure NumPy and is tested directly. The model-loading path is
tested for its *offline* guarantees: with no local model it must refuse rather
than download. If a local causal model happens to be cached, the loaded path is
exercised instead.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gltr.document_features import (  # noqa: E402
    aggregate_document_features,
    aggregate_from_arrays,
    feature_names,
)
from gltr.entropy_analysis import next_token_entropies  # noqa: E402
from gltr.local_model import (  # noqa: E402
    GltrConfig,
    GltrModel,
    GltrUnavailableError,
    local_model_available,
)
from gltr.pipeline import GltrAnalyzer, estimate_requirements  # noqa: E402
from gltr.token_probability import (  # noqa: E402
    causal_shift,
    next_token_probabilities,
    softmax,
)
from gltr.token_rank import rank_statistics, token_ranks  # noqa: E402


def _logits(rows):
    return np.log(np.asarray(rows, dtype=np.float64))


def test_softmax_rows_sum_to_one():
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_causal_shift_aligns_position_with_next_token():
    logits = np.zeros((2, 4, 5))
    ids = np.array([10, 11, 12, 13])
    shifted, targets = causal_shift(logits, ids)
    assert shifted.shape == (2, 3, 5)
    # logits at position i predict token i+1
    assert list(targets) == [11, 12, 13]


def test_next_token_probabilities_are_correct():
    logits = _logits([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    ids = np.array([0, 1, 1])
    probabilities, targets = next_token_probabilities(logits, ids)
    assert list(targets) == [1, 1]
    assert np.allclose(probabilities, [0.5, 0.1])


def test_token_ranks_are_one_based():
    logits = _logits([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    ids = np.array([0, 1, 1])
    ranks = token_ranks(logits, ids)
    # row0: realised token is tied top -> rank 1; row1: realised token is worst -> rank 2
    assert list(ranks) == [1, 2]


def test_rank_statistics_proportions():
    ranks = np.array([1, 2, 3, 1000, 5000])
    stats = rank_statistics(ranks, thresholds=(10, 100, 1000))
    assert stats["count"] == 5
    assert stats["frac_le_10"] == pytest.approx(0.6)
    assert stats["frac_le_1000"] == pytest.approx(0.8)
    assert stats["frac_gt_1000"] == pytest.approx(0.2)


def test_entropy_of_uniform_and_peaked():
    logits = _logits([[0.5, 0.5], [0.999, 0.001], [0.999, 0.001]])
    ids = np.array([0, 0, 0])
    entropies = next_token_entropies(logits, ids)
    assert entropies.shape == (2,)
    assert entropies[0] == pytest.approx(np.log(2), abs=1e-6)
    assert entropies[1] < entropies[0]


def test_document_aggregation_values():
    logits = _logits([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    ids = np.array([0, 1, 1])
    features = aggregate_document_features(logits, ids, rank_thresholds=(1, 2))
    assert features["gltr_token_count"] == 2
    assert features["gltr_mean_prob"] == pytest.approx(0.3)
    assert features["gltr_mean_rank"] == pytest.approx(1.5)
    assert features["gltr_perplexity"] > 1.0
    assert features["gltr_frac_rank_le_1"] == pytest.approx(0.5)
    assert features["gltr_frac_rank_gt_2"] == pytest.approx(0.0)


def test_document_aggregation_handles_degenerate_input():
    empty = aggregate_from_arrays(np.array([]), np.array([]), np.array([]))
    assert empty["gltr_token_count"] == 0
    assert empty["gltr_mean_prob"] is None
    too_short = aggregate_document_features(np.zeros((1, 4)), np.array([0]))
    assert too_short["gltr_token_count"] == 0


def test_feature_names_stable():
    names = feature_names((10, 100, 1000))
    assert len(names) == len(set(names))
    assert "gltr_mean_prob" in names
    assert "gltr_frac_rank_le_1000" in names
    assert feature_names((10, 100, 1000)) == names


def test_missing_model_reports_reason_and_never_downloads():
    available, reason = local_model_available("definitely-not-a-real-model-zzz")
    assert available is False
    assert isinstance(reason, str) and reason
    model = GltrModel(
        config=GltrConfig(model_name="definitely-not-a-real-model-zzz", model_dir=None)
    )
    with pytest.raises(GltrUnavailableError):
        model.load()


def test_analyzer_availability_is_honest():
    analyzer = GltrAnalyzer.from_config(
        {"model_name": "definitely-not-a-real-model-zzz", "model_dir": None}
    )
    available, reason = analyzer.availability()
    assert available is False
    assert reason


def test_estimate_requirements_shape():
    estimate = estimate_requirements(
        ["a" * 400, "b" * 400],
        GltrConfig(
            model_name="definitely-not-a-real-model-zzz",
            model_dir=None,
            max_length=512,
            batch_size=2,
        ),
    )
    for key in (
        "documents", "total_characters", "approx_tokens", "approx_batches",
        "estimated_seconds", "model_available", "availability_reason",
    ):
        assert key in estimate
    assert estimate["documents"] == 2
    assert estimate["approx_tokens"] > 0
    assert estimate["model_available"] is False  # fake model, no local dir


def test_offline_environment_flags_are_set():
    import os

    from gltr.local_model import _set_offline_environment

    _set_offline_environment(allow_download=False)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


# --------------------------------------------------------------------------- #
# Windowed extraction and batching (deterministic fake model — no network)
# --------------------------------------------------------------------------- #
class _CharTokenizer:
    """Deterministic stand-in tokenizer: one id per character plus a leading id."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    name_or_path = "fake-char-tokenizer"
    vocab_size = 64

    def __call__(self, texts, padding=False, truncation=False, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        for text in texts:
            ids = [1] if add_special_tokens else []
            ids += [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]
            input_ids.append(ids)
        return {"input_ids": input_ids}


class _FakeCausalLM:
    """Pseudo-logits that depend only on the token ids, so batching is exact."""

    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self._tokenizer = _CharTokenizer()
        self.calls = []

    def forward(self, input_ids, attention_mask=None):
        import torch

        ids = input_ids.to(torch.float32)
        vocab = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        # Peak logit at the true token id -> well-defined ranks/probabilities.
        logits = -((vocab - ids.unsqueeze(-1)) ** 2) / 4.0
        self.calls.append(tuple(input_ids.shape))
        return logits

    def describe(self):
        return {"model_name": "fake"}


def _fake_analyzer(max_length=8, stride=4, batch_size=2):
    analyzer = GltrAnalyzer(
        config=GltrConfig(
            model_dir=None, max_length=max_length, stride=stride, batch_size=batch_size
        )
    )
    analyzer.model = _FakeCausalLM()
    analyzer._loaded = True
    return analyzer


def test_windows_cover_every_target_exactly_once():
    analyzer = _fake_analyzer(max_length=8, stride=4)
    ids = list(range(30))
    specs = analyzer._windows_for_ids(ids)
    covered = np.concatenate([kept for _, _, kept in specs]).tolist()
    assert sorted(covered) == list(range(1, len(ids)))
    assert len(covered) == len(set(covered))  # never double-counted


def test_zero_stride_is_non_overlapping():
    analyzer = _fake_analyzer(max_length=4, stride=0)
    specs = analyzer._windows_for_ids(list(range(10)))
    starts = [start for start, _, _ in specs]
    assert starts == [0, 4, 8]


def test_batched_matches_single_text_extraction():
    pytest.importorskip("torch")
    texts = ["hello world", "a longer sentence to tokenize here", "", "x", "ok"]
    single = list(_fake_analyzer(batch_size=1).extract_many(texts, batch_size=1))
    batched = list(_fake_analyzer(batch_size=3).extract_many(texts, batch_size=3))
    assert len(single) == len(batched) == len(texts)
    for row_single, row_batched in zip(single, batched):
        assert set(row_single) == set(row_batched)
        for key in row_single:
            if row_single[key] is None:
                assert row_batched[key] is None
            else:
                assert row_batched[key] == pytest.approx(row_single[key])


def test_empty_and_short_text_are_handled():
    pytest.importorskip("torch")
    analyzer = _fake_analyzer()
    empty = analyzer.extract("")
    assert empty["gltr_token_count"] == 0
    assert empty["gltr_mean_prob"] is None
    short = analyzer.extract("x")  # one real token after the leading id
    assert short["gltr_token_count"] == 1
    assert short["gltr_mean_prob"] is not None
    assert math.isfinite(short["gltr_mean_prob"])


def test_gltr_checkpoint_roundtrip_and_config_guard(tmp_path):
    pytest.importorskip("torch")
    from scripts import fingerprint_pipeline as fp

    analyzer = _fake_analyzer()
    config = {"gltr_version": "test-v1"}
    path = tmp_path / "gltr_features.checkpoint.csv"
    processed = {
        "s1": {"sample_id": "s1", **analyzer.extract("hello world")},
        "s2": {"sample_id": "s2", **analyzer.extract("another sample text")},
    }
    fp._write_gltr_checkpoint(path, processed, analyzer, config)
    assert path.exists()
    assert path.with_suffix(path.suffix + ".meta.json").exists()

    loaded = fp._load_gltr_checkpoint(path, analyzer, config)
    assert set(loaded) == {"s1", "s2"}
    # A different feature/model config must invalidate the checkpoint.
    assert fp._load_gltr_checkpoint(path, analyzer, {"gltr_version": "other"}) == {}


def test_real_local_gpt2_inference_if_downloaded():
    """Exercise the real GPT-2 path when models/gltr/gpt2 has been populated."""
    from gltr.local_model import DEFAULT_MODEL_DIR, local_dir_ready

    ready, reason = local_dir_ready(DEFAULT_MODEL_DIR)
    if not ready:
        pytest.skip(f"no local GPT-2 at {DEFAULT_MODEL_DIR}: {reason}")
    analyzer = GltrAnalyzer.from_config(
        {"model_dir": DEFAULT_MODEL_DIR, "max_length": 64, "batch_size": 2}
    )
    features = analyzer.extract("The quick brown fox jumps over the lazy dog.")
    assert features["gltr_token_count"] > 0
    for name in analyzer.feature_names:
        assert features[name] is not None, name
        assert math.isfinite(features[name]), name
    assert 0.0 <= features["gltr_mean_prob"] <= 1.0
    assert features["gltr_perplexity"] >= 1.0
    assert 0.0 <= features["gltr_frac_rank_le_1000"] <= 1.0
