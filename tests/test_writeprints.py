"""Module B tests — Writeprints-style sparse stylometric features.

Synthetic text only. The vocabulary is fit on a tiny in-memory corpus so the
test is deterministic and independent of the pilot data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fingerprinting.writeprints import GROUPS, WriteprintsExtractor  # noqa: E402
from shared.preprocessing.text import normalize_text, tokenize  # noqa: E402

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "The cat sat on the mat; the dog slept.",
    "Dogs and cats run fast, but birds fly higher!",
    "A long sentence with several clauses, commas, and a final period.",
]

LIMITS = {
    "char_unigram": 12,
    "char_bigram": 12,
    "char_trigram": 12,
    "word_unigram": 12,
    "word_bigram": 12,
    "pos_unigram": 8,
    "pos_bigram": 8,
    "punct_pattern": 6,
}


def _tokens(text: str):
    return tokenize(normalize_text(text))


def _fitted(normalization: str = "frequency") -> WriteprintsExtractor:
    extractor = WriteprintsExtractor(
        config={"vocab_limits": LIMITS, "normalization": normalization}
    )
    extractor.fit(CORPUS, [_tokens(text) for text in CORPUS])
    return extractor


def test_fit_is_bounded_by_vocab_limits():
    extractor = _fitted()
    vocab = extractor.vocabulary()["groups"]
    assert set(vocab) == set(GROUPS)
    for group, limit in LIMITS.items():
        assert len(vocab[group]) <= limit


def test_transform_before_fit_raises():
    extractor = WriteprintsExtractor(config={"vocab_limits": LIMITS})
    with pytest.raises(RuntimeError):
        extractor.transform(CORPUS[0], _tokens(CORPUS[0]))


def test_feature_names_match_transform_row():
    extractor = _fitted()
    names = extractor.feature_names()
    row = extractor.transform(CORPUS[0], _tokens(CORPUS[0]))
    assert list(row.keys()) == names
    assert len(names) == len(row)


def test_frequency_normalization_in_unit_interval():
    extractor = _fitted("frequency")
    row = extractor.transform(CORPUS[0], _tokens(CORPUS[0]))
    for name, value in row.items():
        if value is None or name.startswith("wp_summary"):
            continue
        assert 0.0 <= value <= 1.0 + 1e-9, name


def test_l2_normalization_has_unit_norm():
    extractor = _fitted("l2")
    vector = extractor.sparse_row(CORPUS[0], _tokens(CORPUS[0]))
    norm = float(np.linalg.norm(vector))
    assert pytest.approx(norm, abs=1e-6) == 1.0 or norm == 0.0


def test_tfidf_is_non_negative():
    extractor = _fitted("tfidf")
    vector = extractor.sparse_row(CORPUS[0], _tokens(CORPUS[0]))
    assert np.all(vector >= 0.0)


def test_empty_text_is_defined():
    extractor = _fitted()
    row = extractor.transform("", [])
    assert row["wp_summary_ttr"] is None
    assert row["wp_summary_mean_word_length"] is None
    # The vocabulary columns are all present and zero (no mass to normalize).
    assert row["wp_char_unigram_000"] == 0.0


def test_repeated_vocabulary_and_punctuation_patterns():
    extractor = _fitted()
    repeated = extractor.transform("spam spam spam spam", _tokens("spam spam spam spam"))
    assert repeated["wp_summary_ttr"] == pytest.approx(0.25)
    punct = extractor.transform("Hi!!! Yes???", _tokens("Hi!!! Yes???"))
    assert punct["wp_summary_char_entropy"] is not None


def test_fit_is_deterministic():
    first = _fitted().vocabulary()["groups"]
    second = _fitted().vocabulary()["groups"]
    assert first == second


def test_no_raw_ngram_in_column_names():
    extractor = _fitted()
    for name in extractor.feature_names():
        assert "\u0001" not in name
        assert " " not in name
