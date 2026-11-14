"""Module B — Writeprints-style stylometric fingerprints.

A **Writeprints-inspired** implementation: character n-grams, word n-grams,
function-word frequencies, POS n-grams and punctuation patterns, selected to a
bounded vocabulary by document frequency. It is *not* a claim that the original
proprietary/published Writeprints methodology has been reproduced exactly.

Design
------
* The vocabulary is fit on a corpus (the pilot) by document frequency, with a
  deterministic name tie-break, and capped per group by ``vocab_limits``. In a
  real experiment the vocabulary must be fit on the **train split only**; for
  this pilot (feature extraction only, no training) it is fit on the pilot and
  that scope is recorded in the vocabulary artifact.
* Columns are named ``wp_<group>_<index>`` so a header never contains raw
  n-grams; the mapping index -> n-gram lives in the vocabulary JSON.
* The selected high-dimensional n-gram block is also written as a SciPy sparse
  matrix so downstream code can use it without the dense CSV.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from shared.preprocessing.text import Token, normalize_text

from fingerprinting.stats_utils import counter_from, safe_div, stdev

GROUPS: tuple[str, ...] = (
    "char_unigram",
    "char_bigram",
    "char_trigram",
    "word_unigram",
    "word_bigram",
    "pos_unigram",
    "pos_bigram",
    "punct_pattern",
)

DEFAULT_VOCAB_LIMITS: dict[str, int] = {
    "char_unigram": 30,
    "char_bigram": 40,
    "char_trigram": 40,
    "word_unigram": 60,
    "word_bigram": 30,
    "pos_unigram": 20,
    "pos_bigram": 24,
    "punct_pattern": 16,
}

DEFAULT_FUNCTION_WORDS: tuple[str, ...] = (
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what",
)

_PUNCT_RUN_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def _char_ngrams(text: str, size: int) -> list[str]:
    if size <= 0 or len(text) < size:
        return []
    return [text[index : index + size] for index in range(len(text) - size + 1)]


def _seq_ngrams(items: Sequence[str], size: int) -> list[str]:
    if size <= 0 or len(items) < size:
        return []
    if size == 1:
        return list(items)
    return ["\u0001".join(items[index : index + size]) for index in range(len(items) - size + 1)]


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_").lower() or "empty"


@dataclass
class WriteprintsExtractor:
    config: dict[str, Any] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)
    document_frequency: dict[str, Counter] = field(default_factory=dict)
    corpus_size: int = 0
    function_words: tuple[str, ...] = DEFAULT_FUNCTION_WORDS
    _fitted: bool = False

    def __post_init__(self) -> None:
        self.vocab_limits = dict(
            self.config.get("vocab_limits", DEFAULT_VOCAB_LIMITS)
        )
        # Coerce to str defensively: a YAML file can parse bare `on`/`no`/`off`
        # as booleans, and a bool would silently break token matching.
        self.function_words = tuple(
            str(word).lower()
            for word in self.config.get("function_words", DEFAULT_FUNCTION_WORDS)
        )
        self.normalization = str(self.config.get("normalization", "frequency"))
        self.min_df = int(self.config.get("min_document_frequency", 1))

    # -- per-document n-gram counters --------------------------------------- #
    def _group_counters(
        self,
        text: str,
        tokens: Sequence[Token],
        pos_tags: Sequence[str] | None,
    ) -> dict[str, Counter[str]]:
        normalized = normalize_text(text)
        lowered = normalized.lower()
        words = [token.text for token in tokens]
        tags = list(pos_tags) if pos_tags is not None else []

        counters: dict[str, Counter[str]] = {
            "char_unigram": counter_from(_char_ngrams(lowered, 1)),
            "char_bigram": counter_from(_char_ngrams(lowered, 2)),
            "char_trigram": counter_from(_char_ngrams(lowered, 3)),
            "word_unigram": counter_from(words),
            "word_bigram": counter_from(_seq_ngrams(words, 2)),
            "pos_unigram": counter_from(tags),
            "pos_bigram": counter_from(_seq_ngrams(tags, 2)),
            "punct_pattern": counter_from(_PUNCT_RUN_RE.findall(normalized)),
        }
        return counters

    # -- fit / transform ----------------------------------------------------- #
    def fit(
        self,
        texts: Iterable[str],
        token_lists: Iterable[Sequence[Token]],
        pos_lists: Iterable[Sequence[str] | None] | None = None,
    ) -> "WriteprintsExtractor":
        texts = list(texts)
        token_lists = list(token_lists)
        pos_lists = list(pos_lists) if pos_lists is not None else [None] * len(texts)

        df: dict[str, Counter] = {group: Counter() for group in GROUPS}
        for text, tokens, tags in zip(texts, token_lists, pos_lists):
            counters = self._group_counters(text, tokens, tags)
            for group in GROUPS:
                for ngram in counters[group]:
                    df[group][ngram] += 1

        self.document_frequency = df
        self.corpus_size = len(texts)
        self.vocab = {}
        for group in GROUPS:
            limit = int(self.vocab_limits.get(group, 0))
            ranked = sorted(
                ((ngram, freq) for ngram, freq in df[group].items() if freq >= self.min_df),
                key=lambda item: (-item[1], item[0]),
            )
            self.vocab[group] = [ngram for ngram, _ in ranked[:limit]]
        self._fitted = True
        return self

    def _idf(self, group: str, ngram: str) -> float:
        df = self.document_frequency.get(group, Counter()).get(ngram, 0)
        if self.corpus_size <= 0 or df <= 0:
            return 0.0
        return float(np.log((1 + self.corpus_size) / (1 + df)) + 1.0)

    def _normalized(self, group: str, counts: Counter[str]) -> dict[str, float]:
        total = float(sum(counts.values()))
        values: dict[str, float] = {}
        for ngram in self.vocab.get(group, []):
            count = float(counts.get(ngram, 0))
            if self.normalization == "raw_count":
                values[ngram] = count
            elif self.normalization == "tfidf":
                tf = safe_div(count, total) or 0.0
                values[ngram] = tf * self._idf(group, ngram)
            else:  # "frequency" and anything else falls back to relative frequency
                values[ngram] = safe_div(count, total) or 0.0
        return values

    def transform(
        self,
        text: str,
        tokens: Sequence[Token],
        pos_tags: Sequence[str] | None = None,
    ) -> dict[str, float | int | None]:
        if not self._fitted:
            raise RuntimeError("WriteprintsExtractor.fit must be called before transform")

        counters = self._group_counters(text, tokens, pos_tags)
        features: dict[str, float | int | None] = {}

        for group in GROUPS:
            values = self._normalized(group, counters[group])
            for index, ngram in enumerate(self.vocab.get(group, [])):
                features[f"wp_{group}_{index:03d}"] = values.get(ngram, 0.0)

        if self.normalization == "l2":
            ngram_keys = [key for key in features if key.startswith("wp_char") or key.startswith("wp_word") or key.startswith("wp_pos") or key.startswith("wp_punct")]
            norm = float(np.sqrt(sum(float(features[key]) ** 2 for key in ngram_keys)))
            if norm > 0:
                for key in ngram_keys:
                    features[key] = float(features[key]) / norm

        # Fixed function-word frequencies (relative to the word-token count).
        word_total = len(tokens)
        for word in self.function_words:
            features[f"wp_fw_{_sanitize_token(word)}"] = safe_div(
                counters["word_unigram"].get(word, 0), word_total
            )

        # Compact style summaries (always defined).
        word_lengths = [len(token.text) for token in tokens]
        unique_words = len({token.text for token in tokens})
        features["wp_summary_ttr"] = safe_div(unique_words, word_total)
        features["wp_summary_mean_word_length"] = (
            float(np.mean(word_lengths)) if word_lengths else None
        )
        features["wp_summary_word_length_std"] = stdev(word_lengths)
        features["wp_summary_char_entropy"] = _char_entropy(text)
        return features

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for group in GROUPS:
            names.extend(f"wp_{group}_{index:03d}" for index in range(len(self.vocab.get(group, []))))
        names.extend(f"wp_fw_{_sanitize_token(word)}" for word in self.function_words)
        names.extend(
            [
                "wp_summary_ttr",
                "wp_summary_mean_word_length",
                "wp_summary_word_length_std",
                "wp_summary_char_entropy",
            ]
        )
        return names

    def vocabulary(self) -> dict[str, Any]:
        return {
            "groups": {group: list(self.vocab.get(group, [])) for group in GROUPS},
            "vocab_limits": dict(self.vocab_limits),
            "normalization": self.normalization,
            "min_document_frequency": self.min_df,
            "corpus_size": self.corpus_size,
            "fit_scope": "pilot_corpus (must be train-only for a real experiment)",
            "function_words": list(self.function_words),
        }

    def sparse_row(
        self, text: str, tokens: Sequence[Token], pos_tags: Sequence[str] | None = None
    ) -> np.ndarray:
        """Dense vector over the concatenated n-gram vocabulary (for the sparse block)."""
        counters = self._group_counters(text, tokens, pos_tags)
        vector: list[float] = []
        for group in GROUPS:
            values = self._normalized(group, counters[group])
            vector.extend(values.get(ngram, 0.0) for ngram in self.vocab.get(group, []))
        array = np.asarray(vector, dtype=np.float64)
        if self.normalization == "l2":
            norm = float(np.sqrt(np.sum(array ** 2)))
            if norm > 0:
                array = array / norm
        return array


def _char_entropy(text: str) -> float | None:
    from fingerprinting.stats_utils import entropy

    lowered = normalize_text(text).lower()
    if not lowered:
        return None
    return entropy(counter_from(lowered).values())


__all__ = [
    "DEFAULT_FUNCTION_WORDS",
    "DEFAULT_VOCAB_LIMITS",
    "GROUPS",
    "WriteprintsExtractor",
]
