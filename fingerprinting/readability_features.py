"""Module A(f) — readability features.

The two Flesch formulas are computed from word, sentence and syllable counts.
Syllables come from the deterministic heuristic in ``stats_utils`` (no external
dictionary), so the branch stays offline. All four features are ``None`` when
the text has no words or no sentences.
"""

from __future__ import annotations

from typing import Any, Sequence

from shared.preprocessing.text import Token

from fingerprinting.stats_utils import count_syllables, safe_div


def extract(
    text: str,
    *,
    tokens: Sequence[Token],
    sentences: Sequence[str],
    doc: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    n_words = len(tokens)
    n_sentences = len(sentences)
    syllables = sum(count_syllables(token.text) for token in tokens)

    words_per_sentence = safe_div(n_words, n_sentences)
    syllables_per_word = safe_div(syllables, n_words)

    reading_ease = None
    grade = None
    if words_per_sentence is not None and syllables_per_word is not None:
        reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59

    return {
        "read_syllable_count": syllables,
        "read_avg_syllables_per_word": syllables_per_word,
        "read_avg_words_per_sentence": words_per_sentence,
        "read_flesch_reading_ease": reading_ease,
        "read_flesch_kincaid_grade": grade,
    }


__all__ = ["extract"]
