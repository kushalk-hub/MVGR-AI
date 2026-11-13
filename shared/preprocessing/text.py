"""Shared text preprocessing — tokenizer.v1.

This module is the ONLY place where text is normalized or tokenized. Person 2's
graph builders and Person 1's audit/fingerprinting must call it, otherwise the
same text can produce different graphs in different pipelines.

Any change to these rules requires a new ``TOKENIZER_VERSION`` and a new graph
spec version.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from shared.contracts.versions import TOKENIZER_VERSION

# ``tokenizer.v1`` token pattern: unicode word characters, optionally joined by
# apostrophes or hyphens. Punctuation and emoji are not tokens.
TOKEN_PATTERN = r"[^\W_]+(?:['\u2019-][^\W_]+)*"
TOKEN_RE = re.compile(TOKEN_PATTERN, flags=re.UNICODE)

# Fixed sentence boundary heuristic. It is deliberately simple and deterministic:
# a boundary is terminal punctuation followed by whitespace, or a newline run.
# Abbreviations such as "e.g." can create extra boundaries; this is documented
# and applies identically to every graph view and every experiment.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?\u2026])\s+|\n{2,}")

WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)

# tokenizer.v1 stopword lexicon. Frozen with the tokenizer version; changing it
# changes graph structure and therefore requires a version bump.
STOPWORDS_EN: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because
    been before being below between both but by can can't cannot could couldn't did
    didn't do does doesn't doing don't down during each few for from further had
    hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
    herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
    its itself let's me more most mustn't my myself no nor not of off on once only or
    other ought our ours ourselves out over own same shan't she she'd she'll she's
    should shouldn't so some such than that that's the their theirs them themselves
    then there there's these they they'd they'll they're they've this those through
    to too under until up very was wasn't we we'd we'll we're we've were weren't what
    what's when when's where where's which while who who's whom why why's with won't
    would wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)


def normalize_text(text: str) -> str:
    """NFKC normalization and whitespace collapse; no lowercasing."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    return WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_for_hashing(text: str) -> str:
    """Stable normalization used for exact/near-duplicate detection."""
    return normalize_text(text).casefold()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_normalized(text: str) -> str:
    return hashlib.sha256(normalize_for_hashing(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    position: int
    sentence_index: int
    is_stopword: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "token": self.text,
            "position": self.position,
            "sentence_index": self.sentence_index,
            "is_stopword": self.is_stopword,
        }


def split_sentences(normalized_text: str) -> list[str]:
    if not normalized_text:
        return []
    parts = SENTENCE_BOUNDARY_RE.split(normalized_text)
    return [part.strip() for part in parts if part and part.strip()]


def tokenize(normalized_text: str) -> list[Token]:
    """Tokenize already-normalized text into sentence-aware tokens.

    Positions are zero-based indexes over the token stream (not character
    offsets) so that ``distance = abs(position_a - position_b)`` is meaningful
    for a co-occurrence window.
    """
    tokens: list[Token] = []
    position = 0
    for sentence_index, sentence in enumerate(split_sentences(normalized_text)):
        for match in TOKEN_RE.finditer(sentence):
            surface = match.group(0)
            folded = surface.casefold()
            tokens.append(
                Token(
                    text=folded,
                    position=position,
                    sentence_index=sentence_index,
                    is_stopword=folded in STOPWORDS_EN,
                )
            )
            position += 1
    return tokens


def iter_shingles(tokens: Iterable[str], size: int) -> Iterable[tuple[str, ...]]:
    tokens = list(tokens)
    if size <= 0:
        raise ValueError("shingle size must be positive")
    if len(tokens) < size:
        return
    for index in range(len(tokens) - size + 1):
        yield tuple(tokens[index : index + size])


def word_count(text: str) -> int:
    return sum(1 for _ in TOKEN_RE.finditer(normalize_text(text)))


__all__ = [
    "STOPWORDS_EN",
    "TOKEN_PATTERN",
    "TOKEN_RE",
    "TOKENIZER_VERSION",
    "Token",
    "iter_shingles",
    "normalize_for_hashing",
    "normalize_text",
    "sha256_normalized",
    "sha256_text",
    "split_sentences",
    "tokenize",
    "word_count",
]
