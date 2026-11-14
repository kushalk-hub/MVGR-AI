"""Module A — stylometric text fingerprinting orchestrator.

Runs the six vocabulary / sentence / punctuation / POS / syntactic / readability
extractors over one text and returns a single flat feature mapping. The spaCy
model is loaded lazily and shared across calls; if it is missing, the POS and
syntactic features are emitted as ``None`` (the branch degrades honestly instead
of failing or inventing values).

This module extends the existing ``fingerprinting`` package. It does not touch
the SHA-256 / n-gram / MinHash similarity code in the same package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from shared.preprocessing.text import (
    Token,
    normalize_text,
    split_sentences,
    tokenize,
)

from fingerprinting import (
    pos_features,
    punctuation_features,
    readability_features,
    sentence_features,
    syntactic_features,
    vocabulary_features,
)

MODULES = (
    vocabulary_features,
    sentence_features,
    punctuation_features,
    pos_features,
    syntactic_features,
    readability_features,
)


class SpacyUnavailableError(RuntimeError):
    """Raised when a caller explicitly requires spaCy and it is not available."""


@dataclass
class StylometricExtractor:
    config: dict[str, Any] = field(default_factory=dict)
    spacy_model: str = "en_core_web_sm"
    _nlp: Any = None
    _spacy_error: str | None = None
    _spacy_loaded: bool = False

    def __post_init__(self) -> None:
        self.spacy_model = self.config.get("spacy_model", self.spacy_model)

    # -- spaCy --------------------------------------------------------------- #
    def _load_nlp(self) -> Any:
        if self._spacy_loaded:
            return self._nlp
        self._spacy_loaded = True
        try:
            import spacy  # noqa: PLC0415 - lazy optional dependency
        except ImportError as exc:  # pragma: no cover - spaCy is installed in CI
            self._spacy_error = f"spacy is not installed: {exc}"
            return None
        try:
            disable = list(self.config.get("spacy_disable", ["ner"]))
            self._nlp = spacy.load(self.spacy_model, disable=disable)
        except Exception as exc:  # noqa: BLE001 - any load failure is "unavailable"
            self._spacy_error = f"spaCy model {self.spacy_model!r} unavailable: {exc}"
            self._nlp = None
        return self._nlp

    @property
    def spacy_available(self) -> bool:
        return self._load_nlp() is not None

    def status(self) -> dict[str, Any]:
        return {
            "spacy_model": self.spacy_model,
            "spacy_available": self.spacy_available,
            "spacy_error": self._spacy_error,
        }

    # -- extraction ---------------------------------------------------------- #
    def extract(
        self,
        text: str,
        *,
        tokens: Sequence[Token] | None = None,
        sentences: Sequence[str] | None = None,
        doc: Any = None,
    ) -> dict[str, float | int | None]:
        normalized = normalize_text(text)
        if tokens is None:
            tokens = tokenize(normalized)
        if sentences is None:
            sentences = split_sentences(normalized)
        if doc is None:
            nlp = self._load_nlp()
            if nlp is not None:
                doc = nlp(normalized)

        features: dict[str, float | int | None] = {}
        for module in MODULES:
            features.update(
                module.extract(
                    normalized,
                    tokens=tokens,
                    sentences=sentences,
                    doc=doc,
                    config=self.config,
                )
            )
        return features

    def extract_many(self, texts: Iterable[str]) -> Iterator[dict[str, float | int | None]]:
        for text in texts:
            yield self.extract(text)

    def feature_names(self) -> list[str]:
        """The complete, ordered feature list (stable even without spaCy)."""
        return list(
            self.extract("", tokens=[], sentences=[], doc=None).keys()
        )


def feature_names(config: dict[str, Any] | None = None) -> list[str]:
    return StylometricExtractor(config=config or {}).feature_names()


__all__ = [
    "MODULES",
    "SpacyUnavailableError",
    "StylometricExtractor",
    "feature_names",
]
