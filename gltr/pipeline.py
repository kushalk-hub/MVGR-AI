"""GLTR analysis pipeline: tokenization, batched inference, document aggregation.

The analyzer is the single entry point used by the extraction script. It is
honest about availability: :meth:`GltrAnalyzer.available` returns a reason when
no local model is present, and :meth:`GltrAnalyzer.extract` refuses to run
rather than silently downloading a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np

from gltr.document_features import aggregate_from_arrays, feature_names
from gltr.entropy_analysis import next_token_entropies
from gltr.local_model import GltrConfig, GltrModel, GltrUnavailableError
from gltr.token_probability import next_token_probabilities
from gltr.token_rank import token_ranks

# Rough CPU/GPU throughput used only for a pre-run estimate, not a measurement.
_ASSUMED_TOKENS_PER_SECOND = {"cpu": 80.0, "cuda": 1500.0, "auto": 80.0}
# Rough characters-per-token ratio for an English BPE tokenizer, for estimation.
_APPROX_CHARS_PER_TOKEN = 4.0


def estimate_requirements(
    texts: Iterable[str], config: GltrConfig
) -> dict[str, Any]:
    """Estimate model requirements and processing time **without** running it.

    Token counts are approximated from character counts; this is an estimate to
    decide whether to run, not a measurement.
    """
    texts = list(texts)
    total_chars = sum(len(text) for text in texts)
    approx_tokens = int(math.ceil(total_chars / _APPROX_CHARS_PER_TOKEN))
    windows = max(1, int(math.ceil(approx_tokens / max(1, config.max_length))))
    batches = int(math.ceil(windows / max(1, config.batch_size)))
    throughput = _ASSUMED_TOKENS_PER_SECOND.get(config.device, 80.0)
    estimated_seconds = approx_tokens / throughput if throughput else None

    available, reason = GltrModel(config=config).availability()
    return {
        "model_name": config.model_name,
        "revision": config.revision,
        "device": config.device,
        "documents": len(texts),
        "total_characters": total_chars,
        "approx_tokens": approx_tokens,
        "approx_chars_per_token": _APPROX_CHARS_PER_TOKEN,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "approx_forward_passes": windows,
        "approx_batches": batches,
        "assumed_tokens_per_second": throughput,
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": (estimated_seconds / 60.0) if estimated_seconds else None,
        "model_available": available,
        "availability_reason": reason,
        "note": "Estimate only; token counts are approximated from characters.",
    }


@dataclass
class GltrAnalyzer:
    config: GltrConfig = field(default_factory=GltrConfig)
    model: GltrModel | None = None
    _loaded: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "GltrAnalyzer":
        return cls(config=GltrConfig.from_config(config))

    def availability(self) -> tuple[bool, str]:
        return GltrModel(config=self.config).availability()

    def load(self) -> "GltrAnalyzer":
        if not self._loaded:
            self.model = GltrModel(config=self.config).load()
            self._loaded = True
        return self

    @property
    def feature_names(self) -> list[str]:
        return feature_names(self.config.rank_thresholds)

    def _windows(self, input_ids: np.ndarray) -> list[np.ndarray]:
        length = int(input_ids.shape[0])
        if length <= self.config.max_length:
            return [input_ids]
        step = self.config.max_length - self.config.stride
        step = max(1, step)
        windows = []
        for start in range(0, length, step):
            chunk = input_ids[start : start + self.config.max_length]
            if chunk.shape[0] < 2:
                break
            windows.append(chunk)
            if start + self.config.max_length >= length:
                break
        return windows or [input_ids[: self.config.max_length]]

    def extract(self, text: str) -> dict[str, float | int | None]:
        if not self._loaded:
            self.load()
        assert self.model is not None and self.model._tokenizer is not None  # noqa: SLF001
        tokenizer = self.model._tokenizer  # noqa: SLF001

        encoded = tokenizer(
            text,
            return_tensors="np",
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = np.asarray(encoded["input_ids"])[0]

        prob_parts: list[np.ndarray] = []
        rank_parts: list[np.ndarray] = []
        entropy_parts: list[np.ndarray] = []
        for window in self._windows(input_ids):
            logits = self.model.next_token_logits(window[None, :])
            logits = np.asarray(logits)[0]
            probabilities, _ = next_token_probabilities(logits, window)
            ranks = token_ranks(logits, window)
            entropies = next_token_entropies(logits, window)
            prob_parts.append(probabilities)
            rank_parts.append(ranks)
            entropy_parts.append(entropies)

        if not prob_parts:
            return aggregate_from_arrays(
                np.array([]), np.array([]), np.array([]),
                rank_thresholds=self.config.rank_thresholds,
            )
        return aggregate_from_arrays(
            np.concatenate(prob_parts),
            np.concatenate(rank_parts),
            np.concatenate(entropy_parts),
            rank_thresholds=self.config.rank_thresholds,
        )

    def extract_many(self, texts: Iterable[str]) -> Iterator[dict[str, float | int | None]]:
        for text in texts:
            yield self.extract(text)


__all__ = ["GltrAnalyzer", "GltrUnavailableError", "estimate_requirements"]
