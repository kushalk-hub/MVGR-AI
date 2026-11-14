"""GLTR analysis pipeline: tokenization, batched inference, document aggregation.

The analyzer is the single entry point used by the extraction script. It is
honest about availability: :meth:`GltrAnalyzer.available` returns a reason when
no local model is present, and :meth:`GltrAnalyzer.extract` refuses to run
rather than silently downloading a model.

Inference is batched with dynamic padding and an attention mask. Each text is
tokenized without truncation, then split into (optionally overlapping) windows
of ``max_length`` tokens. For every window the model produces next-token logits
once; the per-token probability, rank and entropy are read off the causally
shifted distribution. Overlap tokens are counted exactly once so a long document
is neither double-counted nor truncated away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from gltr.document_features import aggregate_from_arrays, feature_names
from gltr.entropy_analysis import next_token_entropies
from gltr.local_model import GltrConfig, GltrModel, GltrUnavailableError
from gltr.token_probability import next_token_probabilities
from gltr.token_rank import token_ranks

# Rough CPU/GPU throughput used only for a pre-run estimate, not a measurement.
_ASSUMED_TOKENS_PER_SECOND = {"cpu": 40.0, "cuda": 1500.0, "auto": 40.0}
# Rough characters-per-token ratio for an English BPE tokenizer, for estimation.
_APPROX_CHARS_PER_TOKEN = 4.0


def _token_stats(logits: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(probabilities, ranks, entropies)`` for the next-token predictions.

    ``logits`` is ``[seq, vocab]`` and ``ids`` is ``[seq]``; the returned arrays
    have length ``seq - 1`` (one entry per predicted token). Pure NumPy, so it is
    unit-testable without torch.
    """
    probabilities, _ = next_token_probabilities(logits, ids)
    ranks = token_ranks(logits, ids)
    entropies = next_token_entropies(logits, ids)
    return probabilities, ranks, entropies


def estimate_requirements(texts: Iterable[str], config: GltrConfig) -> dict[str, Any]:
    """Estimate model requirements and processing time **without** running it.

    Token counts are approximated from character counts; this is an estimate to
    decide whether to run, not a measurement.
    """
    texts = list(texts)
    total_chars = sum(len(text) for text in texts)
    approx_tokens = int(math.ceil(total_chars / _APPROX_CHARS_PER_TOKEN))
    windows = max(1, int(math.ceil(approx_tokens / max(1, config.max_length))))
    batches = int(math.ceil(windows / max(1, config.batch_size)))
    throughput = _ASSUMED_TOKENS_PER_SECOND.get(config.device, 40.0)
    estimated_seconds = approx_tokens / throughput if throughput else None

    available, reason = GltrModel(config=config).availability()
    return {
        "model_name": config.model_name,
        "revision": config.revision,
        "model_dir": config.model_dir,
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

    # ------------------------------------------------------------------ #
    # windowing
    # ------------------------------------------------------------------ #
    def _windows_for_ids(
        self, ids: Sequence[int]
    ) -> list[tuple[int, list[int], np.ndarray]]:
        """Split a token id sequence into windows.

        Returns a list of ``(start, window_ids, kept_global_targets)``. A target
        token is kept in exactly one window (the first that covers it), so the
        overlap created by ``stride`` never double-counts a token.
        """
        n = len(ids)
        windows: list[tuple[int, list[int], np.ndarray]] = []
        if n < 2:
            return windows
        max_length = max(2, int(self.config.max_length))
        stride = int(self.config.stride)
        if stride <= 0:
            stride = max_length
        stride = min(stride, max_length)
        covered = 0
        start = 0
        while True:
            end = min(start + max_length, n)
            window = list(ids[start:end])
            if len(window) >= 2:
                local_targets = np.arange(1, len(window))
                global_targets = start + local_targets
                keep = global_targets > covered
                if keep.any():
                    windows.append((start, window, global_targets[keep]))
                covered = max(covered, end - 1)
            if end >= n:
                break
            start += stride
        return windows

    def _run_window_batch(
        self, specs: list[tuple[int, list[int], np.ndarray]]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run one dynamically-padded batch of windows through the model."""
        import torch

        assert self.model is not None
        lengths = [len(window) for _, window, _ in specs]
        maxlen = max(lengths)
        ids = np.zeros((len(specs), maxlen), dtype=np.int64)
        mask = np.zeros((len(specs), maxlen), dtype=np.int64)
        for row, (_, window, _) in enumerate(specs):
            ids[row, : len(window)] = window
            mask[row, : len(window)] = 1

        logits = self.model.forward(
            torch.from_numpy(ids), torch.from_numpy(mask)
        )
        logits = logits.detach().to("cpu").numpy()

        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for row, (start, window, kept) in enumerate(specs):
            length = len(window)
            row_logits = logits[row, :length, :]
            probabilities, ranks, entropies = _token_stats(row_logits, np.asarray(window))
            # probabilities index j corresponds to the local target j + 1, i.e.
            # global target start + j + 1; invert for the kept targets.
            index = kept - start - 1
            results.append((probabilities[index], ranks[index], entropies[index]))
        del logits
        return results

    # ------------------------------------------------------------------ #
    # extraction
    # ------------------------------------------------------------------ #
    def extract_many(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        progress: Any = None,
    ) -> Iterator[dict[str, float | int | None]]:
        """Yield one feature mapping per text, in order."""
        if not self._loaded:
            self.load()
        assert self.model is not None
        tokenizer = self.model._tokenizer  # noqa: SLF001
        rank_thresholds = self.config.rank_thresholds
        batch_size = max(1, int(batch_size or self.config.batch_size or 1))
        texts = list(texts)

        for offset in range(0, len(texts), batch_size):
            chunk = texts[offset : offset + batch_size]
            encoded = tokenizer(
                chunk,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            id_lists = encoded["input_ids"]

            per_doc: list[list[list[np.ndarray]]] = [[[], [], []] for _ in chunk]
            specs: list[tuple[int, list[int], np.ndarray]] = []
            owners: list[int] = []
            for doc, ids in enumerate(id_lists):
                for spec in self._windows_for_ids([int(value) for value in ids]):
                    specs.append(spec)
                    owners.append(doc)

            for start in range(0, len(specs), batch_size):
                batch_specs = specs[start : start + batch_size]
                batch_owners = owners[start : start + batch_size]
                results = self._run_window_batch(batch_specs)
                for doc, (probability, rank, entropy) in zip(batch_owners, results):
                    per_doc[doc][0].append(probability)
                    per_doc[doc][1].append(rank)
                    per_doc[doc][2].append(entropy)

            for doc in range(len(chunk)):
                probabilities = (
                    np.concatenate(per_doc[doc][0]) if per_doc[doc][0] else np.array([])
                )
                ranks = np.concatenate(per_doc[doc][1]) if per_doc[doc][1] else np.array([])
                entropies = (
                    np.concatenate(per_doc[doc][2]) if per_doc[doc][2] else np.array([])
                )
                yield aggregate_from_arrays(
                    probabilities, ranks, entropies, rank_thresholds=rank_thresholds
                )
            if progress is not None:
                progress(offset + len(chunk), len(texts))

    def extract(self, text: str) -> dict[str, float | int | None]:
        return next(self.extract_many([text]))


__all__ = ["GltrAnalyzer", "GltrUnavailableError", "estimate_requirements"]
