"""Unigram green-list watermark detection (Kirchenbauer-style, research use).

A generator using this scheme biases each token toward a pseudo-random "green
list" of fraction ``gamma`` derived from a secret key and the previous token.
The detector re-derives the same green list from the key and counts how many
realised tokens are green. Under the null (no watermark) the count is
``Binomial(n, gamma)``; a large positive deviation is evidence that *this*
scheme was used.

This works without the generator's language model because greenness is decided
by hashing the token string itself, so the detector is self-contained and
offline. It cannot detect any other watermark and it must not be used to infer
authorship. The key is supplied at runtime (never stored, never logged).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from shared.preprocessing.text import Token, normalize_text, tokenize

from watermark.statistics import binomial_p_value, normal_survival, z_score
from watermark.verifier import DetectionStatus, WatermarkResult

DEFAULT_GAMMA = 0.25
DEFAULT_WINDOW = 1
DEFAULT_Z_THRESHOLD = 4.0
DEFAULT_P_VALUE_THRESHOLD = 1.0e-5
MIN_TOKENS_FOR_TEST = 20


def _hash_unit_interval(key: str, context: str, token: str, seed: int) -> float:
    payload = f"{key}\u0000{seed}\u0000{context}\u0000{token}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # 8 bytes -> integer -> [0, 1)
    value = int.from_bytes(digest[:8], "big")
    return value / float(1 << 64)


def is_green(key: str, context: str, token: str, *, gamma: float, seed: int) -> bool:
    return _hash_unit_interval(key, context, token, seed) < gamma


@dataclass
class UnigramWatermarkVerifier:
    provider: str = "unigram_greenlist"
    gamma: float = DEFAULT_GAMMA
    window: int = DEFAULT_WINDOW
    hash_seed: int = 0
    z_threshold: float = DEFAULT_Z_THRESHOLD
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "UnigramWatermarkVerifier":
        config = config or {}
        return cls(
            gamma=float(config.get("gamma", DEFAULT_GAMMA)),
            window=int(config.get("window", DEFAULT_WINDOW)),
            hash_seed=int(config.get("hash_seed", 0)),
            z_threshold=float(config.get("z_threshold", DEFAULT_Z_THRESHOLD)),
            p_value_threshold=float(config.get("p_value_threshold", DEFAULT_P_VALUE_THRESHOLD)),
        )

    def verify(
        self,
        text: str,
        *,
        key: str | None = None,
        tokens: Sequence[Token] | None = None,
    ) -> WatermarkResult:
        if not key:
            return WatermarkResult(
                status=DetectionStatus.UNSUPPORTED,
                provider=self.provider,
                confidence=None,
                details=(
                    "No watermark key is configured. Unigram green-list verification "
                    "requires the key used at generation time."
                ),
                assumptions=(
                    "Verification is only meaningful for the exact scheme and key used.",
                    "Absence of detection is not evidence of human authorship.",
                ),
            )

        if tokens is None:
            tokens = tokenize(normalize_text(text))
        surfaces = [token.text for token in tokens]
        if len(surfaces) <= self.window:
            return WatermarkResult(
                status=DetectionStatus.INCONCLUSIVE,
                provider=self.provider,
                details="Text is too short to test the unigram watermark.",
            )

        green = 0
        tested = 0
        for index in range(self.window, len(surfaces)):
            context = " ".join(surfaces[max(0, index - self.window) : index])
            if is_green(key, context, surfaces[index], gamma=self.gamma, seed=self.hash_seed):
                green += 1
            tested += 1

        expected = self.gamma * tested
        variance = tested * self.gamma * (1.0 - self.gamma)
        z = z_score(green, expected, variance)
        p_value = normal_survival(z) if z is not None else None
        exact_p = binomial_p_value(green, tested, self.gamma)

        evidence = {
            "scheme": "unigram_greenlist",
            "gamma": self.gamma,
            "window": self.window,
            "watermarked_token_count": tested,
            "green_list_token_count": green,
            "green_list_proportion": (green / tested) if tested else None,
            "expected_green_proportion": self.gamma,
            "z_score": z,
            "p_value_normal": p_value,
            "p_value_exact": exact_p,
        }

        if tested < MIN_TOKENS_FOR_TEST:
            return WatermarkResult(
                status=DetectionStatus.INCONCLUSIVE,
                provider=self.provider,
                confidence=None,
                details=(
                    f"Only {tested} tokens are available; at least {MIN_TOKENS_FOR_TEST} "
                    "are required for a reliable test."
                ),
                evidence=evidence,
                assumptions=("Small samples give unstable statistics.",),
            )

        detected = z is not None and z >= self.z_threshold and (
            exact_p is None or exact_p <= self.p_value_threshold
        )
        if detected:
            return WatermarkResult(
                status=DetectionStatus.DETECTED,
                provider=self.provider,
                confidence=min(1.0, max(0.0, 1.0 - (exact_p or 0.0))),
                details=(
                    f"Green-list proportion {green / tested:.3f} vs expected "
                    f"{self.gamma:.3f} (z={z:.2f}, p={exact_p:.2e}). Consistent with "
                    "the tested unigram watermark scheme."
                ),
                evidence=evidence,
                assumptions=(
                    "A detected watermark indicates this scheme was applied; it is not "
                    "an authorship judgement.",
                ),
            )
        return WatermarkResult(
            status=DetectionStatus.NOT_DETECTED,
            provider=self.provider,
            confidence=None,
            details=(
                f"No green-list bias detected (z={z:.2f}). This concerns only the "
                "tested unigram scheme and is not evidence of human authorship."
            ),
            evidence=evidence,
            assumptions=(
                "A not_detected result only rules out this scheme with this key.",
            ),
        )


__all__ = [
    "DEFAULT_GAMMA",
    "DEFAULT_P_VALUE_THRESHOLD",
    "DEFAULT_WINDOW",
    "DEFAULT_Z_THRESHOLD",
    "MIN_TOKENS_FOR_TEST",
    "UnigramWatermarkVerifier",
    "is_green",
]
