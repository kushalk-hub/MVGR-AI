"""Watermark verification interface (Person 1).

There is no generic AI watermark. Verification is only meaningful for a
specific provider, algorithm, and key/generation protocol. The default
implementation returns ``unsupported``, which is never evidence of human
authorship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class DetectionStatus(str, Enum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class WatermarkResult:
    status: DetectionStatus
    provider: str
    confidence: float | None = None
    details: str = ""
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "provider": self.provider,
            "confidence": self.confidence,
            "details": self.details,
            "assumptions": list(self.assumptions),
        }

    @property
    def supports_human_authorship(self) -> bool:
        return False


class WatermarkVerifier(Protocol):
    provider: str

    def verify(self, text: str, *, key: str | None = None) -> WatermarkResult:
        ...


class UnsupportedVerifier:
    provider = "unsupported"

    def verify(self, text: str, *, key: str | None = None) -> WatermarkResult:
        return WatermarkResult(
            status=DetectionStatus.UNSUPPORTED,
            provider=self.provider,
            confidence=None,
            details=(
                "No watermark provider is configured. Most text has no verifiable watermark, "
                "and absence of detection does not indicate human authorship."
            ),
            assumptions=(
                "Watermark verification requires a specific provider algorithm, key and generation protocol.",
                "A not_detected result only concerns the tested algorithm.",
            ),
        )


_REGISTRY: dict[str, WatermarkVerifier] = {"unsupported": UnsupportedVerifier()}


def register_verifier(verifier: WatermarkVerifier) -> None:
    if not getattr(verifier, "provider", None):
        raise ValueError("verifier must define a provider name")
    _REGISTRY[str(verifier.provider)] = verifier


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def get_verifier(provider: str | None = None) -> WatermarkVerifier:
    provider = provider or "unsupported"
    if provider not in _REGISTRY:
        return UnsupportedVerifier()
    return _REGISTRY[provider]


def verify_text(
    text: str,
    *,
    provider: str | None = None,
    key: str | None = None,
) -> WatermarkResult:
    if not text or not text.strip():
        return WatermarkResult(
            status=DetectionStatus.INCONCLUSIVE,
            provider=provider or "unsupported",
            details="Empty text cannot be verified.",
            assumptions=("Empty input carries no watermark signal.",),
        )
    return get_verifier(provider).verify(text, key=key)
