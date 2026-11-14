"""Topic-conditioned watermark verification interface.

Some watermarking schemes bias tokens using a *topic* (or semantic) signal in
addition to a key. Such a scheme can only be verified when the exact generation
and verification protocol is known: which topic model, how the topic is derived
from the key, and how the green list is constructed.

No such protocol is configured here, so this verifier returns ``unsupported``.
It exists so the framework has an explicit, honest slot for topic-conditioned
schemes rather than pretending every watermark is a unigram scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from watermark.verifier import DetectionStatus, WatermarkResult


@dataclass
class TopicWatermarkVerifier:
    provider: str = "topic_conditioned"
    protocol_known: bool = False
    protocol: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "TopicWatermarkVerifier":
        config = config or {}
        return cls(
            protocol_known=bool(config.get("protocol_known", False)),
            protocol=dict(config.get("protocol", {})),
        )

    def verify(self, text: str, *, key: str | None = None) -> WatermarkResult:
        if not self.protocol_known:
            return WatermarkResult(
                status=DetectionStatus.UNSUPPORTED,
                provider=self.provider,
                confidence=None,
                details=(
                    "Topic-conditioned watermark verification requires the exact "
                    "generation and verification protocol (topic model, key derivation, "
                    "green-list construction). None is configured."
                ),
                assumptions=(
                    "An unknown scheme cannot be tested; this is not a negative result.",
                    "Absence of detection is not evidence of human authorship.",
                ),
            )
        return WatermarkResult(
            status=DetectionStatus.INCONCLUSIVE,
            provider=self.provider,
            details="A protocol was supplied but no topic-conditioned detector is implemented.",
        )


__all__ = ["TopicWatermarkVerifier"]
