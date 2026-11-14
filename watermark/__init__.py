"""Watermark verification package (Person 1).

Includes the default ``unsupported`` verifier, a research unigram green-list
detector, and a topic-conditioned interface that stays unsupported until a
protocol is known. Verification is never an authorship judgement.
"""

from watermark.statistics import binomial_p_value, normal_survival, z_score
from watermark.topic import TopicWatermarkVerifier
from watermark.unigram import UnigramWatermarkVerifier, is_green
from watermark.verifier import (
    DetectionStatus,
    UnsupportedVerifier,
    WatermarkResult,
    WatermarkVerifier,
    available_providers,
    get_verifier,
    register_verifier,
    verify_text,
)

register_verifier(UnigramWatermarkVerifier())
register_verifier(TopicWatermarkVerifier())

__all__ = [
    "DetectionStatus",
    "TopicWatermarkVerifier",
    "UnigramWatermarkVerifier",
    "UnsupportedVerifier",
    "WatermarkResult",
    "WatermarkVerifier",
    "available_providers",
    "binomial_p_value",
    "get_verifier",
    "is_green",
    "normal_survival",
    "register_verifier",
    "verify_text",
    "z_score",
]
