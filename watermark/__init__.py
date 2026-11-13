"""Watermark verification package (Person 1)."""

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

__all__ = [
    "DetectionStatus",
    "UnsupportedVerifier",
    "WatermarkResult",
    "WatermarkVerifier",
    "available_providers",
    "get_verifier",
    "register_verifier",
    "verify_text",
]
