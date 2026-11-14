"""Module D tests — statistical watermark verification.

Synthetic text only. No key is ever hardcoded outside the test; the detector is
tested with a test key and with no key, and the "unsupported is not negative"
rule is asserted explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from watermark.statistics import binomial_p_value, normal_survival, z_score  # noqa: E402
from watermark.topic import TopicWatermarkVerifier  # noqa: E402
from watermark.unigram import UnigramWatermarkVerifier, is_green  # noqa: E402
from watermark.verifier import (  # noqa: E402
    DetectionStatus,
    UnsupportedVerifier,
    available_providers,
    get_verifier,
    verify_text,
)

TEST_KEY = "unit-test-key-not-a-secret"


def _make_text(key: str, *, green: bool, n: int = 80, seed: int = 0) -> str:
    """Build a token sequence whose tokens are (mostly) green or red for ``key``."""
    words = ["seed"]
    for _ in range(n - 1):
        context = words[-1]
        for candidate_index in range(200000):
            candidate = f"w{candidate_index}"
            flag = is_green(key, context, candidate, gamma=0.25, seed=seed)
            if flag == green:
                words.append(candidate)
                break
        else:  # pragma: no cover - only if the hash is pathological
            raise AssertionError("could not construct a token with the requested greenness")
    return " ".join(words)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_z_score_and_survival():
    assert z_score(observed=10, expected=5, variance=4) == pytest.approx(2.5)
    assert z_score(observed=10, expected=5, variance=0) is None
    assert normal_survival(0.0) == pytest.approx(0.5)
    assert normal_survival(10.0) < 1e-20


def test_binomial_p_value_edges():
    assert binomial_p_value(0, 10, 0.5) == pytest.approx(1.0)
    assert binomial_p_value(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    assert binomial_p_value(5, 0, 0.5) is None


# --------------------------------------------------------------------------- #
# green-list mechanics
# --------------------------------------------------------------------------- #
def test_is_green_is_deterministic_and_key_dependent():
    first = is_green(TEST_KEY, "ctx", "token", gamma=0.25, seed=0)
    second = is_green(TEST_KEY, "ctx", "token", gamma=0.25, seed=0)
    assert first == second
    # Different keys must (in practice) produce different green lists.
    different = [
        is_green(f"key-{index}", "ctx", "token", gamma=0.25, seed=0)
        for index in range(20)
    ]
    assert len(set(different)) == 2  # both outcomes occur across keys


# --------------------------------------------------------------------------- #
# unigram detector
# --------------------------------------------------------------------------- #
def test_no_key_is_unsupported_not_negative():
    verifier = UnigramWatermarkVerifier()
    result = verifier.verify(_make_text(TEST_KEY, green=True), key=None)
    assert result.status is DetectionStatus.UNSUPPORTED
    assert result.status is not DetectionStatus.NOT_DETECTED
    assert result.supports_human_authorship is False


def test_green_text_is_detected():
    verifier = UnigramWatermarkVerifier(z_threshold=4.0)
    text = _make_text(TEST_KEY, green=True)
    result = verifier.verify(text, key=TEST_KEY)
    assert result.status is DetectionStatus.DETECTED
    assert result.evidence["green_list_proportion"] > 0.9
    assert result.evidence["z_score"] > 4.0
    assert result.confidence is not None and result.confidence > 0.9


def test_red_text_is_not_detected():
    verifier = UnigramWatermarkVerifier(z_threshold=4.0)
    text = _make_text(TEST_KEY, green=False)
    result = verifier.verify(text, key=TEST_KEY)
    assert result.status is DetectionStatus.NOT_DETECTED
    assert result.supports_human_authorship is False
    # A negative result only concerns the tested scheme.
    assert "not evidence of human authorship" in result.details.lower()


def test_short_text_is_inconclusive():
    verifier = UnigramWatermarkVerifier()
    result = verifier.verify("too short to test", key=TEST_KEY)
    assert result.status is DetectionStatus.INCONCLUSIVE


# --------------------------------------------------------------------------- #
# unsupported schemes
# --------------------------------------------------------------------------- #
def test_topic_conditioned_is_unsupported():
    verifier = TopicWatermarkVerifier()
    result = verifier.verify("some text", key=TEST_KEY)
    assert result.status is DetectionStatus.UNSUPPORTED
    assert result.supports_human_authorship is False


def test_unknown_provider_falls_back_to_unsupported():
    assert available_providers() == ["topic_conditioned", "unigram_greenlist", "unsupported"]
    result = get_verifier("some-other-watermark").verify("text", key=TEST_KEY)
    assert result.status is DetectionStatus.UNSUPPORTED
    assert isinstance(get_verifier("some-other-watermark"), UnsupportedVerifier)


def test_verify_text_empty_is_inconclusive():
    result = verify_text("   ", provider="unigram_greenlist", key=TEST_KEY)
    assert result.status is DetectionStatus.INCONCLUSIVE


def test_verify_text_never_leaks_key_in_result():
    result = verify_text(_make_text(TEST_KEY, green=True), provider="unigram_greenlist", key=TEST_KEY)
    serialized = str(result.as_dict())
    assert TEST_KEY not in serialized
