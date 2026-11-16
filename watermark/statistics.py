"""Statistical helpers for watermark verification.

A watermark detector is a hypothesis test: under the null hypothesis (no
watermark) the observed green-list count follows a known distribution, and a
large positive deviation is evidence of a watermark. A large deviation is
evidence about the *tested scheme only*; it is never evidence about authorship.
"""

from __future__ import annotations

import math


def z_score(observed: float, expected: float, variance: float) -> float | None:
    if variance is None or variance <= 0:
        return None
    return (float(observed) - float(expected)) / math.sqrt(float(variance))


def normal_survival(z: float) -> float:
    """One-sided upper-tail probability ``P(Z >= z)`` for a standard normal."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def binomial_p_value(successes: int, trials: int, probability: float) -> float | None:
    """Exact one-sided upper-tail ``P(X >= successes)`` for ``X ~ Binomial``.

    The sum is evaluated in log space. The naive product
    ``C(n, k) * p**k * (1-p)**(n-k)`` overflows a float as soon as ``n`` is a
    few thousand, which happens for long documents; the log-space form is
    exact and stable for every trial count.
    """
    if trials <= 0:
        return None
    if successes <= 0:
        return 1.0
    probability = min(max(float(probability), 0.0), 1.0)
    if probability <= 0.0:
        return 0.0 if successes > 0 else 1.0
    if probability >= 1.0:
        return 1.0
    successes = min(int(successes), int(trials))

    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_n_factorial = math.lgamma(trials + 1)

    log_terms: list[float] = []
    for k in range(successes, trials + 1):
        log_terms.append(
            log_n_factorial
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            + k * log_p
            + (trials - k) * log_q
        )

    # Log-sum-exp: the exact tail probability without intermediate overflow.
    largest = max(log_terms)
    total = math.exp(largest) * sum(math.exp(value - largest) for value in log_terms)
    return min(1.0, max(0.0, total))


__all__ = ["binomial_p_value", "normal_survival", "z_score"]
