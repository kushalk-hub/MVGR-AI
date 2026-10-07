import pytest

from paraphrase.qc import monotonicity_violations

# Tolerance guards against float noise; equal values pass.
TOL = 1e-9


def rows(*pairs):
    return [
        {"level": level, "mean_jaccard_overlap_L0": overlap}
        for level, overlap in pairs
    ]


def test_decreasing_overlap_has_no_violations():
    assert (
        monotonicity_violations(
            rows(("L0", 1.0), ("L1", 0.80), ("L2", 0.60), ("L3", 0.40), ("L4", 0.20))
        )
        == []
    )


def test_equal_overlap_passes():
    assert (
        monotonicity_violations(rows(("L0", 1.0), ("L1", 0.70), ("L2", 0.70))) == []
    )


def test_float_noise_passes():
    assert (
        monotonicity_violations(
            rows(("L0", 0.5000000001), ("L1", 0.5), ("L2", 0.4999999999))
        )
        == []
    )


def test_rising_overlap_is_flagged():
    violations = monotonicity_violations(
        rows(("L0", 1.0), ("L1", 0.80), ("L2", 0.90), ("L3", 0.40), ("L4", 0.20))
    )
    assert violations == [("L1", "L2")]


def test_inverted_ladder_is_flagged():
    # This is the pre-fix signature: overlap RISES as levels increase.
    # (L0 self-overlap is 1.0, so L0->L1 is a drop, not a violation;
    # the rises L1->L2->L3->L4 are the violations.)
    violations = monotonicity_violations(
        rows(("L0", 1.0), ("L1", 0.40), ("L2", 0.55), ("L3", 0.70), ("L4", 0.85))
    )
    assert violations == [("L1", "L2"), ("L2", "L3"), ("L3", "L4")]


def test_single_row_is_trivially_valid():
    assert monotonicity_violations(rows(("L0", 0.5))) == []


def test_gap_in_level_coverage_is_still_compared():
    # Absent levels are evaluate.py's hard guard, not qc.py's job. The two rows
    # present are still compared, and a genuine drop is not a violation.
    assert monotonicity_violations(rows(("L0", 1.0), ("L4", 0.30))) == []
