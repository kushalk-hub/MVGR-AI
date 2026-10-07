import csv
import subprocess
import sys
from pathlib import Path

import pytest

from paraphrase.qc import monotonicity_violations

REPO_ROOT = Path(__file__).resolve().parents[1]

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


# --- CLI wiring regression tests -------------------------------------------
# The function tests above would pass even if the --strict branch always
# exited 0 — the original bug. These tests pin the exit codes and output text.

L0_TEXT = "alpha beta gamma delta"
LOW_TEXT = "foo bar baz qux quux"  # zero overlap with L0
MID_TEXT = "alpha beta foo bar"  # partial overlap with L0 (~0.333)


def write_csv(path, texts_by_level):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "source_id", "text", "label", "paraphrase_level"]
        )
        writer.writeheader()
        for level, text in texts_by_level:
            writer.writerow({
                "sample_id": f"1_{level}",
                "source_id": 1,
                "text": text,
                "label": "ai",
                "paraphrase_level": level,
            })


def run_qc(input_csv, output_csv, strict):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "paraphrase" / "qc.py"),
        "--input", str(input_csv),
        "--output", str(output_csv),
    ]
    if strict:
        cmd.append("--strict")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_strict_fails_on_rising_overlap(tmp_path):
    input_csv = tmp_path / "qc_inverted.csv"
    write_csv(input_csv, [("L0", L0_TEXT), ("L1", LOW_TEXT), ("L2", MID_TEXT)])
    proc = run_qc(input_csv, tmp_path / "qc_out.csv", strict=True)
    assert proc.returncode != 0
    assert "STRICT FAIL" in proc.stdout
    assert "L1->L2" in proc.stdout


def test_non_strict_stays_backward_compatible(tmp_path):
    input_csv = tmp_path / "qc_inverted.csv"
    write_csv(input_csv, [("L0", L0_TEXT), ("L1", LOW_TEXT), ("L2", MID_TEXT)])
    proc = run_qc(input_csv, tmp_path / "qc_out.csv", strict=False)
    assert proc.returncode == 0
    assert "STRICT FAIL" not in proc.stdout


def test_strict_passes_on_monotonic_overlap(tmp_path):
    input_csv = tmp_path / "qc_monotonic.csv"
    write_csv(input_csv, [("L0", L0_TEXT), ("L1", MID_TEXT), ("L2", LOW_TEXT)])
    proc = run_qc(input_csv, tmp_path / "qc_out.csv", strict=True)
    assert proc.returncode == 0
    assert "STRICT PASS" in proc.stdout
