import argparse
import sys

import pandas as pd


def jaccard_overlap(text_a, text_b):
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


LEVEL_TOLERANCE = 1e-9


def monotonicity_violations(rows):
    """Return (prev_level, level) pairs whose overlap exceeds the previous level.

    Overlap against L0 should be non-increasing as paraphrase intensity
    rises. This is the check that catches an inverted control-code ladder,
    where overlap climbs with the level number instead of falling.
    """
    violations = []
    for prev, current in zip(rows, rows[1:]):
        prev_overlap = prev.get("mean_jaccard_overlap_L0")
        current_overlap = current.get("mean_jaccard_overlap_L0")
        if prev_overlap is None or current_overlap is None:
            continue
        if current_overlap > prev_overlap + LEVEL_TOLERANCE:
            violations.append((prev.get("level"), current.get("level")))
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pilot/pilot_dataset.csv")
    parser.add_argument("--output", default="data/qc_overlap.csv")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if mean Jaccard overlap against L0 rises from one level to the next.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["text"] = df["text"].fillna("").astype(str)
    ai = df[df["label"] == "ai"]

    l0 = ai[ai["paraphrase_level"] == "L0"]
    l0_by_source = dict(zip(l0["source_id"], l0["text"]))

    rows = []
    for level in ["L0", "L1", "L2", "L3", "L4"]:
        subset = ai[ai["paraphrase_level"] == level]
        records = []
        for _, row in subset.iterrows():
            src_text = l0_by_source.get(row["source_id"], row["text"])
            records.append({
                "source_id": row["source_id"],
                "level": level,
                "overlap": jaccard_overlap(src_text, row["text"]),
                "len_words": len(str(row["text"]).split()),
            })
        if records:
            rec_df = pd.DataFrame(records)
            rows.append({
                "level": level,
                "n": len(rec_df),
                "mean_jaccard_overlap_L0": rec_df["overlap"].mean(),
                "mean_words": rec_df["len_words"].mean(),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"QC saved -> {args.output}")

    if args.strict:
        violations = monotonicity_violations(rows)
        if violations:
            pairs = ", ".join(f"{prev}->{cur}" for prev, cur in violations)
            print(
                f"\nSTRICT FAIL: mean Jaccard overlap against L0 is not "
                f"non-increasing at: {pairs}"
            )
            print(
                "  Hint: paraphrase intensity should reduce overlap as levels "
                "rise. Rising overlap points at an inverted control-code ladder —"
            )
            print(
                "  check that configs/dipper_levels.json holds diversity values "
                "and control_codes() is applied (see paraphrase/levels.py)."
            )
            sys.exit(1)
        print("\nSTRICT PASS: overlap is non-increasing L0->L4.")


if __name__ == "__main__":
    main()