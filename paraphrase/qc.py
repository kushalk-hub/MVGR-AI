import argparse
import pandas as pd


def jaccard_overlap(text_a, text_b):
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pilot/pilot_dataset.csv")
    parser.add_argument("--output", default="data/qc_overlap.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
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
                "len_words": len(row["text"].split()),
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


if __name__ == "__main__":
    main()