import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


LEVELS_ORDER = ["L0", "L1", "L2", "L3", "L4"]
DET_NAMES = ["roberta_score", "binoculars_raw", "gptzero_score"]
DET_LABELS = ["RoBERTa", "Binoculars (small pair)", "GPTZero*"]
DET_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# Binoculars raw score is LOWER for AI text -> flip to keep "higher=AI" for AUROC
DET_DIRECTION = {"roberta_score": 1.0, "gptzero_score": 1.0, "binoculars_raw": -1.0}


def tpr_at_fpr(y_true, y_scores, fpr_limit=0.01):
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
    idx = np.where(fpr_arr <= fpr_limit)[0]
    return float(tpr_arr[idx[-1]]) if len(idx) else 0.0


def threshold_metrics(y_true, y_scores, threshold, direction):
    if direction < 0:
        pred = (y_scores < threshold).astype(int)
    else:
        pred = (y_scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/pilot_scores_eval.csv")
    parser.add_argument("--output-plot", default="results/degradation_plot.png")
    parser.add_argument("--output-table", default="results/results_table.csv")
    parser.add_argument("--output-summary", default="results/summary.md")
    parser.add_argument("--threshold-file", default="results/binoculars_threshold.json")
    parser.add_argument("--fpr-limit", type=float, default=0.01)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input {args.input} not found. Run detectors/calibrate_binoculars.py first.")
        return

    df = pd.read_csv(args.input)

    bin_threshold = None
    if os.path.exists(args.threshold_file):
        with open(args.threshold_file) as f:
            bin_threshold = json.load(f).get("threshold_binoculars_raw")

    rows = list()
    for level in LEVELS_ORDER:
        ai_level = df[(df["paraphrase_level"] == level) & (df["label"] == "ai")]
        human = df[df["label"] == "human"]
        if len(ai_level) == 0:
            continue
        level_df = pd.concat([ai_level, human])
        y = (level_df["label"] == "ai").astype(int).values
        row = {"level": level, "ai_n": len(ai_level), "human_n": len(human)}
        for det_name in DET_NAMES:
            if det_name not in level_df.columns:
                continue
            raw = level_df[det_name].values
            direction = DET_DIRECTION[det_name]
            scores = raw * direction
            ok = ~np.isnan(scores)
            auroc = roc_auc_score(y[ok], scores[ok]) if len(np.unique(y[ok])) == 2 else 0.0
            tpr1pct = tpr_at_fpr(y[ok], scores[ok], args.fpr_limit)

            thr = 0.5
            if det_name == "binoculars_raw" and bin_threshold is not None:
                thr = bin_threshold
            prec, rec, f1 = threshold_metrics(y[ok], raw[ok], thr, direction)

            row[f"{det_name}_auroc"] = round(auroc, 4)
            row[f"{det_name}_tpr1pct"] = round(tpr1pct, 4)
            row[f"{det_name}_f1"] = f1
            row[f"{det_name}_precision"] = prec
            row[f"{det_name}_recall"] = rec
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No evaluation rows produced.")
        return
    out_df.to_csv(args.output_table, index=False)
    print(f"Results table -> {args.output_table}\n")
    print(out_df.to_string(index=False))

    print("\n--- AUROC degradation (L0 minus Li) ---")
    degradation = dict()
    for det_name in DET_NAMES:
        key = f"{det_name}_auroc"
        if key not in out_df.columns:
            continue
        base = out_df.loc[out_df["level"] == "L0", key].values[0]
        vals = {lv: out_df.loc[out_df["level"] == lv, key].values[0] for lv in LEVELS_ORDER}
        degradation[det_name] = {lv: round(base - v, 4) for lv, v in vals.items()}
        print(f"{det_name}: " + "  ".join(f"{lv}={base - v:.4f}" for lv, v in vals.items()))

    fig, ax = plt.subplots(figsize=(9, 6))
    for det_name, color, label in zip(DET_NAMES, DET_COLORS, DET_LABELS):
        key = f"{det_name}_auroc"
        if key not in out_df.columns:
            continue
        vals = [out_df.loc[out_df["level"] == lv, key].values[0] for lv in LEVELS_ORDER]
        ax.plot(range(len(LEVELS_ORDER)), vals, marker="o", label=label, color=color, linewidth=2)
    ax.set_xticks(range(len(LEVELS_ORDER)))
    ax.set_xticklabels(LEVELS_ORDER)
    ax.set_xlabel("Paraphrase level (L0 = original AI text)")
    ax.set_ylabel("AUROC")
    ax.set_title("Detector AUROC under increasing DIPPER paraphrase intensity")
    ax.set_ylim(0.40, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    plt.close(fig)
    print(f"\nPlot -> {args.output_plot}")

    summary_cols = ["level"]
    for det_name in DET_NAMES:
        for metric in ["auroc", "tpr1pct", "f1", "precision", "recall"]:
            key = f"{det_name}_{metric}"
            if key in out_df.columns:
                summary_cols.append(key)
    table_str = out_df[summary_cols].to_csv(index=False)

    lines = [
        "# Stage 1 Evaluation Summary\n",
        f"- Eval samples: {len(df)}",
        f"- AI samples at L0: {int(out_df.loc[out_df['level'] == 'L0', 'ai_n'].values[0]) if len(out_df[out_df['level'] == 'L0']) else 'n/a'}",
        f"- Human samples: {int(df[df['label'] == 'human'].shape[0])}",
        f"- Binoculars calibrated threshold (raw, lower=AI): {bin_threshold}",
        f"- TPR@1%FPR reference: {args.fpr_limit}\n",
        "## AUROC per level\n",
        "```csv",
        table_str,
        "```\n",
        "## AUROC degradation (L0 minus Li)\n",
    ]
    for det_name, deg in degradation.items():
        lines.append(f"- {det_name}: " + "  ".join(f"{lv}={d:.4f}" for lv, d in deg.items()))
    lines.append("\n* GPTZero is a local perplexity-style proxy (gpt2-medium) when no paid API key is configured.")
    lines.append("_Generated automatically by evaluation/evaluate.py_")

    with open(args.output_summary, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary -> {args.output_summary}")


if __name__ == "__main__":
    main()