import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def split_sources(source_ids, val_fraction=0.2, seed=42):
    rng = np.random.RandomState(seed)
    source_ids = np.array(list(source_ids))
    rng.shuffle(source_ids)
    n_val = max(1, int(len(source_ids) * val_fraction))
    val_ids = set(source_ids[:n_val].tolist())
    test_ids = set(source_ids[n_val:].tolist())
    return val_ids, test_ids


def youden_threshold(y, scores, direction=-1):
    order = np.argsort(scores)
    scores_sorted = scores[order]
    y_sorted = y[order]
    best_thr, best_j = None, -np.inf
    for thr in np.unique(scores_sorted):
        pred = (scores_sorted < thr).astype(int)
        tp = ((pred == 1) & (y_sorted == 1)).sum()
        tn = ((pred == 0) & (y_sorted == 0)).sum()
        tpr = tp / max((y_sorted == 1).sum(), 1)
        tnr = tn / max((y_sorted == 0).sum(), 1)
        j = tpr + tnr - 1
        if j > best_j:
            best_j, best_thr = j, thr
    return best_thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/pilot_scores.csv")
    parser.add_argument("--output-eval", default="results/pilot_scores_eval.csv")
    parser.add_argument("--output-threshold", default="results/binoculars_threshold.json")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.input) or "binoculars_raw" not in pd.read_csv(args.input, nrows=3).columns:
        print(f"binoculars_raw not found in {args.input}. Run detectors/run_pilot.py --detectors binoculars first.")
        return

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["binoculars_raw"])

    ai_sources = set(df.loc[df["label"] == "ai", "source_id"])
    human_sources = set(df.loc[df["label"] == "human", "source_id"])

    val_ai, test_ai = split_sources(ai_sources, args.val_fraction, args.seed)
    val_hum, test_hum = split_sources(human_sources, args.val_fraction, args.seed + 1)

    val_set = val_ai | val_hum
    test_set = test_ai | test_hum

    val = df[df["source_id"].isin(val_set)]
    test = df[df["source_id"].isin(test_set)]

    y_val = (val["label"] == "ai").astype(int).values
    scores_val = val["binoculars_raw"].values
    threshold = youden_threshold(y_val, scores_val, direction=-1)

    # direction: Binoculars raw score is LOWER for AI text
    if roc_auc_score(y_val, -scores_val) < 0.5:
        print("WARNING: inverted sign detected; flipping score direction.")
        scores_val = -scores_val
        threshold = youden_threshold(y_val, scores_val, 1)

    result = {
        "threshold_binoculars_raw": float(threshold),
        "calibration_n": int(len(val)),
        "test_n": int(len(test)),
        "direction": "lower_is_ai",
    }
    with open(args.output_threshold, "w") as f:
        json.dump(result, f, indent=2)

    test.to_csv(args.output_eval, index=False)

    print(f"Calibration samples: {len(val)}  |  Eval samples: {len(test)}")
    print(f"Calibrated Binoculars threshold (raw, lower=AI): {threshold:.4f}")
    print(f"Saved -> {args.output_eval}")
    print(f"Saved -> {args.output_threshold}")


if __name__ == "__main__":
    main()