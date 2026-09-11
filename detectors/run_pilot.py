import argparse
import gc
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_roberta(df, batch_size=32):
    from detectors.roberta import RoBERTaDetector
    detector = RoBERTaDetector()
    scores = []
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        scores.extend(detector.score_batch(chunk))
        print(f"roberta: {min(i + batch_size, len(texts))}/{len(texts)}")
    del detector
    free_gpu()
    return scores


def score_binoculars(df, batch_size=8):
    from detectors.binoculars import BinocularsDetector
    detector = BinocularsDetector()
    raw_scores = []
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        raw_scores.extend(detector.compute_score(chunk))
        print(f"binoculars: {min(i + batch_size, len(texts))}/{len(texts)}")
    del detector
    free_gpu()
    return raw_scores


def score_gptzero(df, limit):
    from detectors.gptzero import make_gptzero
    detector, mode = make_gptzero()
    print(f"GPTZero mode: {mode} ({type(detector).__name__})")
    texts = df["text"].tolist()
    if mode == "api":
        scores = detector.score_batch(texts, limit=limit)
        print(f"gptzero(api): {len(scores)}/{len(texts)} scored")
    else:
        scores = detector.score_batch(texts)
        print(f"gptzero(local): {len(scores)}/{len(texts)} scored")
    padded = scores + [float("nan")] * (len(texts) - len(scores))
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pilot/pilot_dataset.csv")
    parser.add_argument("--output", default="results/pilot_scores.csv")
    parser.add_argument("--detectors", nargs="+", default=["roberta", "binoculars", "gptzero"])
    parser.add_argument("--gptzero-limit", type=int, default=80)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = df[["sample_id", "source_id", "text", "label", "paraphrase_level"]].copy()

    if "roberta" in args.detectors:
        result["roberta_score"] = score_roberta(df)
        result.to_csv(args.output, index=False)

    if "binoculars" in args.detectors:
        result["binoculars_raw"] = score_binoculars(df)
        result.to_csv(args.output, index=False)

    if "gptzero" in args.detectors:
        scores = score_gptzero(df, args.gptzero_limit)
        result["gptzero_score"] = scores
        result.to_csv(args.output, index=False)

    result.to_csv(args.output, index=False)
    print(f"Scores saved -> {args.output}")
    print(result[["sample_id", "paraphrase_level", "label"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()