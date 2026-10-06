import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

import nltk
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paraphrase.levels import control_codes, unit_seed

DEFAULT_MODEL = "kalpeshk2011/dipper-paraphraser-xxl"
DEFAULT_TOKENIZER = "google/t5-v1_1-xxl"
COLUMNS = [
    "sample_id",
    "source_id",
    "text",
    "label",
    "paraphrase_level",
    "lex_control",
    "order_control",
    "parent_sample_id",
    "quality_status",
]


def load_model(model_id):
    import transformers

    try:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        return T5ForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quant_config, device_map="auto"
        )
    except (ImportError, TypeError, ValueError, NotImplementedError):
        pass

    try:
        return T5ForConditionalGeneration.from_pretrained(
            model_id, load_in_8bit=True, device_map="auto"
        )
    except Exception as exc:
        try:
            import bitsandbytes
            bnb_ver = bitsandbytes.__version__
        except Exception:
            bnb_ver = "not installed"
        raise SystemExit(
            "\nDIPPER requires 8-bit loading (T5-XXL ~11GB).\n"
            f"  transformers = {transformers.__version__}\n"
            f"  bitsandbytes = {bnb_ver}\n"
            f"  cuda available = {torch.cuda.is_available()}\n"
            "Fix: run the notebook's pip-install cell (it must print success), then re-run.\n"
            f"Last error: {exc!r}"
        ) from exc


def ensure_punkt():
    for name in ("punkt", "punkt_tab"):
        try:
            nltk.download(name, quiet=True)
        except Exception:
            pass


def load_levels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["levels"]


def clean_input(text):
    return " ".join(text.split())


def paraphrase(model, tokenizer, text, lex_diversity, order_diversity, max_length, top_p, sent_interval):
    """Paraphrase one text at the given DIVERSITY levels.

    lex_diversity and order_diversity are desired diversity; they are
    converted to the similarity control codes the DIPPER model reads.
    """
    lex_code, order_code = control_codes(lex_diversity, order_diversity)
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    prefix = ""
    outputs = []
    for i in range(0, len(sentences), sent_interval):
        window = " ".join(sentences[i:i + sent_interval])
        prompt = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {window} </sent>"
        prompt = clean_input(prompt)
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=True, top_p=top_p, top_k=None, max_length=max_length)
        out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        outputs.append(out)
        prefix = (prefix + " " + out).strip()
    return " ".join(outputs).strip()


def write_rows(path, rows):
    new_file = not os.path.exists(path)
    mode = "a" if not new_file else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_done_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("label") == "ai":
                    done.add((row["source_id"], row["paraphrase_level"]))
    return done


class Tally:
    """Counts per-unit generation outcomes so a run's health is visible."""

    def __init__(self):
        self._counts = {"ok": 0, "empty": 0, "error": 0}

    def record(self, status):
        if status not in self._counts:
            raise KeyError(f"unknown status {status!r}")
        self._counts[status] += 1

    def as_dict(self):
        return dict(self._counts)


def is_generated_row(row):
    """True for a paraphrased unit: an ai row above L0.

    Human rows are excluded by label and the seeded L0 rows by level. Both
    carry quality_status="ok", but neither is a generated unit.
    """
    return row["label"] == "ai" and row["paraphrase_level"] != "L0"


def tally_dataset(path):
    """Count outcomes across every generated row in the output CSV.

    Dataset-scoped rather than invocation-scoped: a resumed run skips the
    units already on disk, so counting only what this invocation produced
    would report zero for a dataset that is nearly finished.
    """
    if not os.path.exists(path):
        return Tally().as_dict()

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tally = Tally()
    for row in rows:
        if is_generated_row(row):
            tally.record(row["quality_status"])
    return tally.as_dict()


def build_manifest(run_seed, model, tokenizer, levels, top_p, max_length, sent_interval, tally):
    """Assemble the generation provenance record.

    `tally` is dataset-scoped: it counts every generated paraphrase row in the
    output CSV, not just the units this invocation produced, so the record
    survives a resumed run.

    Records each level as both the requested diversity and the similarity
    control code the model actually read, so a dataset can be traced back to
    the exact attack configuration that produced it.
    """
    import transformers

    def _version(module_name):
        try:
            module = __import__(module_name)
            return getattr(module, "__version__", "unknown")
        except Exception as exc:
            return f"unavailable ({type(exc).__name__})"

    return {
        "seed": run_seed,
        "model": model,
        "tokenizer": tokenizer,
        "load_in_8bit": True,
        "levels": [
            {
                "level": cfg["level"],
                "lex_diversity": cfg["lex"],
                "order_diversity": cfg["order"],
                "lex_code": control_codes(cfg["lex"], cfg["order"])[0],
                "order_code": control_codes(cfg["lex"], cfg["order"])[1],
            }
            for cfg in levels
        ],
        "top_p": top_p,
        "max_length": max_length,
        "sent_interval": sent_interval,
        "tally": dict(tally),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": _version("bitsandbytes"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    ensure_punkt()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/pilot_source.csv")
    parser.add_argument("--output", default="data/pilot/pilot_dataset.csv")
    parser.add_argument("--levels", default="configs/dipper_levels.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.75)
    parser.add_argument("--sent-interval", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    sources = pd.read_csv(args.input)
    ai_sources = sources[sources["label"] == "ai"]
    human_sources = sources[sources["label"] == "human"]
    if args.limit > 0:
        ai_sources = ai_sources.head(args.limit)

    levels = load_levels(args.levels)
    done = load_done_keys(args.output)
    tally = Tally()

    if not os.path.exists(args.output):
        human_rows = [
            {
                "sample_id": f"{sid}_human",
                "source_id": sid,
                "text": row["text"],
                "label": "human",
                "paraphrase_level": "human",
                "lex_control": "",
                "order_control": "",
                "parent_sample_id": "",
                "quality_status": "ok",
            }
            for sid, row in human_sources.iterrows()
        ]
        l0_rows = [
            {
                "sample_id": f"{sid}_L0",
                "source_id": sid,
                "text": row["text"],
                "label": "ai",
                "paraphrase_level": "L0",
                "lex_control": "",
                "order_control": "",
                "parent_sample_id": "",
                "quality_status": "ok",
            }
            for sid, row in ai_sources.iterrows()
        ]
        write_rows(args.output, human_rows + l0_rows)

    tokenizer = T5Tokenizer.from_pretrained(DEFAULT_TOKENIZER)
    model = load_model(args.model)
    model.eval()

    pending = []
    for _, row in ai_sources.iterrows():
        sid = row["source_id"]
        for cfg in levels:
            pending.append((sid, row["text"], cfg))

    for sid, text, cfg in tqdm(pending):
        if (str(sid), cfg["level"]) in done:
            continue
        # Seed per unit, not once per run: each (source_id, level) then
        # reproduces independently of traversal or resume order.
        torch.manual_seed(unit_seed(args.seed, sid, cfg["level"]))
        try:
            out = paraphrase(
                model, tokenizer, clean_input(text), cfg["lex"], cfg["order"],
                args.max_length, args.top_p, args.sent_interval,
            )
            status = "ok" if out else "empty"
        except Exception as exc:
            print(f"failed source={sid} level={cfg['level']}: {exc}")
            out = ""
            status = "error"
        tally.record(status)
        rows = [{
            "sample_id": f"{sid}_{cfg['level']}",
            "source_id": sid,
            "text": out,
            "label": "ai",
            "paraphrase_level": cfg["level"],
            "lex_control": cfg["lex"],
            "order_control": cfg["order"],
            "parent_sample_id": f"{sid}_L0",
            "quality_status": status,
        }]
        write_rows(args.output, rows)

    manifest_path = os.path.join(
        os.path.dirname(args.output) or ".", "generation_manifest.json"
    )
    written = False
    try:
        manifest = build_manifest(
            run_seed=args.seed,
            model=args.model,
            tokenizer=DEFAULT_TOKENIZER,
            levels=levels,
            top_p=args.top_p,
            max_length=args.max_length,
            sent_interval=args.sent_interval,
            tally=tally_dataset(args.output),
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        written = True
    except Exception as exc:
        print(f"warning: manifest not written -> {manifest_path}: {exc}")

    counts = tally.as_dict()
    total = sum(counts.values()) or 1
    print(f"Paraphrase generation complete -> {args.output}")
    print(
        f"  ok={counts['ok']} empty={counts['empty']} error={counts['error']} "
        f"({counts['error'] * 100.0 / total:.1f}% failed)"
    )
    if written:
        print(f"  manifest -> {manifest_path}")


if __name__ == "__main__":
    main()