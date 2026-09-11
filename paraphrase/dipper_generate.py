import argparse
import csv
import json
import nltk
import os
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

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


def paraphrase(model, tokenizer, text, lex, order, max_length, top_p, sent_interval):
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    prefix = ""
    outputs = []
    for i in range(0, len(sentences), sent_interval):
        window = " ".join(sentences[i:i + sent_interval])
        prompt = f"lexical = {lex}, order = {order} {prefix} <sent> {window} </sent>"
        prompt = clean_input(prompt)
        inputs = tokenizer([prompt], return_tensors="pt")
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
    args = parser.parse_args()

    import pandas as pd
    sources = pd.read_csv(args.input)
    ai_sources = sources[sources["label"] == "ai"]
    human_sources = sources[sources["label"] == "human"]
    if args.limit > 0:
        ai_sources = ai_sources.head(args.limit)

    levels = load_levels(args.levels)
    done = load_done_keys(args.output)

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

    print(f"Paraphrase generation complete -> {args.output}")


if __name__ == "__main__":
    main()