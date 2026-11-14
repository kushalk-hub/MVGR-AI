"""One-off, explicit download of the GLTR reference model (GPT-2).

This is the **only** code path that touches the network for GLTR. Experiments
never download: they load the model from ``models/gltr/gpt2`` with
``allow_download=false``. Run this once, with network access, then run offline.

Usage
-----
    python scripts/download_gltr_model.py
    python scripts/download_gltr_model.py --model gpt2 --revision <sha> --out models/gltr/gpt2
    python scripts/download_gltr_model.py --force        # re-download over an existing copy

After a successful download the model directory contains ``config.json``,
the tokenizer files, the weights and a ``MODEL_INFO.json`` provenance record.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.paths import resolve  # noqa: E402

DEFAULT_MODEL = "gpt2"
# gpt2 main commit pinned at download time so the reference model cannot drift.
DEFAULT_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEFAULT_OUT = "models/gltr/gpt2"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="pinned git revision")
    parser.add_argument("--out", default=DEFAULT_OUT, help="destination directory")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    out_dir = resolve(args.out)
    if (out_dir / "config.json").exists() and not args.force:
        print(f"Model already present at {out_dir} (pass --force to re-download).")
        return 0

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # noqa: BLE001
        print(
            "ERROR: torch/transformers are not installed. Install the CPU extra first:\n"
            "  pip install -r requirements/gltr.txt",
            file=sys.stderr,
        )
        print(f"({exc})", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model}@{args.revision} -> {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)

    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)

    num_parameters = int(sum(p.numel() for p in model.parameters()))
    info = {
        "downloaded_at": _stamp(),
        "model_name": args.model,
        "revision": args.revision,
        "destination": str(out_dir),
        "num_parameters": num_parameters,
        "num_parameters_millions": round(num_parameters / 1_000_000, 2),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "dtype": "float32",
        "device": "cpu",
        "note": "Reference causal LM for offline GLTR inference; loaded with local_files_only=True.",
    }
    (out_dir / "MODEL_INFO.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(info, indent=2))
    print(f"\nDone. Experiments will now load the model offline from {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
