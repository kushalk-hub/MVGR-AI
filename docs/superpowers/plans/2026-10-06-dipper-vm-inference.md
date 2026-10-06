# DIPPER Inference Pipeline for a 24 GB VM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the reversed L0–L4 paraphrase ladder, make generation reproducible, and provide two interchangeable environment-setup paths gated by a VRAM preflight.

**Architecture:** A new pure-Python module `paraphrase/levels.py` owns the DIPPER diversity↔similarity conversion and the per-unit seed derivation, so both are testable without loading an 11B model. `paraphrase/dipper_generate.py` imports from it and keeps its existing column-first, append-only structure. Setup lives in `setup/`, where `preflight.py` is the single gate that both the native shell path and the Dockerfile consume.

**Tech Stack:** Python 3.11, PyTorch, transformers, bitsandbytes (int8), pytest, bash, Docker.

## Global Constraints

- Scope is **paraphrase generation only**. Do not modify `detectors/` or `evaluation/`. No detector scoring, AUROC, or evaluation-stage changes.
- **Config values are DIVERSITY.** The DIPPER model reads **SIMILARITY**. Conversion is `code = 100 - diversity`, applied in exactly one place (`paraphrase/levels.py`).
- Valid diversity values are multiples of 20 in `{0, 20, 40, 60, 80, 100}`. Reject anything else.
- The level ladder is fixed: L1 `(20, 0)`, L2 `(40, 20)`, L3 `(60, 40)`, L4 `(60, 60)`.
- Quantization is **int8 only**, via `BitsAndBytesConfig(load_in_8bit=True)`. Do not add an fp16 or 4-bit mode.
- Traversal stays **column-first** (source-major). Do not restructure the generation loop for batching.
- **Do not implement equal-length truncation.** The paper truncates AI text, paraphrase, and human text to the shortest word count before scoring, to stop length becoming a detector cue. That is an evaluation-stage control, out of scope here, and would mean destroying generated text. `qc.py` already records `mean_words` per level, so drift stays visible without modifying data.
- Do **not** implement equal-length truncation. It is an evaluation-stage requirement, explicitly out of scope.
- Never silently loosen a pinned dependency. Report resolution failures with the available options.
- Every script is run from the repo root and takes cwd-relative paths.
- This plan is developed on **Windows** but the artifacts target a **Linux VM**. Tests asserting POSIX-local properties (the exec bit, or driving `bash` with a temp-dir stub) must be guarded with `@pytest.mark.skipif(os.name == "nt")` so they skip locally and execute in full on the VM during Task 9. Content-only assertions run everywhere.
- `data/raw/*.csv`, `data/pilot/*.csv`, and `results/*.csv|*.png` are gitignored. Do not force-add generated data.
- Commit after every task. Do not amend.

---

## File Structure

| File | Responsibility |
|---|---|
| `paraphrase/levels.py` | **Create.** Pure functions: diversity↔similarity conversion, validation, per-unit seed derivation. No torch, no transformers — importable in a bare test run. |
| `paraphrase/dipper_generate.py` | **Modify.** Import from `levels.py`, apply per-unit seed, write manifest, print error tally. Loop structure unchanged. |
| `paraphrase/qc.py` | **Modify.** Add `--strict` monotonic-overlap gate. |
| `configs/dipper_levels.json` | **Modify.** Replace with the fixed diversity ladder. |
| `setup/preflight.py` | **Create.** Environment gate. Non-zero exit with actionable message. |
| `setup/setup_env.sh` | **Create.** Native venv setup path. |
| `setup/run_generation.sh` | **Create.** preflight → prepare → generate → qc. |
| `Dockerfile` | **Create.** Container setup path, same entrypoint. |
| `tests/test_levels.py` | **Create.** Pure-function tests, no GPU needed. |
| `tests/test_qc_strict.py` | **Create.** `--strict` gate tests, no GPU needed. |
| `AGENTS.md` | **Modify.** Fix the false "L1–L3 purely lexical" line; document the scale trap and setup. |

---

## Task 1: Pure levels module with the diversity↔similarity fix

This is the highest-priority task. The ladder is currently inverted: `configs/dipper_levels.json` holds diversity values but `dipper_generate.py:83` passes them to the model raw, so the model reads them as similarity. L1 (`lex=20`) currently means diversity 80 — the ladder runs hardest-to-easiest.

**Files:**
- Create: `paraphrase/levels.py`
- Create: `tests/test_levels.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `VALID_DIVERSITIES: frozenset[int]` = `{0, 20, 40, 60, 80, 100}`
  - `control_codes(lex_diversity: int, order_diversity: int) -> tuple[int, int]` — returns `(lex_code, order_code)` as similarity values.
  - `validate_diversity(value: int, name: str) -> None` — raises `ValueError` if not in `VALID_DIVERSITIES`.
  - `unit_seed(run_seed: int, source_id, level: str) -> int` — deterministic per-`(source_id, level)` seed in `[0, 2**31)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_levels.py`:

```python
import pytest

from paraphrase.levels import (
    VALID_DIVERSITIES,
    control_codes,
    unit_seed,
    validate_diversity,
)


def test_paper_headline_pair_maps_to_documented_codes():
    # DIPPER README: "L60-O60 in the paper corresponds to lex = 40, order = 40"
    assert control_codes(60, 60) == (40, 40)


def test_controls_are_the_inverse_of_diversity():
    assert control_codes(0, 0) == (100, 100)
    assert control_codes(20, 0) == (80, 100)
    assert control_codes(40, 20) == (60, 80)
    assert control_codes(60, 40) == (40, 60)
    assert control_codes(100, 100) == (0, 0)


def test_control_codes_validates_its_inputs():
    with pytest.raises(ValueError, match="lexical diversity"):
        control_codes(25, 0)
    with pytest.raises(ValueError, match="order diversity"):
        control_codes(20, 35)


@pytest.mark.parametrize("value", sorted(VALID_DIVERSITIES))
def test_validate_diversity_accepts_the_documented_grid(value):
    validate_diversity(value, "lexical diversity")


@pytest.mark.parametrize("value", [-20, 10, 35, 75, 120])
def test_validate_diversity_rejects_undocumented_values(value):
    with pytest.raises(ValueError):
        validate_diversity(value, "lexical diversity")


def test_unit_seed_is_deterministic():
    assert unit_seed(42, 7, "L2") == unit_seed(42, 7, "L2")


def test_unit_seed_differs_per_level_and_per_source():
    assert unit_seed(42, 7, "L1") != unit_seed(42, 7, "L2")
    assert unit_seed(42, 7, "L2") != unit_seed(42, 8, "L2")


def test_unit_seed_changes_with_run_seed():
    assert unit_seed(42, 7, "L2") != unit_seed(43, 7, "L2")


def test_unit_seed_is_within_torch_seed_range():
    for sid in range(50):
        for level in ("L0", "L1", "L2", "L3", "L4"):
            seed = unit_seed(42, sid, level)
            assert 0 <= seed < 2**31


def test_unit_seed_survives_source_id_type_changes():
    # source_id arrives as int from pandas but as str when read back from CSV
    assert unit_seed(42, 7, "L2") == unit_seed(42, "7", "L2")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_levels.py -v`

Expected: collection error — `ModuleNotFoundError: No module named 'paraphrase.levels'`

- [ ] **Step 3: Create the levels module**

Create `paraphrase/levels.py`:

```python
"""DIPPER control-code and seeding helpers.

DIPPER's two knobs exist on opposite scales between the paper and the
pretrained model. The paper reports *diversity* (0 = keep the words,
100 = rewrite everything). The model reads *similarity* (100 = stay
maximally faithful, 0 = fidelity irrelevant). So a desired diversity of X
must be fed to the model as the control code 100 - X.

The DIPPER repository states this directly: "L60-O60 in the paper
corresponds to lex = 40, order = 40 as the control code input to the
model."

Values in configs/dipper_levels.json are therefore expressed as
diversity, and control_codes() is the single place the conversion
happens. This module is deliberately dependency-free so it can be
tested without loading an 11B model.
"""

import zlib

VALID_DIVERSITIES = frozenset({0, 20, 40, 60, 80, 100})

_TORCH_SEED_MODULUS = 2**31


def validate_diversity(value, name):
    """Raise ValueError unless value is a documented DIPPER diversity code."""
    if value not in VALID_DIVERSITIES:
        allowed = ", ".join(str(v) for v in sorted(VALID_DIVERSITIES))
        raise ValueError(
            f"{name} must be one of [{allowed}], got {value!r}. "
            "DIPPER controls are multiples of 20 from 0 to 100."
        )


def control_codes(lex_diversity, order_diversity):
    """Convert desired diversity to the similarity codes the model reads.

    Args:
        lex_diversity: desired lexical diversity, multiple of 20 in [0, 100].
        order_diversity: desired order diversity, multiple of 20 in [0, 100].

    Returns:
        (lex_code, order_code) as similarity values for the DIPPER prompt.
    """
    validate_diversity(lex_diversity, "lexical diversity")
    validate_diversity(order_diversity, "order diversity")
    return int(100 - lex_diversity), int(100 - order_diversity)


def unit_seed(run_seed, source_id, level):
    """Derive a stable seed for one (source_id, level) generation unit.

    Uses zlib.crc32 rather than the builtin hash() because hash() on str is
    salted per process by PYTHONHASHSEED, which would silently break
    reproducibility across runs.

    source_id is normalised to str so a pandas int column and the same id
    read back from CSV derive the same seed.
    """
    key = f"{run_seed}|{source_id}|{level}".encode("utf-8")
    return zlib.crc32(key) % _TORCH_SEED_MODULUS
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_levels.py -v`

Expected: PASS — all tests green.

- [ ] **Step 5: Commit**

```bash
git add paraphrase/levels.py tests/test_levels.py
git commit -m "feat: add DIPPER control-code and seeding module

Config values are diversity; the model reads similarity. Owning the
100 - diversity conversion in one dependency-free module keeps the two
scales from drifting apart and makes the mapping testable without
loading the 11B model."
```

---

## Task 2: Apply the fix and seed the generation loop

Task 1 added the conversion but nothing calls it yet — the ladder is still inverted. This task wires it in.

**Files:**
- Modify: `paraphrase/dipper_generate.py:75-92` (`paraphrase()`), `:116-128` (`main()` arg parsing), `:174-206` (generation loop)
- Modify: `configs/dipper_levels.json`

**Interfaces:**
- Consumes: `control_codes(lex_diversity, order_diversity) -> tuple[int, int]` and `unit_seed(run_seed, source_id, level) -> int` from `paraphrase.levels` (Task 1).
- Produces: `paraphrase.dipper_generate.paraphrase(model, tokenizer, text, lex_diversity, order_diversity, max_length, top_p, sent_interval) -> str` — same signature as before, but `lex_diversity`/`order_diversity` now mean **diversity** and are converted internally.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_levels.py`:

```python
def test_ladder_config_is_expressed_as_diversity():
    import json
    from pathlib import Path

    levels = json.loads(
        Path("configs/dipper_levels.json").read_text(encoding="utf-8")
    )["levels"]

    by_level = {entry["level"]: entry for entry in levels}
    assert set(by_level) == {"L1", "L2", "L3", "L4"}

    assert (by_level["L1"]["lex"], by_level["L1"]["order"]) == (20, 0)
    assert (by_level["L2"]["lex"], by_level["L2"]["order"]) == (40, 20)
    assert (by_level["L3"]["lex"], by_level["L3"]["order"]) == (60, 40)
    assert (by_level["L4"]["lex"], by_level["L4"]["order"]) == (60, 60)


def test_ladder_is_monotone_on_both_axes():
    import json
    from pathlib import Path

    levels = json.loads(
        Path("configs/dipper_levels.json").read_text(encoding="utf-8")
    )["levels"]
    ordered = [entry for entry in sorted(levels, key=lambda e: e["level"])]

    lex = [entry["lex"] for entry in ordered]
    order = [entry["order"] for entry in ordered]
    assert lex == sorted(lex)
    assert order == sorted(order)


def test_every_ladder_entry_is_a_valid_diversity_pair():
    import json
    from pathlib import Path

    levels = json.loads(
        Path("configs/dipper_levels.json").read_text(encoding="utf-8")
    )["levels"]
    for entry in levels:
        # raises ValueError on any undocumented value
        control_codes(entry["lex"], entry["order"])


class _FakeEncoding(dict):
    """Mimics a BatchEncoding closely enough for paraphrase()."""

    def to(self, device):
        return self


class _FakeModel:
    """paraphrase() only touches .device and .generate()."""

    device = "cpu"

    def __init__(self, capture):
        self._capture = capture

    def generate(self, **kwargs):
        return [[0, 1, 2]]


class _FakeTokenizer:
    def __init__(self, capture):
        self._capture = capture

    def __call__(self, texts, **kwargs):
        self._capture["prompt"] = texts[0]
        return _FakeEncoding(input_ids=[[0, 1, 2]])

    def batch_decode(self, ids, **kwargs):
        return ["rewritten"]


def test_paraphrase_prompt_uses_similarity_codes(monkeypatch):
    """The prompt must carry similarity codes, not the raw diversity values."""
    from paraphrase import dipper_generate

    capture = {}
    monkeypatch.setattr(
        dipper_generate, "sent_tokenize", lambda text: ["one two three."]
    )

    out = dipper_generate.paraphrase(
        model=_FakeModel(capture),
        tokenizer=_FakeTokenizer(capture),
        text="one two three.",
        lex_diversity=60,
        order_diversity=60,
        max_length=512,
        top_p=0.75,
        sent_interval=3,
    )

    assert "lexical = 40, order = 40" in capture["prompt"]
    assert "lexical = 60, order = 60" not in capture["prompt"]
    assert out == "rewritten"


def test_paraphrase_rejects_undocumented_diversity(monkeypatch):
    from paraphrase import dipper_generate

    capture = {}
    monkeypatch.setattr(dipper_generate, "sent_tokenize", lambda text: ["a b c."])

    with pytest.raises(ValueError, match="lexical diversity"):
        dipper_generate.paraphrase(
            model=_FakeModel(capture),
            tokenizer=_FakeTokenizer(capture),
            text="a b c.",
            lex_diversity=45,
            order_diversity=0,
            max_length=512,
            top_p=0.75,
            sent_interval=3,
        )
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_levels.py -v -k "ladder or prompt"`

Expected: FAIL — `test_ladder_config_is_expressed_as_diversity` fails on L2 (`order` is 0, not 20), and `test_paraphrase_prompt_uses_similarity_codes` fails because the prompt currently emits `lexical = 60, order = 60`.

- [ ] **Step 3: Fix the ladder config**

Replace `configs/dipper_levels.json` with:

```json
{
  "_comment": "Values are DIVERSITY (0 = keep the words, 100 = rewrite everything). DIPPER reads similarity, so paraphrase/dipper_generate.py converts each to 100 - diversity via control_codes() in paraphrase/levels.py. Order diversity varies per level on purpose: it reorders sentences, which perturbs discourse structure, not just wording.",
  "levels": [
    {"level": "L1", "lex": 20, "order": 0},
    {"level": "L2", "lex": 40, "order": 20},
    {"level": "L3", "lex": 60, "order": 40},
    {"level": "L4", "lex": 60, "order": 60}
  ]
}
```

- [ ] **Step 4: Convert inside `paraphrase()`**

In `paraphrase/dipper_generate.py`, add the import after the existing imports. The file currently starts with:

```python
import argparse
import csv
import json
import nltk
import os
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
```

Change the first line block to insert the path fix and package import, so the module works both as `python paraphrase/dipper_generate.py` and as `python -m paraphrase.dipper_generate`:

```python
import argparse
import csv
import json
import os
import sys
import zlib

import nltk
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paraphrase.levels import control_codes, unit_seed
```

Then replace the body of `paraphrase()` (currently lines 75-92) so the control codes are computed once and the prompt uses them:

```python
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
```

- [ ] **Step 5: Add the `--seed` flag**

In `main()`, after the `--limit` argument line, add:

```python
    parser.add_argument("--seed", type=int, default=42)
```

- [ ] **Step 6: Seed each unit and update the call site**

In the generation loop of `main()`, replace the existing block:

```python
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
```

with:

```python
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
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_levels.py -v`

Expected: PASS — all tests green, including the prompt and ladder tests.

- [ ] **Step 8: Verify the module imports both ways**

Run: `python -c "from paraphrase.dipper_generate import control_codes; print(control_codes(60,60))"`

Expected: prints `(40, 40)`

Run: `python paraphrase/dipper_generate.py --help`

Expected: usage text listing `--seed`, no `ImportError`.

- [ ] **Step 9: Commit**

```bash
git add configs/dipper_levels.json paraphrase/dipper_generate.py tests/test_levels.py
git commit -m "fix: correct inverted DIPPER ladder and seed generation per unit

configs/dipper_levels.json held diversity values but the prompt was
built from them raw, so the model read them as similarity and the
attack ran hardest-to-easiest. Control codes are now derived via
control_codes(), and the config carries the documented diversity ladder
with order diversity varying per level.

Generation was also unseeded, so re-runs could not reproduce a dataset.
Each (source_id, level) unit now derives a stable seed via unit_seed()."
```

---

## Task 3: Provenance manifest and error tally

Per-unit failures currently land in `quality_status` and are never counted, so a run can look clean while most units failed. This task makes that visible and records the run configuration.

**Files:**
- Modify: `paraphrase/dipper_generate.py` (imports, `main()` end)

**Interfaces:**
- Consumes: `control_codes`, `unit_seed` from `paraphrase.levels` (Task 1).
- Produces: `data/pilot/generation_manifest.json`, written at the end of a successful `main()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_generation_manifest.py`:

```python
import json

import pytest

from paraphrase import dipper_generate


def test_build_manifest_records_the_run_configuration():
    manifest = dipper_generate.build_manifest(
        run_seed=42,
        model="test/model",
        tokenizer="test/tokenizer",
        levels=[
            {"level": "L1", "lex": 20, "order": 0},
            {"level": "L2", "lex": 40, "order": 20},
            {"level": "L3", "lex": 60, "order": 40},
            {"level": "L4", "lex": 60, "order": 60},
        ],
        top_p=0.75,
        max_length=512,
        sent_interval=3,
        tally={"ok": 1000, "empty": 0, "error": 0},
    )

    assert manifest["seed"] == 42
    assert manifest["model"] == "test/model"
    assert manifest["tokenizer"] == "test/tokenizer"
    assert manifest["load_in_8bit"] is True
    assert manifest["top_p"] == 0.75
    assert manifest["max_length"] == 512
    assert manifest["sent_interval"] == 3
    assert manifest["tally"] == {"ok": 1000, "empty": 0, "error": 0}

    # Levels are recorded as diversity AND as the codes the model read.
    assert manifest["levels"][0]["lex_diversity"] == 20
    assert manifest["levels"][0]["lex_code"] == 80

    for field in ("torch", "transformers", "bitsandbytes", "generated_at"):
        assert field in manifest


def test_tally_counts_a_mixed_run():
    counts = dipper_generate.Tally()
    counts.record("ok")
    counts.record("ok")
    counts.record("empty")
    counts.record("error")
    assert counts.as_dict() == {"ok": 2, "empty": 1, "error": 1}


def test_tally_starts_at_zero():
    assert dipper_generate.Tally().as_dict() == {"ok": 0, "empty": 0, "error": 0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_generation_manifest.py -v`

Expected: FAIL — `AttributeError: module 'paraphrase.dipper_generate' has no attribute 'Tally'`

- [ ] **Step 3: Add the Tally class and build_manifest**

In `paraphrase/dipper_generate.py`, add `from datetime import datetime, timezone` to the imports, then place these two definitions immediately before `def main():`:

```python
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


def build_manifest(run_seed, model, tokenizer, levels, top_p, max_length, sent_interval, tally):
    """Assemble the generation provenance record.

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
```

- [ ] **Step 4: Wire the tally and manifest into the run**

In `main()`, after `done = load_done_keys(args.output)`, add:

```python
    tally = Tally()
```

In the generation loop, add `tally.record(status)` immediately after the `status` is assigned in both branches — that is, right after `status = "ok" if out else "empty"` and right after `status = "error"`. Add it once, after the try/except block, so both paths are counted:

```python
        tally.record(status)
```

Then replace the final print at the end of `main()`:

```python
    print(f"Paraphrase generation complete -> {args.output}")
```

with:

```python
    manifest = build_manifest(
        run_seed=args.seed,
        model=args.model,
        tokenizer=DEFAULT_TOKENIZER,
        levels=levels,
        top_p=args.top_p,
        max_length=args.max_length,
        sent_interval=args.sent_interval,
        tally=tally.as_dict(),
    )
    manifest_path = os.path.join(
        os.path.dirname(args.output) or ".", "generation_manifest.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    counts = tally.as_dict()
    total = sum(counts.values()) or 1
    print(f"Paraphrase generation complete -> {args.output}")
    print(
        f"  ok={counts['ok']} empty={counts['empty']} error={counts['error']} "
        f"({counts['error'] * 100.0 / total:.1f}% failed)"
    )
    print(f"  manifest -> {manifest_path}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_generation_manifest.py -v`

Expected: PASS — all tests green.

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest tests/ -v`

Expected: PASS — no regressions in the levels tests.

- [ ] **Step 7: Commit**

```bash
git add paraphrase/dipper_generate.py tests/test_generation_manifest.py
git commit -m "feat: record generation provenance and per-unit outcome tally

Per-unit failures were swallowed into quality_status with no totals, so
a run could look clean while most units failed. main() now prints an
ok/empty/error summary and writes generation_manifest.json recording the
seed, model, tokenizer, and each level's diversity alongside the
similarity codes the model actually read."
```

---

## Task 4: QC strict gate

Without this, a broken ladder is invisible: `qc.py` prints overlap numbers but exits 0 either way. The gate turns "overlap rose as levels increased" into a hard failure.

**Files:**
- Modify: `paraphrase/qc.py:13-16` (arg parsing), `:47-50` (output)
- Create: `tests/test_qc_strict.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `paraphrase.qc.monotonicity_violations(rows) -> list[tuple[str, str]]` — returns `(prev_level, level)` pairs where overlap increases. `rows` is the list of dicts already built in `main()`, each with `level` and `mean_jaccard_overlap_L0` keys, ordered L0→L4.
- `paraphrase/qc.main()` gains `--strict` (flag) and exits non-zero on any violation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_strict.py`:

```python
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
    violations = monotonicity_violations(
        rows(("L0", 1.0), ("L1", 0.40), ("L2", 0.55), ("L3", 0.70), ("L4", 0.85))
    )
    assert violations == [("L0", "L1"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4")]


def test_single_row_is_trivially_valid():
    assert monotonicity_violations(rows(("L0", 0.5))) == []


def test_gap_in_level_coverage_is_still_compared():
    # Absent levels are evaluate.py's hard guard, not qc.py's job. The two rows
    # present are still compared, and a genuine drop is not a violation.
    assert monotonicity_violations(rows(("L0", 1.0), ("L4", 0.30))) == []
```

The gap case above shows `qc.py` compares whatever rows exist; detecting *absent*
levels is `evaluate.py`'s job via its hard `sys.exit(1)` guard.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_qc_strict.py -v`

Expected: collection error — `ImportError: cannot import name 'monotonicity_violations' from 'paraphrase.qc'`

- [ ] **Step 3: Add the violation detector**

In `paraphrase/qc.py`, after the `jaccard_overlap` function, add:

```python
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
```

- [ ] **Step 4: Add the `--strict` flag and gate**

In `main()`, replace the argument parser block with one that adds the flag:

```python
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pilot/pilot_dataset.csv")
    parser.add_argument("--output", default="data/qc_overlap.csv")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if mean Jaccard overlap against L0 rises from one level to the next.",
    )
    args = parser.parse_args()
```

Then replace the tail of `main()`, currently:

```python
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"QC saved -> {args.output}")
```

with:

```python
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
```

Add `import sys` to the imports at the top of `paraphrase/qc.py`, which currently reads:

```python
import argparse
import pandas as pd
```

so it becomes:

```python
import argparse
import sys

import pandas as pd
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_qc_strict.py -v`

Expected: PASS — all tests green.

- [ ] **Step 6: Verify the gate end to end on synthetic data**

Run:

```bash
python -c "
import pandas as pd
df = pd.DataFrame([
    {'sample_id':'1_L0','source_id':1,'text':'alpha beta','label':'ai','paraphrase_level':'L0'},
    {'sample_id':'1_L1','source_id':1,'text':'alpha beta','label':'ai','paraphrase_level':'L1'},
])
df.to_csv('/tmp/qc_inverted.csv', index=False)
"
python paraphrase/qc.py --input /tmp/qc_inverted.csv --output /tmp/qc_out.csv --strict
```

Expected: exit code 1 with `STRICT FAIL` naming the offending pair.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -v`

Expected: PASS — no regressions.

- [ ] **Step 8: Commit**

```bash
git add paraphrase/qc.py tests/test_qc_strict.py
git commit -m "feat: add --strict monotonic overlap gate to QC

qc.py printed overlap numbers but always exited 0, so an inverted
control-code ladder was silent. --strict exits non-zero when mean
Jaccard overlap against L0 rises between levels and points at the
likely cause."
```

---

## Task 5: Preflight environment gate

The repo's current failure mode is silent: `calibrate_binoculars.py` and `evaluate.py` both `return` with exit 0 when their input is missing, so a broken chain reads as a clean run. The preflight inverts that — non-zero exit with an actionable message, so generation never starts on a bad environment.

**Files:**
- Create: `setup/preflight.py`

**Interfaces:**
- Consumes: nothing.
- Produces: exit code 0 when the environment can run DIPPER int8, non-zero otherwise. Prints device name, total/free VRAM, package versions, and post-load VRAM.

- [ ] **Step 1: Write the failing test**

Create `tests/test_preflight.py`:

```python
import pytest

from setup import preflight


def test_check_reports_a_healthy_environment(monkeypatch, capsys):
    fake = preflight.Environment(
        torch_version="2.5.1",
        transformers_version="4.44.0",
        bitsandbytes_version="0.43.0",
        cuda_available=True,
        device_name="NVIDIA A10",
        total_vram_gb=24.0,
        free_vram_gb=23.5,
        model_loads=True,
        loaded_vram_gb=11.2,
    )
    assert preflight.check(fake) == 0
    out = capsys.readouterr().out
    assert "NVIDIA A10" in out
    assert "24.0" in out


def test_check_fails_without_cuda(monkeypatch, capsys):
    fake = preflight.Environment(
        torch_version="2.5.1",
        transformers_version="4.44.0",
        bitsandbytes_version="0.43.0",
        cuda_available=False,
        device_name="",
        total_vram_gb=0.0,
        free_vram_gb=0.0,
        model_loads=False,
        loaded_vram_gb=0.0,
    )
    assert preflight.check(fake) != 0
    assert "CUDA" in capsys.readouterr().out


def test_check_fails_when_model_cannot_load(monkeypatch, capsys):
    fake = preflight.Environment(
        torch_version="2.5.1",
        transformers_version="4.44.0",
        bitsandbytes_version="0.43.0",
        cuda_available=True,
        device_name="NVIDIA A10",
        total_vram_gb=24.0,
        free_vram_gb=23.5,
        model_loads=False,
        loaded_vram_gb=0.0,
    )
    assert preflight.check(fake) != 0
    assert "int8" in capsys.readouterr().out


def test_check_fails_on_insufficient_vram(monkeypatch, capsys):
    fake = preflight.Environment(
        torch_version="2.5.1",
        transformers_version="4.44.0",
        bitsandbytes_version="0.43.0",
        cuda_available=True,
        device_name="NVIDIA T4",
        total_vram_gb=15.0,
        free_vram_gb=14.0,
        model_loads=False,
        loaded_vram_gb=0.0,
    )
    assert preflight.check(fake) != 0
    out = capsys.readouterr().out
    assert "VRAM" in out


def test_required_vram_for_dipper_int8_is_eleven_gib():
    # DIPPER XXL in int8 is ~11GB; anything under this cannot load.
    assert preflight.REQUIRED_VRAM_GB <= 12.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_preflight.py -v`

Expected: collection error — `ModuleNotFoundError: No module named 'setup'`

- [ ] **Step 3: Create the preflight module**

Create `setup/preflight.py`:

```python
"""Environment gate for DIPPER int8 generation.

Exits non-zero with an actionable message when the environment cannot run
the pipeline. This is deliberately the inverse of the rest of the repo,
where a missing input makes a script return exit code 0 and a broken
chain reads as a clean run.

Consumed by both setup/setup_env.sh and the Dockerfile so the two setup
paths cannot drift apart.
"""

import sys

# DIPPER XXL in int8 needs roughly 11GB; leave headroom for activations.
REQUIRED_VRAM_GB = 12.0

DIPPER_MODEL = "kalpeshk2011/dipper-paraphraser-xxl"
DIPPER_TOKENIZER = "google/t5-v1_1-xxl"


class Environment:
    """Everything preflight needs to know, as plain data.

    Kept as a plain object so check() can be tested without a GPU.
    """

    def __init__(
        self,
        torch_version,
        transformers_version,
        bitsandbytes_version,
        cuda_available,
        device_name,
        total_vram_gb,
        free_vram_gb,
        model_loads,
        loaded_vram_gb,
    ):
        self.torch_version = torch_version
        self.transformers_version = transformers_version
        self.bitsandbytes_version = bitsandbytes_version
        self.cuda_available = cuda_available
        self.device_name = device_name
        self.total_vram_gb = total_vram_gb
        self.free_vram_gb = free_vram_gb
        self.model_loads = model_loads
        self.loaded_vram_gb = loaded_vram_gb


def check(env):
    """Print a report and return 0 if generation can proceed, else non-zero."""
    print("=== PARAGON Stage 1 preflight ===")
    print(f"torch        {env.torch_version}")
    print(f"transformers {env.transformers_version}")
    print(f"bitsandbytes {env.bitsandbytes_version}")

    if not env.cuda_available:
        print("\nFAIL: CUDA is not available to torch.")
        print("  Fix: check `nvidia-smi` works, and that torch is a CUDA build.")
        print("       On Linux, a torch CPU wheel is the usual cause.")
        return 1

    print(f"device       {env.device_name}")
    print(f"vram         {env.free_vram_gb:.1f} GiB free / {env.total_vram_gb:.1f} GiB total")

    if env.total_vram_gb < REQUIRED_VRAM_GB:
        print(f"\nFAIL: {env.total_vram_gb:.1f} GiB VRAM is below the {REQUIRED_VRAM_GB:.1f} GiB")
        print("  DIPPER XXL needs for int8 loading.")
        print("  Fix: use a larger GPU, or confirm quantization_config is int8.")
        return 1

    if not env.model_loads:
        print("\nFAIL: DIPPER int8 model did not load.")
        print(f"  Fix: confirm bitsandbytes works on this GPU, then retry:")
        print(f"       python -c \"from transformers import BitsAndBytesConfig; "
              f"BitsAndBytesConfig(load_in_8bit=True)\"")
        return 1

    print(f"model loaded {env.loaded_vram_gb:.1f} GiB resident")
    print("\nOK: environment is ready for DIPPER int8 generation.")
    return 0


def probe():
    """Inspect the real environment and load DIPPER in int8."""
    import torch
    import transformers

    cuda_available = torch.cuda.is_available()
    device_name = ""
    total_vram_gb = 0.0
    free_vram_gb = 0.0

    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        device_name = properties.name
        total_vram_gb = properties.total_memory / (1024**3)
        free_vram_gb = total_vram_gb - torch.cuda.mem_get_info(0)[0] / (1024**3)

    try:
        import bitsandbytes

        bnb_version = bitsandbytes.__version__
    except Exception as exc:
        bnb_version = f"MISSING ({type(exc).__name__})"

    model_loads = False
    loaded_vram_gb = 0.0
    if cuda_available and total_vram_gb >= REQUIRED_VRAM_GB:
        try:
            from transformers import (
                AutoTokenizer,
                BitsAndBytesConfig,
                T5ForConditionalGeneration,
            )

            tokenizer = AutoTokenizer.from_pretrained(DIPPER_TOKENIZER)
            model = T5ForConditionalGeneration.from_pretrained(
                DIPPER_MODEL,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                loaded_vram_gb = torch.cuda.memory_reserved(0) / (1024**3)
            model_loads = True
        except Exception as exc:
            print(f"\nModel load raised: {type(exc).__name__}: {exc}")

    return Environment(
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        bitsandbytes_version=bnb_version,
        cuda_available=cuda_available,
        device_name=device_name,
        total_vram_gb=total_vram_gb,
        free_vram_gb=free_vram_gb,
        model_loads=model_loads,
        loaded_vram_gb=loaded_vram_gb,
    )


if __name__ == "__main__":
    sys.exit(check(probe()))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_preflight.py -v`

Expected: PASS — all tests green.

Note: `setup/` has no `__init__.py` and does not need one, since Python 3
resolves it as a namespace package.

- [ ] **Step 5: Run preflight for real**

Run: `python setup/preflight.py`

Expected: prints the report and exits 0 on a CUDA machine with enough VRAM. On a
CPU-only machine it exits 1 with the CUDA fix-it message. Either outcome is a
pass for this task — the point is that the exit code is meaningful.

- [ ] **Step 6: Commit**

```bash
git add setup/preflight.py tests/test_preflight.py
git commit -m "feat: add DIPPER int8 environment preflight gate

Verifies CUDA, VRAM headroom, bitsandbytes, and an actual int8 model
load, exiting non-zero with a fix-it message on failure. Inverts the
repo's habit of returning exit 0 when an input is missing."
```

---

## Task 6: Native setup path and run script

**Files:**
- Create: `setup/setup_env.sh`
- Create: `setup/run_generation.sh`

**Interfaces:**
- Consumes: `setup/preflight.py` (Task 5), `requirements.txt`.
- Produces: two executable shell scripts. `setup_env.sh` builds `.venv`; `run_generation.sh` runs the generation chain and exits non-zero if any stage fails.

- [ ] **Step 1: Write the failing test**

Create `tests/test_setup_scripts.py`:

```python
import os
import stat
import subprocess
from pathlib import Path

import pytest

SETUP_ENV = Path("setup/setup_env.sh")
RUN_GENERATION = Path("setup/run_generation.sh")

# The scripts are POSIX and target a Linux VM. These assertions check
# POSIX-local properties -- the exec bit, and driving bash with a temp-dir
# interpreter stub -- that a Windows checkout cannot satisfy: the filesystem
# carries no exec bit (core.fileMode=false) and `bash` resolves to WSL, which
# cannot see Windows temp paths. They run in full on the VM (Task 9); on
# Windows, Task 6 verifies via `bash -n` plus these content checks.
posix_only = pytest.mark.skipif(
    os.name == "nt", reason="POSIX-local assertions; run on the Linux VM"
)


def read(path):
    return path.read_text(encoding="utf-8")


def test_setup_env_exists():
    assert SETUP_ENV.exists()


def test_run_generation_exists():
    assert RUN_GENERATION.exists()


@posix_only
def test_setup_env_is_executable():
    mode = SETUP_ENV.stat().st_mode
    assert mode & stat.S_IXUSR, "setup_env.sh must be executable"


@posix_only
def test_run_generation_is_executable():
    mode = RUN_GENERATION.stat().st_mode
    assert mode & stat.S_IXUSR, "run_generation.sh must be executable"


def test_setup_env_starts_with_a_shebang():
    for script in (SETUP_ENV, RUN_GENERATION):
        assert read(script).startswith("#!/usr/bin/env bash")


def test_setup_env_fails_fast_on_bad_python():
    result = subprocess.run(
        ["bash", str(SETUP_ENV), "--python", "/nonexistent/python"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_run_generation_stops_when_preflight_fails(tmp_path):
    """A broken preflight must abort the chain, not let generation start."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    stub = fake_bin / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == *preflight* ]]; then exit 3; fi\n"
        "echo \"STUB RAN: $*\"\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["PYTHON_BIN"] = str(stub)

    result = subprocess.run(
        ["bash", str(RUN_GENERATION)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "STUB RAN" not in result.stdout
```

Mark `test_setup_env_fails_fast_on_bad_python` and
`test_run_generation_stops_when_preflight_fails` with `@posix_only` as well,
since both invoke `bash` with a stubbed interpreter.


def test_run_generation_runs_the_full_chain_in_order():
    body = read(RUN_GENERATION)
    steps = [
        "setup/preflight.py",
        "data/prepare_pilot.py",
        "paraphrase/dipper_generate.py",
        "paraphrase/qc.py",
    ]
    positions = [body.index(step) for step in steps]
    assert positions == sorted(positions), "chain steps are out of order"
    assert "--strict" in body


def test_run_generation_honours_limit_and_seed_env():
    body = read(RUN_GENERATION)
    assert "DIPPER_LIMIT" in body
    assert "DIPPER_SEED" in body


def test_setup_env_reports_a_resolver_failure_instead_of_retrying():
    """A broken pip install must surface with guidance, not be retried."""
    script = read(SETUP_ENV)
    assert "pip install -r requirements.txt" in script
    # The failure branch must not silently change the dependency set.
    assert "--no-deps" not in script
    assert "uninstall" not in script
    assert "pip install torch" not in script


def test_setup_env_respects_an_explicit_python_flag():
    assert "--python" in read(SETUP_ENV)
```

Note which tests are platform-independent: `test_setup_env_exists`,
`test_run_generation_exists`, `test_setup_env_starts_with_a_shebang`,
`test_run_generation_runs_the_full_chain_in_order`,
`test_run_generation_honours_limit_and_seed_env`, and
`test_setup_env_reports_a_resolver_failure_instead_of_retrying` all read file
contents only, so they run everywhere. The four `@posix_only` tests are the
ones that execute or stat the scripts.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_setup_scripts.py -v`

Expected: FAIL — `test_setup_env_exists` fails; the files do not exist yet.
On Windows the `@posix_only` tests report as skipped rather than failing.

- [ ] **Step 3: Create `setup/setup_env.sh`**

Create `setup/setup_env.sh`:

```bash
#!/usr/bin/env bash
# Build the native Python environment for DIPPER int8 generation.
# Primary setup path; the Dockerfile is the fallback. Both consume
# requirements.txt and setup/preflight.py so they cannot drift apart.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${1:-}" == "--python" ]]; then
    PYTHON_BIN="$2"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "FAIL: python interpreter not found: $PYTHON_BIN" >&2
    echo "  Fix: install python3, or pass --python /path/to/python3" >&2
    exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing pinned requirements"
if ! pip install -r requirements.txt; then
    echo >&2
    echo "FAIL: dependency resolution failed. Not retrying with different" >&2
    echo "      versions, because a silent change to the pinned set would" >&2
    echo "      invalidate reproducibility of the generated dataset." >&2
    echo >&2
    echo "  Known friction: requirements.txt pins bitsandbytes>=0.41,<0.46." >&2
    echo "  Options, in order of preference:" >&2
    echo "    1. Install a torch version compatible with that bitsandbytes range." >&2
    echo "    2. Widen the pin deliberately and record the change in git." >&2
    echo "  See the resolver output above for the exact conflict." >&2
    exit 1
fi

echo "==> Downloading NLTK punkt data"
python - <<'PY'
import nltk
for name in ("punkt", "punkt_tab"):
    try:
        nltk.download(name, quiet=True)
    except Exception as exc:
        print(f"  WARNING: could not download {name}: {exc}")
PY

echo "==> Running preflight"
if ! python setup/preflight.py; then
    echo >&2
    echo "FAIL: preflight rejected the environment. Fix the issue above" >&2
    echo "      before running generation." >&2
    exit 1
fi

echo "==> Writing requirements-lock.txt"
pip freeze > requirements-lock.txt

echo "==> Environment ready. Activate with: source $VENV_DIR/bin/activate"
```

- [ ] **Step 4: Create `setup/run_generation.sh`**

Create `setup/run_generation.sh`:

```bash
#!/usr/bin/env bash
# Run the Stage 1 paraphrase generation chain.
#
# preflight -> prepare_pilot -> dipper_generate -> qc --strict
#
# Every stage must succeed. set -e plus an explicit check after each call
# means a broken link stops the chain instead of producing a partial
# dataset that looks complete.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="$PYTHON_BIN"
elif [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="${PYTHON_BIN:-python}"
else
    PY="${PYTHON_BIN:-python3}"
fi

DIPPER_LIMIT="${DIPPER_LIMIT:-0}"
DIPPER_SEED="${DIPPER_SEED:-42}"

echo "==> [1/4] Preflight"
"$PY" setup/preflight.py

echo "==> [2/4] Preparing pilot source"
"$PY" data/prepare_pilot.py

echo "==> [3/4] Generating paraphrases (limit=$DIPPER_LIMIT seed=$DIPPER_SEED)"
"$PY" paraphrase/dipper_generate.py --limit "$DIPPER_LIMIT" --seed "$DIPPER_SEED"

echo "==> [4/4] QC (strict)"
"$PY" paraphrase/qc.py --strict

echo "==> Done. Artifacts:"
echo "    data/pilot/pilot_dataset.csv"
echo "    data/pilot/generation_manifest.json"
echo "    data/qc_overlap.csv"
```

- [ ] **Step 5: Mark both executable (POSIX hosts only)**

Run on Linux/macOS/VM: `bash -c "chmod +x setup/setup_env.sh setup/run_generation.sh"`

On Windows the exec bit cannot be set and git has `core.fileMode=false`, so
skip this step; the `@posix_only` tests skip correspondingly and Task 9 confirms
executability on the VM.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_setup_scripts.py -v`

Expected: PASS, with the four `@posix_only` tests reported as skipped on Windows.

- [ ] **Step 7: Syntax-check both scripts**

Run: `bash -n setup/setup_env.sh && bash -n setup/run_generation.sh && echo "shell syntax OK"`

Expected: prints `shell syntax OK`. WSL's `bash` works for this because it maps
the Windows cwd into `/mnt/...`.

- [ ] **Step 8: Verify the chain aborts on a failing stage**

Run on the VM, or on any POSIX host with a GPU: `bash setup/run_generation.sh`

Expected: fails at the preflight stage when VRAM or CUDA is inadequate, and does
**not** reach `prepare_pilot.py`. This proves the fail-fast behaviour on a real
machine, matching the stubbed test.

If you need to verify the abort locally on Windows, run it under WSL from
`/mnt/d/MVGR-AI` instead — the stubbed test is skipped there, so this is the only
local coverage of the abort path.

- [ ] **Step 9: Commit**

```bash
git add setup/setup_env.sh setup/run_generation.sh tests/test_setup_scripts.py
git add --chmod=+x setup/setup_env.sh setup/run_generation.sh 2>/dev/null || true
git commit -m "feat: add native setup path and fail-fast generation runner

setup_env.sh builds the venv, installs pinned requirements, downloads
punkt, runs preflight, and writes a lockfile; a resolver failure surfaces
with options instead of silently loosening the pins.

run_generation.sh chains preflight, prepare, generate, and qc --strict
with set -e plus per-stage checks, so a broken link stops the run rather
than leaving a partial dataset that looks complete."
```

---

## Task 7: Dockerfile fallback path

**Files:**
- Create: `Dockerfile`

**Interfaces:**
- Consumes: `requirements.txt`, `setup/preflight.py` (Task 5), `setup/run_generation.sh` (Task 6).
- Produces: a container image whose entrypoint is `setup/run_generation.sh`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dockerfile.py`:

```python
import re
from pathlib import Path

DOCKERFILE = Path("Dockerfile")


def read():
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_exists():
    assert DOCKERFILE.exists()


def test_base_image_is_a_build_arg():
    body = read()
    assert re.search(r"^ARG CUDA_IMAGE=", body, re.MULTILINE)
    assert "FROM ${CUDA_IMAGE}" in body


def test_base_image_is_a_cuda_runtime_with_cudnn():
    body = read()
    default = re.search(r"^ARG CUDA_IMAGE=(\S+)", body, re.MULTILINE).group(1)
    assert default.startswith("nvidia/cuda:")
    assert "runtime" in default
    assert "cudnn" in default


def test_installs_the_shared_requirements_not_a_private_copy():
    body = read()
    assert "requirements.txt" in body
    # A hardcoded pip list would drift from the venv path.
    assert not re.search(r"pip install .*transformers==", body)


def test_does_not_copy_the_host_venv():
    body = read()
    assert ".venv" not in body


def test_entrypoint_delegates_to_the_shared_runner():
    body = read()
    assert "setup/run_generation.sh" in body
    assert "ENTRYPOINT" in body or "CMD" in body


def test_dipper_cache_dir_is_configured():
    body = read()
    assert "HF_HOME" in body
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dockerfile.py -v`

Expected: FAIL — `assert DOCKERFILE.exists()` fails.

- [ ] **Step 3: Create the Dockerfile**

Create `Dockerfile`:

```dockerfile
# Container fallback for the native setup path in setup/setup_env.sh.
# Both consume requirements.txt and setup/preflight.py, so the two routes
# install the same dependency set and pass the same environment gate.
#
# Build:
#   docker build -t paragon-stage1 .
#   docker run --gpus all --rm \
#     -e DIPPER_LIMIT=0 -e DIPPER_SEED=42 \
#     -v "$PWD/data:/work/paragon/data" \
#     paragon-stage1
#
# The CUDA image tag must match the host driver. Check `nvcc --version` and
# `nvidia-smi` on the VM; override at build time if needed:
#   docker build --build-arg CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04 .

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    NLTK_DATA=/opt/nltk-data

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/paragon

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Sentence splitting needs punkt at generation time.
RUN python3 -c "\
import nltk; \
[nltk.download(name, quiet=True) for name in ('punkt', 'punkt_tab')]"

COPY . .

RUN chmod +x setup/run_generation.sh

ENTRYPOINT ["setup/run_generation.sh"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dockerfile.py -v`

Expected: PASS — all tests green.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile tests/test_dockerfile.py
git commit -m "feat: add Dockerfile fallback for the VM environment

Consumes the same requirements.txt and setup/preflight.py as the
native venv path, so both routes install one dependency set and pass
one environment gate. CUDA image is a build ARG because the VM's CUDA
version must match the host driver."
```

---

## Task 8: Correct AGENTS.md

The existing file states "L1–L3 are lex 20/40/60 with order 0", which was true before this work and is false after it. A stale agent instruction is worse than a missing one.

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code. Documentation accuracy.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agents_md_accuracy.py`:

```python
from pathlib import Path

AGENTS = Path("AGENTS.md")


def read():
    return AGENTS.read_text(encoding="utf-8")


def test_agents_md_exists():
    assert AGENTS.exists()


def test_stale_purely_lexical_claim_is_gone():
    body = read()
    assert "L1-L3 are lex 20/40/60 with order 0" not in body
    assert "purely lexical" not in body


def test_agents_md_records_the_scale_trap():
    body = read().lower()
    assert "diversity" in body
    assert "similarity" in body


def test_agents_md_documents_the_ladder():
    body = read()
    assert "| L1 | 20 | 0 | 80, 100 |" in body
    assert "| L4 | 60 | 60 | 40, 40 |" in body


def test_agents_md_documents_setup_and_seeding():
    body = read()
    assert "setup_env.sh" in body
    assert "preflight" in body.lower()
    assert "seed" in body.lower()


def test_agents_md_keeps_the_pipeline_order_section():
    assert "## Pipeline order" in read()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_agents_md_accuracy.py -v`

Expected: FAIL — `test_stale_purely_lexical_claim_is_gone` and
`test_agents_md_records_the_scale_trap` fail against the current text.

- [ ] **Step 3: Fix the DIPPER section**

In `AGENTS.md`, replace the section currently titled `## DIPPER generation is resumable and append-only` with the following, which covers resumability, the ladder, the scale trap, and seeding:

```markdown
## DIPPER generation is resumable and append-only

`dipper_generate.py` writes humans + L0 only when the output CSV does not exist,
then skips any `(source_id, level)` pair already present. Re-running the cell
completes an interrupted run. Changing `configs/dipper_levels.json` does **not**
regenerate existing rows — delete `data/pilot/pilot_dataset.csv` to start over.
`--limit` trims AI sources only, so a limited smoke run still gets the full
human set.

### The two scales: diversity vs similarity

**`configs/dipper_levels.json` holds DIVERSITY. The DIPPER model reads
SIMILARITY.** For diversity X, feed the model `100 - X`. The paper's own
`L60-O60` means control codes `lex = 40, order = 40`.

The conversion lives in exactly one place — `control_codes()` in
`paraphrase/levels.py` — and a test pins it. Passing a raw diversity value
into the prompt inverts the whole ladder, which `qc.py --strict` catches
because overlap against L0 then *rises* with the level number.

Current ladder, all inside the paper's validated grid:

| Level | Diversity L | Diversity O | Control codes |
|---|---|---|---|
| L1 | 20 | 0 | 80, 100 |
| L2 | 40 | 20 | 60, 80 |
| L3 | 60 | 40 | 40, 60 |
| L4 | 60 | 60 | 40, 40 |

Order diversity varies per level on purpose: it reorders sentences, so it
perturbs discourse structure rather than just wording — the signal the
graph/GNN branch will eventually test.

### Seeding and provenance

Generation is seeded per `(source_id, level)` unit via `unit_seed()`
(crc32 of `run_seed|source_id|level`), so a unit reproduces independently
of traversal or resume order. `crc32` rather than `hash()` because `hash()`
on strings is salted per process by `PYTHONHASHSEED`. Each run writes
`data/pilot/generation_manifest.json` with the seed, model, tokenizer,
quantization, and each level's diversity *and* the codes the model read.
```

- [ ] **Step 4: Add a setup section**

Insert this section after the `## Pipeline order` block in `AGENTS.md`. The outer
fence is four backticks because the section itself contains a bash fence:

````markdown
## VM setup (24 GB)

Two interchangeable routes, both consuming `requirements.txt` and
`setup/preflight.py` so they cannot drift. If one fails, use the other.

```bash
# native (primary)
bash setup/setup_env.sh
bash setup/run_generation.sh

# container (fallback)
docker build -t paragon-stage1 .
docker run --gpus all --rm -v "$PWD/data:/work/paragon/data" paragon-stage1
```

`DIPPER_LIMIT` and `DIPPER_SEED` are read from the environment.
`setup/preflight.py` verifies CUDA, VRAM headroom, bitsandbytes, and an
actual int8 model load, and exits non-zero with a fix-it message — the
inverse of `calibrate_binoculars.py` and `evaluate.py`, which return exit 0
on missing input. Run it directly any time to diagnose the environment.

The Dockerfile's `CUDA_IMAGE` build ARG must match the host driver; check
`nvcc --version` on the VM.
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_agents_md_accuracy.py -v`

Expected: PASS — all tests green.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -v`

Expected: PASS — every test green.

- [ ] **Step 7: Commit**

```bash
git add AGENTS.md tests/test_agents_md_accuracy.py
git commit -m "docs: correct AGENTS.md for the diversity ladder and VM setup

The old claim that L1-L3 are purely lexical (order 0) is false under the
new ladder. Replaces it with the diversity/similarity mapping, the
actual ladder, seeding behaviour, and the two setup paths."
```

---

## Task 9: End-to-end smoke verification on the VM

Everything above is testable without a GPU. This task requires the 24 GB VM and proves the pipeline actually produces a correct dataset.

**Files:** none — verification only.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: confirmed-correct `data/pilot/pilot_dataset.csv` with a monotone overlap curve.

- [ ] **Step 1: Push the branch so the VM can clone it**

Run: `git push`

Expected: branch pushed. Substitute your branch and remote for the real ones.

- [ ] **Step 2: Set up the environment on the VM**

Run on the VM:

```bash
git clone <your-remote> paragon && cd paragon
bash setup/setup_env.sh
```

Expected: venv created, dependencies installed, preflight prints `OK`, and
`requirements-lock.txt` is written. If this fails, use the Dockerfile route in
Step 3 instead — that is what the fallback is for.

- [ ] **Step 3: Fallback path check, only if Step 2 failed**

Run on the VM:

```bash
docker build -t paragon-stage1 .
docker run --gpus all --rm \
  -e DIPPER_LIMIT=4 -e DIPPER_SEED=42 \
  -v "$PWD/data:/work/paragon/data" \
  --entrypoint setup/run_generation.sh paragon-stage1
```

Expected: the same chain runs inside the container.

- [ ] **Step 4: Run a small smoke generation**

Run on the VM:

```bash
DIPPER_LIMIT=4 DIPPER_SEED=42 bash setup/run_generation.sh
```

Expected: `data/pilot/pilot_dataset.csv` gains 4 AI sources × 4 levels, QC
prints `STRICT PASS`, and the summary shows `error=0`.

- [ ] **Step 5: Verify the manifest records the codes, not the diversity**

Run on the VM:

```bash
python -c "
import json
m = json.load(open('data/pilot/generation_manifest.json'))
print('seed       ', m['seed'])
print('int8       ', m['load_in_8bit'])
print('tally      ', m['tally'])
for lv in m['levels']:
    print(f\"{lv['level']}  diversity=({lv['lex_diversity']},{lv['order_diversity']})  \"
          f\"codes=({lv['lex_code']},{lv['order_code']})\")
"
```

Expected: L1 shows `diversity=(20,0) codes=(80,100)` and L4 shows
`diversity=(60,60) codes=(40,40)`. If diversity and codes match, the ladder is
still inverted.

- [ ] **Step 6: Verify overlap falls with level**

Run on the VM:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/qc_overlap.csv')
print(df[['level','n','mean_jaccard_overlap_L0','mean_words']].to_string(index=False))
"
```

Expected: `mean_jaccard_overlap_L0` decreases from L0 to L4. Before the fix it
rose. Record the numbers; they are the evidence the ladder is correct.

- [ ] **Step 7: Verify determinism**

Run on the VM:

```bash
rm -rf /tmp/determinism && mkdir -p /tmp/determinism
for run in a b; do
  DIPPER_LIMIT=2 DIPPER_SEED=42 bash setup/run_generation.sh >/dev/null 2>&1
  cp data/pilot/pilot_dataset.csv /tmp/determinism/$run.csv
  cp data/pilot/generation_manifest.json /tmp/determinism/$run.json
  rm -f data/pilot/pilot_dataset.csv
done
python -c "
import csv
def texts(p):
    with open(p, encoding='utf-8') as f:
        return {r['sample_id']: r['text'] for r in csv.DictReader(f)}
a, b = texts('/tmp/determinism/a.csv'), texts('/tmp/determinism/b.csv')
assert a == b, 'NOT deterministic: outputs differ between identical runs'
print('determinism OK:', len(a), 'samples byte-identical')
"
```

Expected: `determinism OK: 20 samples byte-identical` (2 sources × 4 levels + 2
L0 rows, plus humans — the exact count depends on how many human rows were
written, so assert equality rather than a fixed number).

Note: delete `data/pilot/pilot_dataset.csv` between runs so the second run
regenerates rather than skipping every unit as already-done. That deletion is
what makes this test meaningful.

- [ ] **Step 8: Run the full 250-source generation**

Run on the VM:

```bash
DIPPER_LIMIT=0 DIPPER_SEED=42 bash setup/run_generation.sh
```

Expected: the full dataset, `error=0`, `STRICT PASS`. At roughly 3,000 generate
calls this takes hours, not minutes.

- [ ] **Step 9: Confirm the full dataset shape**

Run on the VM:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/pilot/pilot_dataset.csv')
print('rows        ', len(df))
print('by level    ', df['paraphrase_level'].value_counts().to_dict())
print('by status   ', df['quality_status'].value_counts().to_dict())
print('ai sources  ', df[df['label']=='ai']['source_id'].nunique())
"
```

Expected: 250 AI sources; L1–L4 each 250 rows; L0 250; humans 250; all statuses
`ok`.

- [ ] **Step 10: Commit the VM evidence**

The generated data is gitignored, so there is nothing to commit. Record the QC
table and the determinism result in the project report instead:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/qc_overlap.csv')
print(df.to_markdown(index=False))
" > docs/qc_overlap_250.md
git add docs/qc_overlap_250.md
git commit -m "docs: record 250-source QC overlap and determinism evidence"
```

---

## Appendix: Commands Reference

```bash
# Tests (no GPU needed for tasks 1-8)
python -m pytest tests/ -v

# Individual suites
python -m pytest tests/test_levels.py -v
python -m pytest tests/test_generation_manifest.py -v
python -m pytest tests/test_qc_strict.py -v
python -m pytest tests/test_preflight.py -v
python -m pytest tests/test_setup_scripts.py -v
python -m pytest tests/test_dockerfile.py -v
python -m pytest tests/test_agents_md_accuracy.py -v

# Environment
python setup/preflight.py

# Setup (VM)
bash setup/setup_env.sh
bash setup/run_generation.sh

# Generation knobs
DIPPER_LIMIT=4  DIPPER_SEED=42 bash setup/run_generation.sh   # smoke
DIPPER_LIMIT=0  DIPPER_SEED=42 bash setup/run_generation.sh   # full
```

### Task dependency order

Tasks 1 → 2 are strictly sequential (Task 2 consumes `control_codes` and
`unit_seed`). Task 3 depends on Task 2. Tasks 4, 5, 6, 7, and 8 are
independent of each other and can run in any order or in parallel. Task 9
depends on all of them and requires the VM.

### Scope reminder for the implementer

If a task seems to want a change in `detectors/` or `evaluation/`, that is out of
scope — those files must be untouched by this plan. `run_pilot.py`,
`calibrate_binoculars.py`, `evaluate.py`, `binoculars.py`, `roberta.py`, and
`gptzero.py` are read-only for this work. Do not add length truncation. Do not
add a 4-bit or fp16 quantization mode. Do not restructure the generation loop
for batching.