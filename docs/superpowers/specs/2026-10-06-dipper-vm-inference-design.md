# DIPPER Inference Pipeline for a 24 GB VM — Design

**Date:** 2026-10-06
**Scope:** Stage 1 — paraphrase generation only. No detector or scoring changes.

## Goal

Generate the L0–L4 paraphrase conditions for 250 AI sources on a 24 GB VRAM VM,
reproducibly, with a setup path that works via two interchangeable routes
(native venv, Docker) and a preflight gate that refuses to start generation on a
broken environment.

Out of scope: detector scoring, evaluation, AUROC computation, length matching.
Equal-length truncation is a future evaluation-stage requirement, not a
generation requirement.

## Background: the similarity/diversity inversion

DIPPER's two control knobs exist on **opposite scales** between the paper and the
pretrained model.

- The **paper** reports **diversity**. `0` = keep the words, `100` = rewrite everything.
- The **model** reads **similarity**. `100` = stay maximally faithful, `0` = fidelity irrelevant.

So diversity `X` must be fed as control code `100 - X`. The DIPPER repository
states this directly, and its own `paraphrase.py` iterates
`(40, 40), (40, 100), (60, 100), (80, 100)` — which as similarity codes means
diversity 60/60, 60/0, 40/0, 20/0.

| Diversity | Control code |
|---:|---:|
| 0 | 100 |
| 20 | 80 |
| 40 | 60 |
| 60 | 40 |
| 80 | 20 |
| 100 | 0 |

Valid diversity values are multiples of 20 in `{0, 20, 40, 60, 80, 100}`.

### The current defect

`configs/dipper_levels.json` contains the paper's **diversity** values
(`lex: 20, 40, 60`), but `paraphrase/dipper_generate.py:83` interpolates them
into the prompt raw, with no inversion:

```python
prompt = f"lexical = {lex}, order = {order} {prefix} <sent> {window} </sent>"
```

No inversion exists anywhere in the repo. The model therefore reads diversity
values as similarity, inverting the entire ladder: L1 (`lex=20`) is read as
similarity 20, i.e. **diversity 80**, while L4 (`lex=60, order=60`) is read as
**diversity 40/40**. The attack currently runs hardest-to-easiest.

`paraphrase/qc.py` measures mean Jaccard overlap against L0, so this defect is
observable: overlap currently *rises* as levels increase.

## Design

### 1. Level ladder

`configs/dipper_levels.json` holds **diversity** values — the intuitive direction.
Monotone increasing on both axes, entirely inside the paper's validated grid
(`L ∈ {20,40,60} × O ∈ {0,60}`):

| Level | Diversity L | Diversity O | Control code (lex, order) |
|---|---:|---:|---|
| L1 | 20 | 0 | 80, 100 |
| L2 | 40 | 20 | 60, 80 |
| L3 | 60 | 40 | 40, 60 |
| L4 | 60 | 60 | 40, 40 |

L0 is the original AI text and takes no config entry.

Order diversity varies at every level rather than staying pinned at 0. This is
deliberate. Order diversity reorders sentences and so perturbs discourse
structure, whereas lexical substitution mostly rewrites words in place and
leaves a dependency or co-occurrence graph largely intact. For a project whose
research question is whether *structural* evidence survives paraphrasing, a
purely lexical ladder would under-test the signal of interest.

The ladder stops at L60/O60 rather than reaching for higher diversity because the
paper never tested beyond it, and diversity 80+ risks semantic drift that would
invalidate the attack as meaning-preserving. `paraphrase/qc.py --strict` is the
gate that would catch this. Higher settings remain available as a future
recorded experiment.

### 2. Control-code conversion

`paraphrase/dipper_generate.py` gains a single helper that owns the conversion,
so the two scales cannot drift apart:

```python
def control_codes(lex_div, order_div):
    return int(100 - lex_div), int(100 - order_div)
```

`paraphrase()` calls it and interpolates the returned pair. A unit test pins the
paper's documented mapping: `control_codes(60, 60) == (40, 40)`.

### 3. Deterministic seeding

`generate(do_sample=True)` currently runs with no seed anywhere, so re-runs do
not reproduce a single-pass run and no dataset can be regenerated. This
conflicts with the provenance requirement in `PARAGON_MASTER_PROJECT_NEW.md` §32.

- A `--seed` CLI flag is added, default 42, recorded in the manifest.
- Seed is derived **per `(source_id, level)` unit** and applied immediately before
  that unit's `paraphrase()` call, so each unit is independent of traversal or
  resume order.
- Derivation uses `zlib.crc32(f"{source_id}|{level}")` mixed with the run seed,
  **not** Python's built-in `hash()`, which is salted per process by
  `PYTHONHASHSEED` and would silently break reproducibility across runs.

### 4. Provenance manifest

After generation, write `data/pilot/generation_manifest.json` containing:

`seed`, `model`, `tokenizer`, `load_in_8bit`, `levels`, `top_p`, `max_length`,
`sent_interval`, `torch_version`, `transformers_version`, `bitsandbytes_version`,
`timestamp`.

A sidecar file rather than a new CSV column, so the column contract read by
`detectors/run_pilot.py`, `evaluation/evaluate.py`, and `thin_dataset.py` is
untouched.

### 5. Setup components

Four new files, each single-purpose:

| File | Purpose |
|---|---|
| `setup/preflight.py` | Verify environment; report VRAM; exit non-zero on failure |
| `setup/setup_env.sh` | Build `.venv`, install deps, punkt, run preflight, write lockfile |
| `setup/run_generation.sh` | preflight → prepare → generate → qc |
| `Dockerfile` | CUDA base image, same entrypoint |

Both setup routes consume the *same* two files — `requirements.txt` and
`setup/preflight.py` — so they cannot drift apart. They are interchangeable: if
the native path fails, the container path is a genuine fallback rather than a
second thing that happens to work.

#### `setup/preflight.py`

Checks in order, exiting non-zero with an actionable fix-it message on failure:

1. `torch` imports; `torch.cuda.is_available()` is true
2. Device name, total VRAM, free VRAM
3. `bitsandbytes` imports; version reported
4. Tokenizer and int8 model actually load
5. Post-load VRAM reported

This is deliberately the inverse of the repo's current failure mode, where
`calibrate_binoculars.py` and `evaluate.py` `return` with exit code 0 on missing
input so a broken chain reads as a clean run. The preflight refuses to let
generation start on a bad environment.

#### `setup/setup_env.sh`

Creates `.venv`, installs `requirements.txt`, downloads NLTK punkt, runs
preflight, then writes `requirements-lock.txt` via `pip freeze` for
reproducibility.

If `pip install` fails on the `bitsandbytes>=0.41,<0.46` pin, the script prints
the resolver error and the available loosening options rather than silently
retrying with different versions.

#### `Dockerfile`

Takes the CUDA base image as a build ARG, defaulting to
`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, since the VM's CUDA version is
not yet known. Must be set to match `nvcc --version` on the target VM. Entry
point delegates to `setup/run_generation.sh`.

### 6. Data flow

Unchanged from current behaviour, so no downstream edits are required:

```
data/prepare_pilot.py        -> data/raw/pilot_source.csv
paraphrase/dipper_generate.py -> data/pilot/pilot_dataset.csv   (+ generation_manifest.json)
paraphrase/qc.py             -> data/qc_overlap.csv
```

`detectors/run_pilot.py`, `detectors/calibrate_binoculars.py`, and
`evaluation/evaluate.py` continue to read the same columns. Existing append-only
resume and the `evaluate.py` missing-level guard both keep working.

Traversal stays **column-first** (source-major): all four levels of one source are
generated consecutively. Batching would require restructuring the `prefix`
accumulation loop into level-major window-interleaved batches, which risks
discourse continuity and is not worth roughly 2x throughput at 250 sources.

### 7. Error handling and QC

- Per-unit generation failures still land in `quality_status` as `empty` or
  `error`, unchanged.
- `dipper_generate.py` adds an end-of-run tally of `ok` / `empty` / `error`
  counts, so "ran clean" is distinguishable from "assumed it ran clean".
- `paraphrase/qc.py` gains `--strict`: exit non-zero when mean Jaccard overlap
  against L0 fails to be **non-increasing** across L0→L4. A violation is any level
  whose overlap exceeds that of the preceding level by more than a small
  floating-point tolerance (`1e-9`); equal values pass. L4 at `O=60` is the row
  most likely to trip this.

### 8. Verification

Four checks, cheapest first:

1. `configs/dipper_levels.json` parses and contains four levels.
2. Unit test: `control_codes(60, 60) == (40, 40)`.
3. Smoke run at `--limit 4` produces all four levels and QC passes.
4. Determinism test: generate the same two sources twice with the same seed;
   the outputs must be byte-identical.

Checks 3 and 4 require the VM's GPU. Checks 1 and 2 run anywhere and should pass
in CI or locally.

## Out of scope

- Detector scoring, AUROC, evaluation-stage changes.
- **Equal-length truncation.** The paper truncates AI text, paraphrase, and human
  text all to the shortest word count before scoring, to stop length becoming a
  detector cue. That is an evaluation-stage control and would mean destroying
  generated text. Recorded as a known limitation to address when the detection
  experiment is built. `qc.py` already records `mean_words` per level, so length
  drift is visible in QC output without modifying any text.
- Graph construction, GNNs, deterministic graph features, fusion. These remain
  Stages 3–8 per the project's scope gate.
- `thin_dataset.py`'s stray `PYEOF` token, which is unrelated to this work.

## Documentation update

`AGENTS.md` requires three corrections:

- The "L1–L3 are purely lexical (order 0)" line is false under the new ladder.
- Record the similarity/diversity mapping as a trap: config holds diversity,
  the model reads similarity.
- Add the setup commands and the seeding/provenance behaviour.