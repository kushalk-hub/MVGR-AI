# AGENTS.md

PARAGON — research project on paraphrase-robust AI-text detection. Only **Stage 1**
(the mid-semester checkpoint) is implemented: it builds the DIPPER attack
conditions and scores three existing detectors across paraphrase levels L0–L4.

## Scope gate

Stage 1 ships the `detectors/` branch and the attack pipeline. Graph construction,
GNNs, deterministic graph features, and fusion/ensembles are Stages 3–8 and stay
unbuilt until the research gates pass. This is a research project: record the
DIPPER `(L, O)` values and detector checkpoints actually used rather than
treating any example in the docs as fixed.

**Design decisions and open questions** → `PARAGON_MASTER_PROJECT_NEW.md`
(§37 locked, §38 open, §23 leakage controls, §30 gates). Stage 1 scope and build
order → `README.md`.

## Where execution happens

The pipeline targets **Kaggle GPU** (T4 x2): Binoculars loads two fp16 LLMs and
DIPPER loads T5-XXL in 8-bit (~11GB). Local machines author code; running the
detectors is the Kaggle notebook's job. `notebooks/stage1_kaggle.ipynb` clones
the repo, pip-installs, and runs each script as a cell — outputs land in
`/kaggle/working/paragon/results/` and must be downloaded back into the repo.

Every script takes cwd-relative paths (`data/pilot/pilot_dataset.csv`,
`results/pilot_scores.csv`), so **run from the repo root**. There is no package
install: `detectors/` has no `__init__.py` and `run_pilot.py` puts the repo root
on `sys.path` itself.

## Pipeline order

Each step consumes the previous step's output:

```bash
python data/prepare_pilot.py                      # HF stream -> data/raw/pilot_source.csv (250 human + 250 ai)
python paraphrase/dipper_generate.py --limit 20   # -> data/pilot/pilot_dataset.csv, appends L1-L4 (--limit 0 = all)
python paraphrase/qc.py                            # -> data/qc_overlap.csv, mean Jaccard overlap vs L0
python detectors/run_pilot.py --detectors roberta binoculars gptzero --gptzero-limit 80
python detectors/calibrate_binoculars.py           # -> results/pilot_scores_eval.csv + binoculars_threshold.json
python evaluation/evaluate.py                      # -> results/degradation_plot.png, results_table.csv, summary.md
```

**Check artifacts, not exit codes.** `calibrate_binoculars.py` and
`evaluate.py` `return` (exit 0) when their input file is missing, so a broken
chain looks like a clean run. The one hard guard is `evaluate.py`'s missing-level
check, which `sys.exit(1)s` when any of L0–L4 is absent.

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

## Score direction

Higher score = more likely AI, **except** `binoculars_raw`. `run_pilot.py` stores
Binoculars' un-negated `compute_score` output, where **lower = AI**.
`BinocularsDetector.score()` negates it; `evaluate.py` flips it back via
`DET_DIRECTION`. Mixing the two inverts AUROC silently.

## Adding or changing a detector

Detector column names are hardcoded at both ends: `run_pilot.py` writes them,
`evaluation/evaluate.py` reads `roberta_score` / `binoculars_raw` /
`gptzero_score`. A new detector must be registered in `DET_NAMES`,
`DET_DIRECTION`, `DET_LABELS`, and `DET_COLORS` (the last two zip together, so
lengths must match).

`evaluate.py` skips any detector whose column is absent from the CSV, so a
detector that produced nothing yields a completed run with a missing chart line
rather than an error. Confirm the score column exists before trusting a report.

GPTZero has two modes via `make_gptzero()`: the paid API when `GPTZERO_API_KEY`
or `SECRET_GPTZERO_API_KEY` is set, otherwise a local `gpt2-medium`
log-perplexity proxy. `--gptzero-limit` caps only the API path; local mode scores
every row. Failed API calls become `NaN` and are masked out during evaluation.
The proxy's docstring claims a burstiness term the code does not compute.

## Data contract

`sample_id` is `{source_id}_L0`..`_L4` for AI text and `{source_id}_human` for
human text. Humans carry `paraphrase_level="human"` and are excluded from the
L-level tables; `evaluate.py` pairs each level against the shared human set.

`label`, `paraphrase_level`, `lex_control`, `order_control`, and
`parent_sample_id` are **metadata, never classifier features** (§23.4). All
paraphrase descendants of one `source_id` belong to the same split.

`data/raw/*.csv`, `data/pilot/*.csv`, and `results/*.csv|*.png` are gitignored —
only `.gitkeep` is tracked. A fresh clone has no data and no results, so any
pipeline step must be run end-to-end before evaluation means anything.

## Known defects

- `thin_dataset.py:69` ends with a bare `PYEOF` token → `NameError` after the
  thinned CSV is written (exit code 1 despite succeeding).
