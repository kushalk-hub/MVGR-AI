# PARAGON — Stage 1 Build Instructions (Mid-Semester Review)

> This file is written as a build prompt for an AI coding agent (e.g. opencode).
> Paste it in as the task/context and it should be able to scaffold and run Stage 1
> of the project. It also works as a standalone README for the repo.

---

## 1. Project Summary

**PARAGON** — Paraphrase-aware robust detection of AI-generated text using
graph-based structural evidence and hybrid detection signals.

**Core research question:**
How robust is AI-generated-text detection to controlled adversarial paraphrasing,
and can graph-based structural evidence provide complementary information that
improves robustness?

**Full pipeline (for context — not all built in Stage 1):**
1. Controlled paraphrase generation using **DIPPER** (creates attack conditions L0–L4).
2. Graph-based representations + GNNs (structural signal — later stage).
3. Deterministic graph-derived mathematical features (later stage).
4. **Existing AI-text detectors** — RoBERTa, GPTZero, Binoculars (**this is Stage 1**).
5. Hybrid/ensemble fusion of all signals (final stage).

**Task type:** Binary classification — Human-written vs AI-generated paragraph.
Paraphrasing is a *robustness condition* applied to AI text, not a separate class.

**Dataset:** [`andythetechnerd03/AI-human-text`](https://huggingface.co/datasets/andythetechnerd03/AI-human-text)
on HuggingFace. Target: ~5,000 human + 5,000 AI paragraphs (pilot can use a small subset, e.g. 100–300 pairs).

**Main paraphrase attack paper:** DIPPER — *"Paraphrasing evades detectors of
AI-generated text, but retrieval is an effective defense"* (NeurIPS 2023).
Repo: https://github.com/martiansideofthemoon/ai-detection-paraphrases

---

## 2. What "Stage 1" Means for This Review

Per the project roadmap, the mid-sem checkpoint covers:

**Stage 1 — Reproduce the attack conditions**
- Obtain and run DIPPER.
- Generate a small pilot set of paraphrases at 4 controlled levels (L1–L4), plus
  the original AI text as L0 (baseline, no paraphrase).
- Confirm the four paraphrase conditions produce increasing lexical/order diversity.
- Sanity-check semantic preservation (the paraphrase should still mean the same thing).

**Stage 2 — Establish existing-detector baselines**
- Run RoBERTa, GPTZero, and Binoculars on: human text, original AI text (L0), and
  each paraphrase level (L1–L4).
- Produce AUROC per detector per paraphrase level.
- Plot the **degradation curve**: detector performance vs paraphrase intensity.

**Deliverable for the review:** a pilot dataset + a scored CSV + one chart showing
detector AUROC dropping as paraphrase intensity increases. This is the empirical
evidence for the project's core motivation, before any graph/GNN work begins.

**Explicitly out of scope for this checkpoint:** graph construction, GNN training,
deterministic graph features, fusion/ensemble. Those are Stages 3–8.

---

## 3. Platform / Environment

| Purpose | Platform | Why |
|---|---|---|
| Writing & organizing code | **VS Code (local)** | Editing, folder structure, git — not for running heavy models |
| Version control | **GitHub repo** | Keeps history, needed before/after the review |
| Running the detectors (GPU) | **Kaggle Notebooks** | Free GPU quota (~30 hrs/week), enough for RoBERTa + a smaller Binoculars model pair — local laptop CPU cannot run Binoculars (needs two LLMs loaded at once) |
| Dataset + DIPPER checkpoint hosting | **HuggingFace Hub** | Source dataset and any model checkpoints load directly via `datasets`/`transformers` |

**Workflow:** write/scaffold code locally in VS Code → push to GitHub → clone the
repo into a Kaggle Notebook with GPU accelerator turned on → run the pilot there →
pull results/plots back into the repo.

---

## 4. Repository Structure to Build

```
paragon/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # downloaded HF dataset subset
│   └── pilot/                # pilot_dataset.csv (text, label, paraphrase_level)
├── paraphrase/
│   └── dipper_generate.py    # runs DIPPER to create L1-L4 from AI text (L0)
├── detectors/
│   ├── base.py                # BaseDetector abstract interface
│   ├── roberta.py             # supervised neural detector
│   ├── binoculars.py          # zero-shot statistical detector
│   ├── gptzero.py             # API-wrapped external detector
│   └── run_pilot.py           # scores all detectors across L0-L4, saves CSV
├── notebooks/
│   └── stage1_kaggle.ipynb    # the notebook actually run on Kaggle (GPU)
└── results/
    ├── pilot_scores.csv
    └── degradation_plot.png
```

---

## 5. Build Steps (in order)

1. **Set up repo locally in VS Code.**
   - `git init`, create the folder structure above, add `requirements.txt`
     (`transformers`, `torch`, `datasets`, `pandas`, `scikit-learn`, `matplotlib`).
   - Push to GitHub.

2. **Load a small subset of the dataset.**
   - Pull ~100–300 human + ~100–300 AI paragraphs from
     `andythetechnerd03/AI-human-text` via the `datasets` library.
   - Save as `data/raw/pilot_source.csv`.

3. **Generate DIPPER paraphrases (L1–L4).**
   - Clone/install DIPPER from the official repo.
   - Define 4 `(lexical, order)` diversity settings — exact values are an open
     research decision; start with something like low/medium/high/very-high
     diversity and document the actual values used.
   - For each AI paragraph, generate 4 paraphrased versions (L1–L4). Original
     AI text = L0.
   - Save to `data/pilot/pilot_dataset.csv` with columns:
     `text, label (human/ai), paraphrase_level (L0-L4), source_id`.

4. **Implement the detector interface (`detectors/`).**
   - `base.py`: abstract `BaseDetector` with a `.score(text) -> float` method
     (higher = more likely AI-generated).
   - `roberta.py`: load a RoBERTa-based AI-text detector checkpoint (e.g.
     `roberta-base-openai-detector`) via `transformers`, return the AI-class
     probability.
   - `binoculars.py`: implement per the official Binoculars repo
     (https://github.com/ahans30/Binoculars) using an observer/performer model
     pair. For the pilot, use the smallest viable model pair that fits Kaggle's
     GPU memory.
   - `gptzero.py`: wrap the GPTZero API (requires an API key — check current
     GPTZero docs for the endpoint/response schema, as these can change).

5. **Move execution to Kaggle.**
   - Create a Kaggle Notebook, enable GPU accelerator (T4 x2 or P100).
   - `!git clone <your-repo-url>`, `!pip install -r requirements.txt`.
   - Run `detectors/run_pilot.py` (or paste it into cells) to score every row
     of `pilot_dataset.csv` with each detector.
   - Save output to `results/pilot_scores.csv`.

6. **Evaluate and plot.**
   - For each detector, compute AUROC separately at each paraphrase level
     (L0, L1, L2, L3, L4) against the human-vs-AI label.
   - Plot AUROC (y-axis) vs paraphrase level (x-axis), one line per detector.
     This is the core "detectors degrade under paraphrasing" chart.
   - Save as `results/degradation_plot.png`.

7. **Write up findings for the review.**
   - Report: pilot dataset size, DIPPER settings used, per-detector AUROC at
     each level, and the degradation trend. Note any detector that failed to
     run and why (e.g. GPTZero API limits).

---

## 6. Notes / Constraints to Respect

- This is a **research project** — the exact DIPPER `(L, O)` values, and any
  detector checkpoint choices, are open decisions. Document whatever values
  are actually used; don't treat them as fixed truths.
- Do not build the graph/GNN branch, deterministic features, or fusion layer
  yet — that's explicitly out of scope for this checkpoint.
- Keep the pilot small (100–300 examples) — the goal is a working, reproducible
  pipeline and a believable degradation trend, not the full 5K+5K dataset.
- If Binoculars' full model pair doesn't fit on Kaggle's free GPU, substitute
  smaller causal LMs for the pilot and note this as a scaling limitation to
  revisit later.
