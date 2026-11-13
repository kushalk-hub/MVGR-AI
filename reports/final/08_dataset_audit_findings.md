# 20. Dataset audit findings

Every number below is computed from the dataset by `mvgr.py audit`, never copied
from a report or hard-coded. The audit runs read-only against the canonical
copy and fails closed if provenance cannot be verified.

* **File:** `data/processed/mvgr_ai_combined.parquet`
* **SHA-256:** `9c824a635ed66bc02421c572381910aacb0198209a0e1f8e2d9111728a2eb917`
* **Provenance:** byte-identical to the immutable upstream
  `datasets/processed/mvgr_ai_combined.parquet` (`match: true`)
* **Rows:** 128,709 · **Columns:** 12
* **Dataset version:** `mvgr-ai/2026-11-13`
* **Gate status:** **FAIL** (1 failing check, 3 warnings, 6 passes)

Reproduce with:

```powershell
python mvgr.py audit
```

Outputs land in `reports/dataset_audit/` (`inventory.json`, `leakage_audit.json`,
`confounding.json`, `audit_summary.json`, `family_summary.json`, plus CSV
reconciliations).

---

## Composition

### Labels

| Label | Meaning | Rows |
|---|---|---|
| 0 | human_authored | 36,349 |
| 1 | ai_authored | 92,360 |

No null labels. Class imbalance (AI:human) is 2.54.

### Sources

| Source | Rows | Human | AI |
|---|---|---|---|
| PADBen | 80,651 | 31,961 | 48,690 |
| BiScope | 48,058 | 4,388 | 43,670 |

### Tasks

| Source / task | Rows | Human | AI |
|---|---|---|---|
| BiScope | 48,058 | 4,388 | 43,670 |
| PADBen sentence-pair-task1 | 24,730 | 12,312 | 12,418 |
| PADBen sentence-pair-task2 | 24,057 | 11,521 | 12,536 |
| PADBen sentence-pair-task5 | 15,929 | 3,437 | 12,492 |
| PADBen exhaustive-task1 | 7,457 | 3,644 | 3,813 |
| PADBen exhaustive-task2 | 4,739 | 1,047 | 3,692 |
| PADBen exhaustive-task5 | 3,739 | 0 | 3,739 |

### Conditions (`text_type`)

| Value | Rows |
|---|---|
| paraphrased | 53,972 |
| original | 29,710 |
| unknown | 28,796 |
| deep_paraphrased | 16,231 |

`paraphrase_status`: True 70,203, False 29,710, missing 28,796 (the missing
values are exactly the `text_type=unknown` rows).

### Existing splits

| Split | Rows | Human | AI |
|---|---|---|---|
| train | 108,211 | 33,198 | 75,013 |
| validation | 11,861 | 2,061 | 9,800 |
| test | 8,637 | 1,090 | 7,547 |

The shipped split is heavily skewed (train 84%) and is **not** family-verified;
it is the reason the manifest stage re-splits from scratch.

---

## Text and lineage

* Length: mean 738, median 137, p25 99, p75 562, max 40,512, min 10.
* 281 texts are shorter than 40 characters (a warning, not a failure).
* No empty rows and **no duplicate normalized texts** — every normalized text is
  unique (128,709 unique of 128,709).
* Lineage: 102,530 unique `group_id`s, no rows missing a group id. Group sizes
  are 1 (76,351 rows) and 2 (26,179 rows) — i.e. pair-level only.

The pair-level structure comes entirely from PADBen. BiScope's `group_id` is an
exact-text hash, so it groups only identical texts and carries no paraphrase-family
link.

---

## Gate results

| Check | Status | Detail |
|---|---|---|
| unique_sample_ids | pass | 128,709 unique ids |
| text_present | warn | 281 texts shorter than 40 chars |
| label_domain | pass | binary 0/1 only |
| family_disjoint | pass | 102,530 families, none spanning splits |
| text_hash_disjoint | pass | no normalized text crosses a split |
| split_ratios | warn | train 0.841 (target 0.70), val 0.092, test 0.067 |
| label_coverage | pass | all 7 task/label test cells ≥ 30 rows |
| class_imbalance | pass | AI:human ratio 2.54 |
| **lineage_resolution** | **FAIL** | 48,058 BiScope rows rely on exact-text-only or unresolved lineage |
| condition_confounding | warn | 16 confounded cells covering 126,492 rows |

### Why `lineage_resolution` fails

BiScope supplies original and paraphrased rows, but the paraphrased flag is a row
property, not a family link: the paraphrased counterpart of a given original text
is not identified by the export. Family lineage therefore cannot be recovered,
and a paraphrase pair could in principle be split across train and test. The
check fails closed so that pooled paraphrase-family claims are unsupported
without an explicit, recorded waiver.

This is a **data limitation**, not a code defect. PADBen, whose pairs are linked
by `group_id`, does not have this problem.

---

## Confounding findings (16 cells)

Perfect confounding is present in both directions. Two examples:

* **Condition predicts label.** BiScope `text_type=paraphrased` → 21,785 rows,
  all label = 1. PADBen `sentence-pair-task5` / `deep_paraphrased` → 12,492 rows,
  all label = 1; the same task's `original` rows (3,437) are all label = 0. In
  these cells the condition and the label move together, so no paraphrase effect
  can be estimated there.
* **Label predicts condition.** PADBen `sentence-pair-task1` label 0 (12,312) and
  label 1 (12,418) both occur only as `paraphrased`. BiScope label 0 (4,388)
  occurs only as `original`.

Consequence, as recorded by the audit: these cells are reported as **confounded**,
never as a paraphrase effect. Pooled condition claims are invalid; only
condition-controlled, per-task comparisons are reportable.

---

## Label–length confound

Human text has a median length of 222 characters; AI text has a median of 816.
Length alone is a strong predictor of the label, so the evaluation protocol
requires a length-only baseline and reports results across length bins. A model
that does not beat the length-only baseline has demonstrated nothing about
authorship.

---

## Warnings from the inventory

1. **No generator metadata is available for any row**, so model-specific claims
   (e.g. "this is GPT-4 text") are unsupported.
2. **Some rows have `text_type=unknown`**, so condition-specific claims must
   exclude them.

---

## Candidate split manifests

Three candidates were built. None is approved.

| Manifest id | Mode | Scope | train / val / test | Gate | Failing |
|---|---|---|---|---|---|
| `man_f3aea517be133c1421c1` | regenerate | pooled | 90,099 / 19,306 / 19,304 | fail | `lineage_resolution` |
| `man_a4b3d451ff3129f4f95a` | existing | pooled | 108,211 / 11,861 / 8,637 | fail | `lineage_resolution` |
| `man_20ddf272c292efbc8c8c` | regenerate | source_scoped (PADBen) | 56,457 / 12,098 / 12,096 | **pass** | — |

* The regenerated pooled manifest fixes the split-ratio warning and is
  family-disjoint, but still fails `lineage_resolution` because BiScope is in
  scope.
* The existing-split pooled manifest also fails `lineage_resolution` and
  additionally warns on `split_ratios`.
* The PADBen source-scoped manifest **passes** all gates.

### Approval path

```powershell
# Option A — PADBen source-scoped (passes; no waiver needed)
python mvgr.py approve-manifest --manifest data/manifests/candidate/man_20ddf272c292efbc8c8c/manifest.json \
  --approver "<name>" --statement "PADBen source-scoped split approved; no cross-source claims."

# Option B — pooled (needs an explicit waiver for BiScope lineage)
python mvgr.py approve-manifest --manifest data/manifests/candidate/man_f3aea517be133c1421c1/manifest.json \
  --approver "<name>" --statement "WAIVER: BiScope paraphrase families are unrecoverable; pooled paraphrase-family claims are limited to <scope>."
```

An approval statement for a failing gate must begin with `WAIVER:`. This is a
research decision for the contributors, not an engineering step.

---

## Open questions for the contributors

1. **Scope** — PADBen source-scoped, or pooled with a documented waiver?
2. **PADBen task semantics** — confirm each task's meaning against the dataset
   card before pooled training. The audit deliberately never infers task
   semantics from the label distribution.
3. **`text_type=unknown`** — decide how these 28,796 rows are handled in
   condition-specific analyses (currently excluded).
4. **Short texts** — decide whether the 281 texts under 40 characters are kept
   (they are kept today, with a warning).
