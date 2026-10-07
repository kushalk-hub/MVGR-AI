#!/usr/bin/env bash
# Run the Stage 1 paraphrase generation chain.
#
# preflight -> prepare_pilot -> dipper_generate -> qc --strict
#
# Every stage must succeed. set -euo pipefail means a broken link stops
# the chain instead of producing a partial dataset that looks complete.
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
