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
