"""Test-set access lock.

The test split is not a development split. Every access is logged per
experiment, and a second access requires an explicit override that is recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.contracts.validators import utc_now
from shared.paths import REPO_ROOT, evaluation_config, resolve


class TestLockError(RuntimeError):
    pass


def _lock_path() -> Path:
    config = evaluation_config()
    return resolve(config["evaluation"]["test_set_lock"]["path"])


def request_test_access(
    *,
    experiment_id: str,
    purpose: str,
    override: bool = False,
) -> Path:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: list[dict] = []
    if path.exists():
        previous = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    already = [entry for entry in previous if entry.get("experiment_id") == experiment_id]
    if already and not override:
        raise TestLockError(
            f"experiment {experiment_id} already accessed the test split at "
            f"{already[-1]['accessed_at']}. Pass --override-test-lock with a reason to record a repeat access."
        )
    entry = {
        "experiment_id": experiment_id,
        "accessed_at": utc_now(),
        "purpose": purpose,
        "override": bool(override),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def test_access_log() -> list[dict]:
    path = _lock_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
