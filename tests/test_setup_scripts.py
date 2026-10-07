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


@posix_only
def test_setup_env_fails_fast_on_bad_python():
    result = subprocess.run(
        ["bash", str(SETUP_ENV), "--python", "/nonexistent/python"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


@posix_only
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
