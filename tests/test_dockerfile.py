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
