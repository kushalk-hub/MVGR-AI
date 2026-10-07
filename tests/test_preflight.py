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


def test_check_fails_on_the_vram_gate_itself(monkeypatch, capsys):
    fake = preflight.Environment(
        torch_version="2.5.1",
        transformers_version="4.44.0",
        bitsandbytes_version="0.43.0",
        cuda_available=True,
        device_name="NVIDIA GTX 1650",
        total_vram_gb=10.0,
        free_vram_gb=9.5,
        model_loads=False,
        loaded_vram_gb=0.0,
    )
    assert preflight.check(fake) != 0
    assert "below the" in capsys.readouterr().out
