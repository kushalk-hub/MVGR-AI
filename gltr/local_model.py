"""Local causal language-model loading for GLTR, with hard offline guarantees.

The loader never reaches the network unless ``allow_download`` is explicitly
true. Before importing ``transformers`` it sets the offline environment flags
(``HF_HUB_OFFLINE``, ``TRANSFORMERS_OFFLINE``) and disables telemetry, so an
accidental hub call fails fast instead of silently downloading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class GltrUnavailableError(RuntimeError):
    """Raised when GLTR is requested but no local model / runtime is available."""


def transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("transformers") is not None


def _set_offline_environment(allow_download: bool) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def local_model_available(model_name: str, revision: str | None = None) -> tuple[bool, str]:
    """Return ``(available, reason)`` for a locally cached model.

    Availability means: ``transformers`` is importable *and* the model's
    ``config.json`` is present in the local Hugging Face cache. No network is
    touched.
    """
    if not transformers_available():
        return False, "transformers is not installed (pip install -r requirements/gltr.txt)"
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception as exc:  # noqa: BLE001
        return False, f"huggingface_hub unavailable: {exc}"
    try:
        cached = try_to_load_from_cache(model_name, "config.json", revision=revision)
    except Exception as exc:  # noqa: BLE001 - offline / missing cache raises
        return False, f"model {model_name!r} is not in the local cache: {exc}"
    if not cached or cached in (None,):
        return False, f"model {model_name!r} is not in the local cache (no config.json)"
    return True, f"cached at {cached}"


@dataclass
class GltrConfig:
    model_name: str = "gpt2"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 4
    max_length: int = 512
    stride: int = 0
    rank_thresholds: tuple[int, ...] = (10, 100, 1000)
    top_p: float = 0.9
    seed: int = 42
    local_files_only: bool = True
    allow_download: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "GltrConfig":
        config = config or {}
        return cls(
            model_name=str(config.get("model_name", "gpt2")),
            revision=config.get("revision"),
            device=str(config.get("device", "auto")),
            dtype=str(config.get("dtype", "float32")),
            batch_size=int(config.get("batch_size", 4)),
            max_length=int(config.get("max_length", 512)),
            stride=int(config.get("stride", 0)),
            rank_thresholds=tuple(int(v) for v in config.get("rank_thresholds", (10, 100, 1000))),
            top_p=float(config.get("top_p", 0.9)),
            seed=int(config.get("seed", 42)),
            local_files_only=bool(config.get("local_files_only", True)),
            allow_download=bool(config.get("allow_download", False)),
        )


@dataclass
class GltrModel:
    """Lazily loaded causal LM + tokenizer with a deterministic forward pass."""

    config: GltrConfig = field(default_factory=GltrConfig)
    _model: Any = None
    _tokenizer: Any = None
    _device: str | None = None

    def availability(self) -> tuple[bool, str]:
        return local_model_available(self.config.model_name, self.config.revision)

    def _resolve_device(self, torch: Any) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> "GltrModel":
        available, reason = self.availability()
        if not available and not self.config.allow_download:
            raise GltrUnavailableError(
                f"GLTR model unavailable: {reason}. No download is attempted "
                f"(allow_download=False). Place the model in the local cache or set "
                f"allow_download=true to opt in explicitly."
            )
        _set_offline_environment(self.config.allow_download)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise GltrUnavailableError(
                "transformers/torch are not installed (pip install -r requirements/gltr.txt)"
            ) from exc

        torch.manual_seed(self.config.seed)
        self._device = self._resolve_device(torch)
        dtype = getattr(torch, self.config.dtype, torch.float32)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            local_files_only=self.config.local_files_only and not self.config.allow_download,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            local_files_only=self.config.local_files_only and not self.config.allow_download,
            torch_dtype=dtype,
        )
        self._model.to(self._device)
        self._model.eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "device": self._device,
            "dtype": self.config.dtype,
            "max_length": self.config.max_length,
            "local_files_only": self.config.local_files_only,
            "allow_download": self.config.allow_download,
            "loaded": self.loaded,
        }
        try:
            import transformers

            info["transformers_version"] = transformers.__version__
        except Exception:  # noqa: BLE001
            info["transformers_version"] = None
        if self._tokenizer is not None:
            info["tokenizer_name"] = getattr(self._tokenizer, "name_or_path", None)
            info["vocab_size"] = int(getattr(self._tokenizer, "vocab_size", 0) or 0)
        if self._model is not None:
            try:
                info["num_parameters"] = int(
                    sum(p.numel() for p in self._model.parameters())
                )
            except Exception:  # noqa: BLE001
                info["num_parameters"] = None
        return info

    def next_token_logits(self, input_ids: Any) -> Any:
        """Forward pass returning logits ``[batch, seq, vocab]`` (no grad)."""
        if self._model is None:
            raise GltrUnavailableError("GltrModel.load must be called before inference")
        import torch

        with torch.no_grad():
            output = self._model(input_ids=input_ids.to(self._device))
        return output.logits


__all__ = [
    "GltrConfig",
    "GltrModel",
    "GltrUnavailableError",
    "local_model_available",
    "transformers_available",
]
