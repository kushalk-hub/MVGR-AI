"""Local causal language-model loading for GLTR, with hard offline guarantees.

The loader never reaches the network unless ``allow_download`` is explicitly
true. Before importing ``transformers`` it sets the offline environment flags
(``HF_HUB_OFFLINE``, ``TRANSFORMERS_OFFLINE``) and disables telemetry, so an
accidental hub call fails fast instead of silently downloading.

Model resolution order (offline-first):

1. an explicit local directory (``model_dir``, e.g. ``models/gltr/gpt2``) that
   contains ``config.json`` — this is what the experiments use after the one-off
   download script has populated it;
2. the local Hugging Face cache for ``model_name`` at ``revision`` (if present).

Only when ``allow_download`` is true *and* neither of the above exists does the
loader touch the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.paths import REPO_ROOT

DEFAULT_MODEL_DIR = "models/gltr/gpt2"


class GltrUnavailableError(RuntimeError):
    """Raised when GLTR is requested but no local model / runtime is available."""


def transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("transformers") is not None


def torch_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _set_offline_environment(allow_download: bool) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_model_dir(model_dir: str | os.PathLike[str] | None) -> Path | None:
    """Resolve ``model_dir`` relative to the repository root (unless absolute)."""
    if not model_dir:
        return None
    path = Path(model_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def local_dir_ready(model_dir: str | os.PathLike[str] | None) -> tuple[bool, str]:
    """Return ``(ready, reason)`` for an explicit local model directory."""
    path = resolve_model_dir(model_dir)
    if path is None:
        return False, "no model_dir configured"
    if not path.exists():
        return False, f"local model directory not found: {path}"
    if not (path / "config.json").exists():
        return False, f"local model directory missing config.json: {path}"
    return True, f"local model at {path}"


def local_model_available(
    model_name: str,
    revision: str | None = None,
    model_dir: str | os.PathLike[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(available, reason)`` for a locally available model.

    Availability means: an explicit local directory is ready, **or**
    ``transformers`` is importable *and* the model's ``config.json`` is present
    in the local Hugging Face cache. No network is touched.
    """
    if model_dir:
        ready, reason = local_dir_ready(model_dir)
        if ready:
            return True, reason
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


def _load_pretrained(cls: Any, source: str, *, revision: str | None, local_files_only: bool,
                     dtype: Any) -> Any:
    """``from_pretrained`` tolerant of the ``dtype`` / ``torch_dtype`` rename."""
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision:
        kwargs["revision"] = revision
    try:
        return cls.from_pretrained(source, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(source, torch_dtype=dtype, **kwargs)


@dataclass
class GltrConfig:
    model_name: str = "gpt2"
    # Pinned revision. ``gpt2`` main as of the pinned download; overridable.
    revision: str | None = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    model_dir: str | None = DEFAULT_MODEL_DIR
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 4
    max_length: int = 256
    stride: int = 128
    rank_thresholds: tuple[int, ...] = (10, 100, 1000)
    top_p: float = 0.9
    seed: int = 42
    local_files_only: bool = True
    allow_download: bool = False
    eval_mode: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "GltrConfig":
        config = config or {}
        return cls(
            model_name=str(config.get("model_name", "gpt2")),
            revision=config.get("revision", cls.revision),
            model_dir=config.get("model_dir", DEFAULT_MODEL_DIR),
            device=str(config.get("device", "cpu")),
            dtype=str(config.get("dtype", "float32")),
            batch_size=int(config.get("batch_size", 4)),
            max_length=int(config.get("max_length", 256)),
            stride=int(config.get("stride", 128)),
            rank_thresholds=tuple(int(v) for v in config.get("rank_thresholds", (10, 100, 1000))),
            top_p=float(config.get("top_p", 0.9)),
            seed=int(config.get("seed", 42)),
            local_files_only=bool(config.get("local_files_only", True)),
            allow_download=bool(config.get("allow_download", False)),
            eval_mode=bool(config.get("eval_mode", True)),
        )


@dataclass
class GltrModel:
    """Lazily loaded causal LM + tokenizer with a deterministic forward pass.

    Exactly one model instance is held by the analyzer; the pipeline never loads
    a second copy.
    """

    config: GltrConfig = field(default_factory=GltrConfig)
    _model: Any = None
    _tokenizer: Any = None
    _device: str | None = None
    _source: str | None = None
    _source_kind: str | None = None

    def availability(self) -> tuple[bool, str]:
        return local_model_available(
            self.config.model_name, self.config.revision, self.config.model_dir
        )

    def _resolve_device(self, torch: Any) -> str:
        if self.config.device and self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_source(self) -> tuple[str, str, str | None]:
        """Return ``(source, kind, revision)`` for ``from_pretrained``."""
        ready, _ = local_dir_ready(self.config.model_dir)
        if ready:
            return str(resolve_model_dir(self.config.model_dir)), "local_dir", None
        return self.config.model_name, "hub_cache", self.config.revision

    def load(self) -> "GltrModel":
        available, reason = self.availability()
        if not available and not self.config.allow_download:
            raise GltrUnavailableError(
                f"GLTR model unavailable: {reason}. No download is attempted "
                f"(allow_download=False). Run scripts/download_gltr_model.py once to "
                f"populate {self.config.model_dir}, or set allow_download=true to opt in."
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
        source, kind, revision = self._resolve_source()
        local_only = self.config.local_files_only and not self.config.allow_download
        # A local directory is always offline; revision only applies to the hub.
        self._source, self._source_kind = source, kind

        self._tokenizer = _load_pretrained(
            AutoTokenizer, source, revision=revision, local_files_only=local_only, dtype=None
        )
        if getattr(self._tokenizer, "pad_token", None) is None:
            # GPT-2 has no pad token; reuse EOS so batched padding is well defined.
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # We never truncate at tokenization time: long documents are split into
        # ``max_length`` windows before the forward pass. Raise the tokenizer's
        # advisory limit so it does not warn about inputs longer than the
        # reference model's 1024-position window (which we never exceed).
        model_max = getattr(self._tokenizer, "model_max_length", 0) or 0
        if 0 < model_max < 10**12:
            self._tokenizer.model_max_length = 10**12
        self._model = _load_pretrained(
            AutoModelForCausalLM, source, revision=revision, local_files_only=local_only, dtype=dtype
        )
        self._model.to(self._device)
        if self.config.eval_mode:
            self._model.eval()
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "model_dir": str(resolve_model_dir(self.config.model_dir)) if self.config.model_dir else None,
            "source": self._source,
            "source_kind": self._source_kind,
            "device": self._device,
            "dtype": self.config.dtype,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "stride": self.config.stride,
            "eval_mode": self.config.eval_mode,
            "local_files_only": self.config.local_files_only,
            "allow_download": self.config.allow_download,
            "loaded": self.loaded,
        }
        try:
            import transformers

            info["transformers_version"] = transformers.__version__
        except Exception:  # noqa: BLE001
            info["transformers_version"] = None
        try:
            import torch

            info["torch_version"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            info["torch_version"] = None
            info["cuda_available"] = None
        if self._tokenizer is not None:
            info["tokenizer_name"] = getattr(self._tokenizer, "name_or_path", None)
            info["vocab_size"] = int(getattr(self._tokenizer, "vocab_size", 0) or 0)
        if self._model is not None:
            try:
                info["num_parameters"] = int(sum(p.numel() for p in self._model.parameters()))
            except Exception:  # noqa: BLE001
                info["num_parameters"] = None
        return info

    def forward(self, input_ids: Any, attention_mask: Any = None) -> Any:
        """Batched forward pass returning logits ``[batch, seq, vocab]``.

        Runs under ``torch.inference_mode()`` with no gradient tracking.
        """
        if self._model is None:
            raise GltrUnavailableError("GltrModel.load must be called before inference")
        import torch

        device = self._device
        ids = input_ids.to(device)
        mask = attention_mask.to(device) if attention_mask is not None else None
        with torch.inference_mode():
            output = self._model(input_ids=ids, attention_mask=mask)
        return output.logits

    def next_token_logits(self, input_ids: Any) -> Any:
        """Backwards-compatible single-batch forward pass (no grad)."""
        return self.forward(input_ids)


__all__ = [
    "DEFAULT_MODEL_DIR",
    "GltrConfig",
    "GltrModel",
    "GltrUnavailableError",
    "local_dir_ready",
    "local_model_available",
    "resolve_model_dir",
    "torch_available",
    "transformers_available",
]
