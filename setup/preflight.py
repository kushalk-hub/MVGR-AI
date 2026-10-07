"""Environment gate for DIPPER int8 generation.

Exits non-zero with an actionable message when the environment cannot run
the pipeline. This is deliberately the inverse of the rest of the repo,
where a missing input makes a script return exit code 0 and a broken
chain reads as a clean run.

Consumed by both setup/setup_env.sh and the Dockerfile so the two setup
paths cannot drift apart.
"""

import sys

# DIPPER XXL in int8 needs roughly 11GB; leave headroom for activations.
REQUIRED_VRAM_GB = 12.0

DIPPER_MODEL = "kalpeshk2011/dipper-paraphraser-xxl"
DIPPER_TOKENIZER = "google/t5-v1_1-xxl"


class Environment:
    """Everything preflight needs to know, as plain data.

    Kept as a plain object so check() can be tested without a GPU.
    """

    def __init__(
        self,
        torch_version,
        transformers_version,
        bitsandbytes_version,
        cuda_available,
        device_name,
        total_vram_gb,
        free_vram_gb,
        model_loads,
        loaded_vram_gb,
    ):
        self.torch_version = torch_version
        self.transformers_version = transformers_version
        self.bitsandbytes_version = bitsandbytes_version
        self.cuda_available = cuda_available
        self.device_name = device_name
        self.total_vram_gb = total_vram_gb
        self.free_vram_gb = free_vram_gb
        self.model_loads = model_loads
        self.loaded_vram_gb = loaded_vram_gb


def check(env):
    """Print a report and return 0 if generation can proceed, else non-zero."""
    print("=== PARAGON Stage 1 preflight ===")
    print(f"torch        {env.torch_version}")
    print(f"transformers {env.transformers_version}")
    print(f"bitsandbytes {env.bitsandbytes_version}")

    if not env.cuda_available:
        print("\nFAIL: CUDA is not available to torch.")
        print("  Fix: check `nvidia-smi` works, and that torch is a CUDA build.")
        print("       On Linux, a torch CPU wheel is the usual cause.")
        return 1

    print(f"device       {env.device_name}")
    print(f"VRAM         {env.free_vram_gb:.1f} GiB free / {env.total_vram_gb:.1f} GiB total")

    if env.total_vram_gb < REQUIRED_VRAM_GB:
        print(f"\nFAIL: {env.total_vram_gb:.1f} GiB VRAM is below the {REQUIRED_VRAM_GB:.1f} GiB")
        print("  DIPPER XXL needs for int8 loading.")
        print("  Fix: use a larger GPU, or confirm quantization_config is int8.")
        return 1

    if not env.model_loads:
        print("\nFAIL: DIPPER int8 model did not load.")
        print(f"  Fix: confirm bitsandbytes works on this GPU, then retry:")
        print(f"       python -c \"from transformers import BitsAndBytesConfig; "
              f"BitsAndBytesConfig(load_in_8bit=True)\"")
        return 1

    print(f"model loaded {env.loaded_vram_gb:.1f} GiB resident")
    print("\nOK: environment is ready for DIPPER int8 generation.")
    return 0


def probe():
    """Inspect the real environment and load DIPPER in int8."""
    import torch
    import transformers

    cuda_available = torch.cuda.is_available()
    device_name = ""
    total_vram_gb = 0.0
    free_vram_gb = 0.0

    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        device_name = properties.name
        total_vram_gb = properties.total_memory / (1024**3)
        free_vram_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)

    try:
        import bitsandbytes

        bnb_version = bitsandbytes.__version__
    except Exception as exc:
        bnb_version = f"MISSING ({type(exc).__name__})"

    model_loads = False
    loaded_vram_gb = 0.0
    if cuda_available and total_vram_gb >= REQUIRED_VRAM_GB:
        try:
            from transformers import (
                AutoTokenizer,
                BitsAndBytesConfig,
                T5ForConditionalGeneration,
            )

            tokenizer = AutoTokenizer.from_pretrained(DIPPER_TOKENIZER)
            model = T5ForConditionalGeneration.from_pretrained(
                DIPPER_MODEL,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                loaded_vram_gb = torch.cuda.memory_reserved(0) / (1024**3)
            model_loads = True
        except Exception as exc:
            print(f"\nModel load raised: {type(exc).__name__}: {exc}")

    return Environment(
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        bitsandbytes_version=bnb_version,
        cuda_available=cuda_available,
        device_name=device_name,
        total_vram_gb=total_vram_gb,
        free_vram_gb=free_vram_gb,
        model_loads=model_loads,
        loaded_vram_gb=loaded_vram_gb,
    )


if __name__ == "__main__":
    sys.exit(check(probe()))
