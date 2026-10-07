# Container fallback for the native setup path in setup/setup_env.sh.
# Both consume requirements.txt and setup/preflight.py, so the two routes
# install the same dependency set and pass the same environment gate.
#
# Build:
#   docker build -t paragon-stage1 .
#   docker run --gpus all --rm \
#     -e DIPPER_LIMIT=0 -e DIPPER_SEED=42 \
#     -v "$PWD/data:/work/paragon/data" \
#     paragon-stage1
#
# The CUDA image tag must match the host driver. Check `nvcc --version` and
# `nvidia-smi` on the VM; override at build time if needed:
#   docker build --build-arg CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04 .

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    NLTK_DATA=/opt/nltk-data

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/paragon

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Sentence splitting needs punkt at generation time.
RUN python3 -c "\
import nltk; \
[nltk.download(name, quiet=True) for name in ('punkt', 'punkt_tab')]"

COPY . .

RUN chmod +x setup/run_generation.sh

ENTRYPOINT ["setup/run_generation.sh"]
