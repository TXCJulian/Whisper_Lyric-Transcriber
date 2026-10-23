# syntax=docker/dockerfile:1
#
# One Dockerfile, four images (GPU_BACKEND build arg):
#   nvidia  -- universal: modern env (cu130 torch + faster-whisper) and
#              legacy env (cu126 torch + openai-whisper) for Maxwell/Pascal/
#              Volta; docker/select-profile.sh picks one at start, CPU if no
#              GPU is visible
#   intel   -- torch xpu + openai-whisper
#   amd     -- torch ROCm + openai-whisper
#   cpu     -- torch cpu + faster-whisper
#
# No vendor base images (nvidia/cuda, oneAPI, ROCm): PyTorch's wheels bundle
# their own CUDA/oneAPI/ROCm runtime libraries, and the container runtime
# injects the host driver. A slim distro base keeps size and CVE count down.

ARG GPU_BACKEND=nvidia
ARG UV_VERSION=0.13.5
ARG PYTHON_VERSION=3.14

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ── Base images per backend ────────────────────────────────────────────────
FROM debian:trixie-20261020-slim AS base-nvidia
# The nvidia/cuda base image used to set these; without them the NVIDIA
# container runtime injects neither the driver's CUDA libraries nor
# nvidia-smi, and every GPU node silently falls back to CPU.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM debian:trixie-20261020-slim AS base-amd
FROM debian:trixie-20261020-slim AS base-cpu

# Intel needs the user-space GPU driver (Level-Zero + compute-runtime) inside
# the container. Debian's and Ubuntu 24.04's packaged versions are too old for
# Battlemage (Arc B580) -- an old compute-runtime crashes encoding GPU
# commands ("Abort was called ... command_encoder_xehp_and_later.inl").
# Ubuntu 26.04's own archive ships compute-runtime 26.x, so no PPA is needed.
FROM ubuntu:26.04 AS base-intel

# ── Build stages: Python + virtual environments ─────────────────────────────
FROM base-${GPU_BACKEND} AS builder
ARG PYTHON_VERSION
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install ${PYTHON_VERSION}
WORKDIR /build
COPY pyproject.toml uv.lock ./

# Primary environment: /opt/venv/<GPU_BACKEND>.
FROM builder AS venv-primary
ARG GPU_BACKEND
ARG PYTHON_VERSION
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv/${GPU_BACKEND} \
    uv sync --frozen --no-install-project --no-dev --python ${PYTHON_VERSION} --extra ${GPU_BACKEND}
# nvidia: record the oldest GPU architecture the cu130 torch wheel ships
# kernels for, so select-profile.sh's legacy cutoff follows the wheel instead
# of a hard-coded value. _cuda_getArchFlags() works without a GPU (unlike
# torch.cuda.get_arch_list()); it's private API, hence the fallback.
RUN if [ "$GPU_BACKEND" = "nvidia" ]; then \
        /opt/venv/nvidia/bin/python -c "import re, torch; \
archs = [int(a) for a in re.findall(r'sm_(\d+)', torch._C._cuda_getArchFlags())]; \
m = min(archs); print(f'{m // 10}.{m % 10}')" > /opt/venv/nvidia.min-compute-cap \
        || { echo "WARNING: could not read torch CUDA arch list, assuming 7.5" >&2; \
             echo 7.5 > /opt/venv/nvidia.min-compute-cap; }; \
        echo "nvidia min compute capability: $(cat /opt/venv/nvidia.min-compute-cap)"; \
    fi

# Second environment, nvidia only: /opt/venv/nvidia-legacy. Its own stage so
# BuildKit builds both environments in parallel and each lands in its own
# image layer. Other backends leave this empty.
FROM builder AS venv-secondary
ARG GPU_BACKEND
ARG PYTHON_VERSION
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /opt/venv \
    && if [ "$GPU_BACKEND" = "nvidia" ]; then \
        UV_PROJECT_ENVIRONMENT=/opt/venv/nvidia-legacy \
        uv sync --frozen --no-install-project --no-dev --python ${PYTHON_VERSION} --extra nvidia-legacy; \
    fi

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM base-${GPU_BACKEND} AS runtime
ARG GPU_BACKEND
USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && if [ "$GPU_BACKEND" = "intel" ]; then \
        apt-get install -y --no-install-recommends \
            libze-intel-gpu1 \
            libze1 \
            intel-opencl-icd; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. IDs are assigned by the system and re-mapped to
# PUID/PGID at container start.
#
# Ubuntu images ship a default "ubuntu" user at UID/GID 1000. entrypoint.sh
# force-remaps appuser onto PUID/PGID (1000 by default) via `usermod -o`,
# which would otherwise collide with that account -- setpriv --init-groups
# then resolves the shared UID back to "ubuntu" and loads its groups instead
# of appuser's, silently dropping GPU device group membership. Remove it so
# PUID=1000 stays unambiguous.
RUN userdel -r ubuntu 2>/dev/null; \
    groupadd appgroup \
    && useradd -g appgroup -M -s /bin/false appuser \
    && mkdir -p /app/models /app/jobs \
    && chown -R appuser:appgroup /app

COPY --from=builder /opt/python /opt/python
COPY --from=venv-primary /opt/venv/ /opt/venv/
COPY --from=venv-secondary /opt/venv/ /opt/venv/

ENV PATH=/opt/venv/${GPU_BACKEND}/bin:${PATH} \
    IMAGE_VARIANT=${GPU_BACKEND} \
    TORCH_HOME=/app/models/torch \
    HF_HOME=/app/models/huggingface \
    XDG_CACHE_HOME=/app/models/whisper \
    PYTHONUNBUFFERED=1
# openai-whisper's CUDA median-filter kernel (used for word-alignment) is
# JIT-compiled via Triton, which defaults its cache to ~/.triton -- HOME
# still points at /root (root's own env, untouched by entrypoint.sh's
# setpriv) when appuser actually runs the process, so that write fails.
ENV TRITON_CACHE_DIR=/app/models/triton
# Level-Zero's sysman subsystem, which torch.xpu uses for device
# enumeration, is disabled by default -- without this, torch.xpu sees
# zero devices and gpu_backend.py silently falls back to CPU.
ENV ZES_ENABLE_SYSMAN=1

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
COPY docker/select-profile.sh /usr/local/lib/select-profile.sh

WORKDIR /app
# App code last: it changes most often, so everything above stays cached.
COPY pyproject.toml ./
COPY app/ ./app/

# Build metadata, declared last so a new commit only touches metadata.
ARG APP_VERSION=
ARG APP_BUILD=dev
ARG APP_COMMIT=unknown
ARG BUILD_DATE=
ENV APP_VERSION=${APP_VERSION} \
    APP_BUILD=${APP_BUILD} \
    APP_COMMIT=${APP_COMMIT}
LABEL org.opencontainers.image.title="lyric-transcriber" \
      org.opencontainers.image.description="Song lyrics transcription: Demucs + Whisper + Genius (${GPU_BACKEND})" \
      org.opencontainers.image.source="https://github.com/TXCJulian/Whisper_Lyric-Transcriber" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${APP_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

EXPOSE 3334

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3334", "--no-access-log"]
