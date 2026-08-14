ARG GPU_BACKEND=nvidia

# ── Base images per backend ────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base-nvidia
FROM intel/oneapi-basekit:2025.3.1-0-devel-ubuntu24.04 AS base-intel
FROM rocm/rocm-terminal:6.4 AS base-amd
FROM python:3.11-slim AS base-cpu

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM base-${GPU_BACKEND} AS runtime

ARG GPU_BACKEND=nvidia
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV GPU_BACKEND=${GPU_BACKEND}

# Install Python 3.11 + system deps (skip for cpu base which already has Python)
#
# Intel driver repo (see below) is added here too, before the python3 ->
# 3.11 symlink swap: add-apt-repository's apt_pkg module is only built
# for the base image's original Python, so calling it after the swap
# fails with "ModuleNotFoundError: No module named 'apt_pkg'".
RUN if [ "$GPU_BACKEND" != "cpu" ]; then \
        apt-get update && apt-get install -y \
            software-properties-common \
        && add-apt-repository ppa:deadsnakes/ppa \
        && if [ "$GPU_BACKEND" = "intel" ]; then add-apt-repository -y ppa:kobuk-team/intel-graphics; fi \
        && apt-get update && apt-get install -y \
            python3.11 \
            python3.11-venv \
            python3.11-dev \
            ffmpeg \
            libsndfile1 \
            curl \
        && rm -rf /var/lib/apt/lists/* \
        && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
        && ln -sf /usr/bin/python3.11 /usr/bin/python \
        && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11; \
    else \
        apt-get update && apt-get install -y \
            ffmpeg \
            libsndfile1 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Intel: oneapi-basekit bundles an older Level-Zero/compute-runtime
# (1.6.x) that crashes encoding GPU commands on Battlemage (Arc B580) --
# "Abort was called ... command_encoder_xehp_and_later.inl". Install the
# current driver from Intel's official PPA (repo added above) per
# https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html
# Requires the ubuntu24.04 base image above: this PPA has no jammy (22.04)
# build, and Intel's own repositories.intel.com jammy repo (already
# configured in the base image) tops out at driver 24.39, still too old
# for B580. The PPA's noble build matches the host's driver version exactly.
RUN if [ "$GPU_BACKEND" = "intel" ]; then \
        apt-get update && apt-get install -y \
            libze-intel-gpu1 \
            libze1 \
            intel-opencl-icd \
            intel-metrics-discovery \
            intel-gsc \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Install backend-specific requirements
COPY requirements.txt requirements-${GPU_BACKEND}.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-${GPU_BACKEND}.txt

COPY app/ ./app/
COPY entrypoint.sh /entrypoint.sh

ENV TORCH_HOME=/app/models/torch
ENV HF_HOME=/app/models/huggingface
ENV XDG_CACHE_HOME=/app/models/whisper
ENV PYTHONUNBUFFERED=1

# Level-Zero's sysman subsystem, which torch.xpu uses for device
# enumeration, is disabled by default -- without this, torch.xpu sees
# zero devices and gpu_backend.py silently falls back to CPU.
ENV ZES_ENABLE_SYSMAN=1

# torch's pip-installed oneAPI runtime libs (dist-packages/../../.. -> /usr/local/lib)
# must resolve before the base image's own system oneAPI install, or symbol
# versions can mismatch between the two copies despite matching sonames.
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Non-root runtime user. IDs are assigned by the system (some base images
# already occupy GID/UID 1000) and re-mapped to PUID/PGID at container start.
RUN groupadd appgroup \
    && useradd -g appgroup -M -s /bin/false appuser \
    && mkdir -p /app/models /app/jobs \
    && chown -R appuser:appgroup /app \
    && chmod +x /entrypoint.sh

EXPOSE 3334

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3334", "--no-access-log"]
