ARG GPU_BACKEND=nvidia

# ── Base images per backend ────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base-nvidia
FROM intel/oneapi-basekit:2025.3.1-0-devel-ubuntu22.04 AS base-intel
FROM rocm/rocm-terminal:6.4 AS base-amd
FROM python:3.11-slim AS base-cpu

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM base-${GPU_BACKEND} AS runtime

ARG GPU_BACKEND=nvidia
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV GPU_BACKEND=${GPU_BACKEND}

# Install Python 3.11 + system deps (skip for cpu base which already has Python)
RUN if [ "$GPU_BACKEND" != "cpu" ]; then \
        apt-get update && apt-get install -y \
            software-properties-common \
        && add-apt-repository ppa:deadsnakes/ppa \
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
