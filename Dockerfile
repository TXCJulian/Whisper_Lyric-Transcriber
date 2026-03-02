FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
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
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11 (apt python3-pip is tied to system Python 3.10)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV TORCH_HOME=/app/models/torch
ENV HF_HOME=/app/models/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 3334

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3334", "--no-access-log"]
