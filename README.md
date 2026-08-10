# Lyric Transcriber

Automated song lyrics transcription service. Audio files are processed through vocal separation, speech-to-text transcription, and optional lyrics correction using Genius as a reference. Exposed as an async HTTP API with job management.

## Disclaimer

This project uses web scraping (via [`lyricsgenius`](https://github.com/johnwmillr/LyricsGenius)) to fetch lyrics from Genius for correction purposes. The Genius API does not provide lyrics directly — scraping their website to obtain them **violates the [Genius Terms of Service](https://genius.com/static/terms)**. By providing your own API key and running this service, you accept full responsibility for compliance with Genius's terms and applicable copyright laws. Lyrics are processed locally and are not stored or redistributed by this project.

## Pipeline

```text
Audio Upload ─► Vocal Separation (Demucs) ─► Transcription (faster-whisper) ─► Lyrics Correction (Genius) ─► .lrc / .txt
```

1. **Vocal Separation** — Isolates vocals from the audio using Demucs v4 (`htdemucs_ft` by default, configurable via `DEMUCS_MODEL`), the successor to the older HDemucs bundle previously used here. `apply_model()` splits long tracks into overlapping segments internally to stay within VRAM limits. The model is unloaded after this step to free GPU memory for Whisper.
2. **Transcription** — Converts vocals to text with word-level timestamps. Uses faster-whisper (CTranslate2) on NVIDIA/CPU or OpenAI Whisper (PyTorch) on Intel XPU/AMD ROCm. VAD filtering removes silence. An initial prompt with artist/title is passed to improve proper name recognition. The model is kept warm for `WHISPER_IDLE_UNLOAD_SECONDS` (default 15s) after each job so back-to-back requests skip the reload, then auto-unloads to free GPU memory when idle.
3. **Lyrics Correction** — Fetches reference lyrics from Genius and applies word-level fuzzy matching (`difflib.SequenceMatcher`) to fix transcription errors. Splits long lines at natural break points based on the Genius line structure. Handles Genius's Cyrillic/Greek homoglyph copy-protection by mapping them back to Latin equivalents.

Each step is optional and can be run independently via separate endpoints.

## Requirements

- Docker (recommended)
- A supported GPU (optional, CPU fallback available):
  - **NVIDIA** — NVIDIA Container Toolkit + CUDA GPU — **the only backend verified to work end-to-end**
  - **Intel Arc** — Intel GPU with oneAPI/Level Zero drivers. Requires **Above 4G Decoding** and **Resizable BAR** enabled in the system BIOS/UEFI, or the GPU will not be usable by the container. ⚠️ **Untested** — the Docker image builds and the CI pipeline publishes it, but the pipeline has not been run against real Intel Arc hardware.
  - **AMD Radeon** — ROCm-supported GPU. ⚠️ **Untested** — the Docker image builds and the CI pipeline publishes it, but the pipeline has not been run against real AMD hardware.
- Genius API access token (for lyrics correction)

## Quick Start

### 1. Genius API Setup

1. Create an API client at [genius.com](https://genius.com/api-clients)
2. Copy your access token
3. Create a `.env` file in the project root (see `.env.example`):

   ```bash
   GENIUS_ACCESS_TOKEN=your_token_here

   # Optional: speeds up the first model download (~5 GB) by lifting the
   # anonymous Hugging Face rate limit. Create one at
   # https://huggingface.co/settings/tokens with read scope.
   HF_TOKEN=
   ```

### 2. Run with Docker Compose

```bash
# NVIDIA (default — docker-compose.override.yml auto-loads GPU reservation)
docker compose up -d

# Intel Arc
GPU_BACKEND=intel docker compose -f docker-compose.yml -f docker-compose.intel.yml up -d --build

# AMD Radeon
GPU_BACKEND=amd docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d --build

# CPU only
GPU_BACKEND=cpu docker compose -f docker-compose.yml up -d --build
```

This builds the image with the appropriate base (CUDA 12.8, oneAPI, ROCm, or Python 3.11-slim), starts the service on port **3334**, and creates a named volume for model caching.

#### Prebuilt Images

CI publishes prebuilt images to GitHub Container Registry on every push to `master`:

| Backend | Image |
| --- | --- |
| NVIDIA | `ghcr.io/txcjulian/whisper-lyric-transcriber:latest-nvidia` |
| Intel Arc | `ghcr.io/txcjulian/whisper-lyric-transcriber:latest-intel` |
| AMD Radeon | `ghcr.io/txcjulian/whisper-lyric-transcriber:latest-amd` |

Only the NVIDIA image is verified to work end-to-end — see [Requirements](#requirements) for the Intel/AMD caveats.

### 3. Verify

```bash
curl http://localhost:3334/health
```

```json
{"status": "ok", "gpu_backend": "cuda", "gpu_name": "NVIDIA GeForce RTX ...", "transcription_engine": "faster-whisper"}
```

## API Reference

All processing endpoints are asynchronous — they return a `job_id` immediately. Poll the job status endpoint until completion, then download results.

### Health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status, GPU backend, GPU name, transcription engine |

### Processing Endpoints

All accept `multipart/form-data` and return `{"job_id": "<id>"}`.

#### `POST /transcribe` — Full Pipeline

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | *(required)* | Audio file |
| `format` | string | `lrc` | Output format: `lrc`, `txt`, or `all` |
| `no_separation` | bool | `false` | Skip vocal separation (if vocals are already isolated) |
| `demucs_model` | string | `htdemucs_ft` | Demucs model to use for vocal separation |
| `whisper_model` | string | `large-v3-turbo` | Whisper model to use |
| `language` | string | auto-detect | Force language code (e.g. `de`, `en`) |
| `artist` | string | from metadata | Artist name for Genius lookup (overrides audio metadata) |
| `title` | string | from metadata | Song title for Genius lookup (overrides audio metadata) |
| `no_correction` | bool | `false` | Skip Genius lyrics correction |

#### `POST /separate` — Vocal Separation Only

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | *(required)* | Audio file |
| `demucs_model` | string | `htdemucs_ft` | Demucs model to use |

Returns the isolated vocals as a `.wav` file.

#### `POST /transcribe-only` — Transcription Only

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | *(required)* | Audio file (should be vocals) |
| `format` | string | `lrc` | Output format |
| `whisper_model` | string | `large-v3-turbo` | Whisper model |
| `language` | string | auto-detect | Force language |

#### `POST /correct` — Transcribe + Correct (No Separation)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | *(required)* | Audio file |
| `artist` | string | *(required)* | Artist name for Genius lookup |
| `title` | string | *(required)* | Song title for Genius lookup |
| `format` | string | `lrc` | Output format |

### Job Management

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/jobs/{job_id}` | Job status (`pending` / `processing` / `completed` / `failed`), progress message, errors, warnings |
| `GET` | `/jobs/{job_id}/result` | Download result — single file directly, multiple files as ZIP, `204` if no output |
| `GET` | `/jobs/{job_id}/result/info` | Result file metadata without downloading |
| `DELETE` | `/jobs/{job_id}` | Delete job and its files (`409` if still processing) |

### Usage Example

```bash
# Submit a transcription job
JOB_ID=$(curl -s -F "file=@song.flac" -F "language=de" http://localhost:3334/transcribe | jq -r '.job_id')

# Poll until completed
curl http://localhost:3334/jobs/$JOB_ID

# Download result
curl -o lyrics.lrc http://localhost:3334/jobs/$JOB_ID/result
```

## Output Formats

### LRC (`.lrc`)

Timestamped lyrics in standard LRC format, compatible with most music players:

```text
[00:10.96] Schon wieder Outro, oder was?
[00:15.11] Da kann man nichts machen
```

### TXT (`.txt`)

Plain text lyrics without timestamps:

```text
Schon wieder Outro, oder was?
Da kann man nichts machen
```

## Configuration

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `GENIUS_ACCESS_TOKEN` | *(required)* | Genius API token for lyrics correction |
| `HF_TOKEN` | *(optional)* | Hugging Face token. Lifts the anonymous rate limit, speeding up the first model download (~5 GB). Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope) |
| `PRELOAD_MODELS` | `false` | Load Demucs + Whisper into memory at startup (slower start, faster first request) |
| `JOB_TTL_SECONDS` | `3600` | Seconds before completed/failed jobs are automatically cleaned up |
| `WHISPER_IDLE_UNLOAD_SECONDS` | `15` | Seconds of inactivity before the Whisper model is auto-unloaded from GPU memory |
| `DEMUCS_MODEL` | `htdemucs_ft` | Demucs model name for vocal separation (e.g. `htdemucs` for faster/lower-latency single-pass separation) |
| `DEMUCS_SEGMENT_SECONDS` | model default | Override the chunk length `apply_model()` uses to split long tracks; lower it if you hit GPU OOM on very long tracks |
| `JOBS_DIR` | `/app/jobs` | Base directory for job input/output files |
| `GPU_BACKEND` | auto-detect | Override GPU detection: `cuda`/`nvidia`, `xpu`/`intel`, `rocm`/`amd`, or `cpu` |
| `PUID` | `1000` | UID the container process runs as (container runs as a non-root `appuser`) |
| `PGID` | `1000` | GID the container process runs as |

### Docker Compose

The base `docker-compose.yml` contains no GPU reservation. Per-backend override files handle device passthrough:

- **Port**: `3334`
- **Volume**: `model-cache` persists downloaded model weights (~5 GB) across container restarts
- **Restart**: `unless-stopped`
- **Build arg**: `GPU_BACKEND` selects base image and requirements

#### GPU Device Passthrough

| Backend | Compose command |
| --- | --- |
| NVIDIA | `docker compose up` (override auto-loaded) |
| Intel | `GPU_BACKEND=intel docker compose -f docker-compose.yml -f docker-compose.intel.yml up --build` |
| AMD | `GPU_BACKEND=amd docker compose -f docker-compose.yml -f docker-compose.amd.yml up --build` |
| CPU | `GPU_BACKEND=cpu docker compose -f docker-compose.yml up --build` |

## Local Development (Without Docker)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install for your GPU:
pip install -r requirements-nvidia.txt   # NVIDIA CUDA
pip install -r requirements-intel.txt    # Intel Arc
pip install -r requirements-amd.txt      # AMD Radeon
pip install -r requirements-cpu.txt      # CPU only
```

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3334
```

Optionally override auto-detection: `GPU_BACKEND=cpu uvicorn app.main:app --host 0.0.0.0 --port 3334`

## Project Structure

```text
├── Dockerfile              # Multi-stage build with GPU_BACKEND arg
├── docker-compose.yml      # Base service definition (no GPU config)
├── docker-compose.override.yml # NVIDIA GPU reservation (auto-loaded)
├── docker-compose.nvidia.yml   # NVIDIA GPU reservation (explicit)
├── docker-compose.intel.yml    # Intel Arc device passthrough
├── docker-compose.amd.yml      # AMD ROCm device passthrough
├── requirements.txt        # Shared Python dependencies
├── requirements-nvidia.txt # PyTorch CUDA wheels
├── requirements-intel.txt  # PyTorch XPU + IPEX wheels
├── requirements-amd.txt    # PyTorch ROCm wheels
├── requirements-cpu.txt    # PyTorch CPU wheels
├── .env                    # Genius API token (not committed)
└── app/
    ├── main.py             # FastAPI app, route handlers, startup logic
    ├── models.py           # Pydantic request/response models
    ├── job_manager.py      # Async job queue with threading, TTL eviction
    ├── pipeline.py         # Orchestrates separation → transcription → correction
    ├── separation.py       # Demucs (htdemucs_ft) vocal separation
    ├── transcription.py    # Data classes (Segment, WordTiming), LRC/TXT output
    ├── transcription_engine.py # Engine abstraction (faster-whisper / OpenAI Whisper)
    ├── gpu_backend.py      # GPU vendor detection (CUDA/XPU/ROCm/CPU)
    └── correction.py       # Genius lyrics fetch, homoglyph cleanup, word-level alignment
```

## How Lyrics Correction Works

The correction step uses `difflib.SequenceMatcher` to align words between the Whisper transcription and the Genius reference lyrics at the word level. Line breaks don't need to match — only the words matter.

- **Equal words**: Genius spelling/casing is adopted, Whisper timestamps are kept
- **Replaced words**: Genius version is used (fixes Whisper mishearings), timing is interpolated
- **Deleted words** (Whisper-only): Dropped as hallucinations
- **Inserted words** (Genius-only): Added with timing estimated from neighboring words
- **Line splitting**: Long Whisper segments are re-split at Genius line break positions using word-level timestamps
- **Capitalization**: Mid-line words that Genius incorrectly capitalizes (line-start convention) are lowercased

Artist and title for the Genius lookup are read from audio metadata tags (via mutagen). Use the `artist` and `title` form fields to override if metadata is missing.

## Tech Stack

| Component | Technology |
| --- | --- |
| HTTP Server | FastAPI + Uvicorn |
| Vocal Separation | Demucs v4 (`htdemucs_ft` by default, via the `demucs` package) |
| Speech-to-Text | faster-whisper (CUDA/CPU), OpenAI Whisper (XPU/ROCm) |
| Lyrics Reference | Genius API (`lyricsgenius`) |
| Audio Metadata | mutagen |
| GPU Support | NVIDIA CUDA, Intel XPU (IPEX), AMD ROCm, CPU fallback |
| Container Runtime | Docker, multi-stage build |

## License

[MIT](LICENSE) — Copyright (c) 2026 TXCJulian
