# Lyric Transcriber

Automated song lyrics transcription service. Audio files are processed through vocal separation, speech-to-text transcription, and optional lyrics correction using Genius as a reference. Exposed as an async HTTP API with job management.

## Disclaimer

This project uses web scraping (via [`lyricsgenius`](https://github.com/johnwmillr/LyricsGenius)) to fetch lyrics from Genius for correction purposes. The Genius API does not provide lyrics directly — scraping their website to obtain them **violates the [Genius Terms of Service](https://genius.com/static/terms)**. By providing your own API key and running this service, you accept full responsibility for compliance with Genius's terms and applicable copyright laws. Lyrics are processed locally and are not stored or redistributed by this project.

## Related Project

This service is designed to run standalone and be consumed purely via its HTTP API — it has no UI of its own. [Media-Helper](https://github.com/TXCJulian/Media-Helper) is a reference consumer: a containerized media management tool (tv-show/music renaming, media cutting, downloading) whose **Lyrics Transcription** module proxies to this service and builds a full UI on top of it (job submission, real-time progress via SSE, GPU health/model-fit display). Media-Helper deliberately keeps this service external and optional — see its README's own "Related Projects" section — so deployments that don't need lyrics transcription never have to build or pull this GPU-capable image, and this service can run on whatever machine actually has the right hardware, pointed at via `TRANSCRIBER_URL`.

## Pipeline

```text
Audio Upload ─► Vocal Separation (Demucs) ─► Transcription (faster-whisper) ─► Lyrics Correction (Genius) ─► .lrc / .txt
```

1. **Vocal Separation** — Isolates vocals from the audio using Demucs v4 (`htdemucs` by default, configurable via `DEMUCS_MODEL`), the successor to the older HDemucs bundle previously used here. `apply_model()` splits long tracks into overlapping segments internally to stay within VRAM limits. The model is unloaded after this step to free GPU memory for Whisper.
2. **Transcription** — Converts vocals to text with word-level timestamps. Uses faster-whisper (CTranslate2) on NVIDIA (Turing and newer) and CPU, or OpenAI Whisper (PyTorch) on older NVIDIA cards, Intel XPU and AMD ROCm. VAD filtering removes silence. An initial prompt with artist/title is passed to improve proper name recognition. The model is kept warm for `WHISPER_IDLE_UNLOAD_SECONDS` (default 15s) after each job so back-to-back requests skip the reload, then auto-unloads to free GPU memory when idle.
3. **Lyrics Correction** — Fetches reference lyrics from Genius and applies word-level fuzzy matching (`difflib.SequenceMatcher`) to fix transcription errors. Splits long lines at natural break points based on the Genius line structure. Handles Genius's Cyrillic/Greek homoglyph copy-protection by mapping them back to Latin equivalents.

Each step is optional and can be run independently via separate endpoints.

## Requirements

- Docker (recommended)
- A supported GPU (optional, CPU fallback available):
  - **NVIDIA** — NVIDIA Container Toolkit + CUDA GPU — verified end-to-end on a **RTX 3060 Ti** on ***Ubuntu 26.04 Server*** running ***Kubernetes (k3s)***. This should also work on ***Windows 11*** via Docker Desktop, though that path is no longer part of end-to-end verification for newer releases of this project.
  - **Older NVIDIA cards** (Maxwell/Pascal/Volta, e.g. **Quadro M4000** / **Quadro P2200**) — handled by the same `nvidia` image: at start it reads the GPU's compute capability and, below 7.5, switches to a second environment built on PyTorch's legacy cu126 wheels (the newer wheels no longer ship kernels for these architectures). That environment uses the OpenAI Whisper engine, since faster-whisper's CTranslate2 backend also dropped these architectures, and runs it in fp32: these chips have no fast fp16 math, so on a Quadro P2200 fp32 was 2.6× faster and is the only precision in which `large-v3-turbo` fits in its 5 GB of VRAM. Cards without a power connector, like the P2200, draw short spikes well above their 75 W rating under load, so the power supply needs headroom: on a 150 W power brick the card repeatedly dropped off the PCIe bus (Xid 79), on a 500 W ATX supply it ran fine. Triton JIT kernels aren't supported on these cards either, so word-alignment falls back to whisper's slower pure-PyTorch implementation — expect noticeably longer transcription times than on newer cards.
  - **Intel Arc** — Intel GPU with Level Zero drivers. Requires **Above 4G Decoding** and **Resizable BAR** enabled in the system BIOS/UEFI, or the GPU will not be usable by the container. Verified end-to-end on an **Arc B580** (Battlemage) on ***Ubuntu 26.04 Desktop*** running ***Docker***. The image is based on Ubuntu 26.04 and installs a current Level-Zero/compute-runtime driver from [Intel's official PPA](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html) at build time, since Ubuntu's own packaged driver is too old for Battlemage (Whisper decoding fails on a B580 with it).
  - **AMD Radeon** — ROCm-supported GPU. ⚠️ **Untested** — the Docker image builds and the CI pipeline publishes it, but the pipeline has not been run against real AMD hardware.
- Genius API access token (for lyrics correction)

## Quick Start

### 1. Genius API Setup

1. Create an API client at [genius.com](https://genius.com/api-clients)
2. Copy your access token
3. Create a `.env` file in the repository root (see `.env.example`):

   ```bash
   GENIUS_ACCESS_TOKEN=your_token_here

   # Optional: speeds up the first model download (~5 GB) by lifting the
   # anonymous Hugging Face rate limit. Create one at
   # https://huggingface.co/settings/tokens with read scope.
   HF_TOKEN=
   ```

### 2. Run with Docker Compose

Run from the repository root:

```bash
# NVIDIA (any generation — older cards are detected automatically)
docker compose -f deploy/compose/docker-compose.yml -f deploy/compose/docker-compose.nvidia.yml --env-file .env up -d --build

# Intel Arc
GPU_BACKEND=intel docker compose -f deploy/compose/docker-compose.yml -f deploy/compose/docker-compose.intel.yml --env-file .env up -d --build

# AMD Radeon
GPU_BACKEND=amd docker compose -f deploy/compose/docker-compose.yml -f deploy/compose/docker-compose.amd.yml --env-file .env up -d --build

# CPU only
GPU_BACKEND=cpu docker compose -f deploy/compose/docker-compose.yml --env-file .env up -d --build
```

This builds the image for your hardware (slim Debian base — Ubuntu 26.04 for Intel — with Python 3.14), starts the service on port **3334**, and creates a named volume for model caching.

#### Prebuilt Images

CI publishes prebuilt images to GitHub Container Registry (`ghcr.io/txcjulian/whisper-lyric-transcriber`), one per hardware variant:

| Variant | Runs on |
| --- | --- |
| `nvidia` | Every NVIDIA generation from Maxwell on (picks the modern or legacy environment from the GPU's compute capability at start), and CPU when no GPU is visible |
| `intel` | Intel Arc (XPU) |
| `amd` | AMD Radeon (ROCm) |
| `cpu` | CPU only — the smallest image |

Tags, with `<variant>` one of the above:

| Tag | Moves? | Meaning |
| --- | --- | --- |
| `latest-<variant>` | yes | Newest build from `master` |
| `X.Y.Z-<variant>`, `X.Y-<variant>`, `X-<variant>` | yes | Newest build of that version line |
| `X.Y.Z-N-<variant>` | **no** | One specific build — pin this in deployments |
| `beta-<variant>` | yes | Newest build from any feature branch (testing only) |

The version `X.Y.Z` is the one in `pyproject.toml`. `N` is the build number: it starts at 1 for each new version and goes up with every rebuild of the same version. CI rebuilds `master` weekly to pick up OS security updates, so `2.0.0-3-nvidia` is the same code as `2.0.0-1-nvidia` with newer system packages. `GET /health` reports which version, build and commit is running.

> **`*-nvidia-legacy` tags are discontinued** and no longer updated. Switch to `*-nvidia` — it detects Maxwell/Pascal/Volta cards on its own. An existing `GPU_BACKEND=nvidia-legacy` setting keeps working and forces the legacy environment.

### 3. Verify

```bash
curl http://localhost:3334/health
```

```json
{
  "status": "ok",
  "version": {"version": "2.0.0", "build": "3", "commit": "4f1c2e9…", "image_variant": "nvidia"},
  "runtime_profile": "nvidia",
  "gpu_backend": "cuda",
  "gpu_name": "NVIDIA GeForce RTX ...",
  "transcription_engine": "faster-whisper",
  "vram_total_mb": 24564,
  "whisper_model_fit": {
    "tiny": true, "base": true, "small": true,
    "medium": true, "large-v2": true, "large-v3": true, "large-v3-turbo": true
  }
}
```

## API Reference

All processing endpoints are asynchronous — they return a `job_id` immediately. Poll the job status endpoint until completion, then download results.

Interactive documentation of every endpoint, field and response is served at **`/docs`** (Swagger UI) and **`/redoc`**.

### Health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service status, running version/build/commit and image variant (`version`), the environment the image chose at start (`runtime_profile`: `nvidia`, `nvidia-legacy`, `intel`, `amd` or `cpu`), GPU backend/name, transcription engine, total VRAM, per-model VRAM fit (`whisper_model_fit`) so callers can rule out model sizes too large for the GPU, the default correction threshold (`correction_min_match`) and batch settings (`batch`) |

### Processing Endpoints

All accept `multipart/form-data` and return `{"job_id": "<id>"}`.

#### `POST /transcribe` — Full Pipeline

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | *(required)* | Audio file |
| `format` | string | `lrc` | Output format: `lrc`, `txt`, or `all` |
| `no_separation` | bool | `false` | Skip vocal separation (if vocals are already isolated) |
| `demucs_model` | string | `htdemucs` | Demucs model to use for vocal separation |
| `whisper_model` | string | `large-v3-turbo` | Whisper model to use |
| `language` | string | auto-detect | Force language code (e.g. `de`, `en`) |
| `artist` | string | from metadata | Artist name for Genius lookup (overrides audio metadata) |
| `title` | string | from metadata | Song title for Genius lookup (overrides audio metadata) |
| `no_correction` | bool | `false` | Skip Genius lyrics correction |
| `correction_min_match` | number 0–100 | `CORRECTION_MIN_MATCH` | Keep the Whisper text if the Genius lyrics match less than this percentage (see [How Lyrics Correction Works](#how-lyrics-correction-works)) |

#### `POST /separate` — Vocal Separation Only

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | *(required)* | Audio file |
| `demucs_model` | string | `htdemucs` | Demucs model to use |

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
| `correction_min_match` | number 0–100 | `CORRECTION_MIN_MATCH` | Keep the Whisper text below this Genius match percentage |

### Job Management

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/jobs/{job_id}` | Job status (`pending` / `processing` / `completed` / `failed`), progress message, error, and warnings (`warnings` lists all of them; `warning` keeps the most recent) |
| `GET` | `/jobs/{job_id}/result` | Download result — single file directly, multiple files as ZIP, `204` if no output |
| `GET` | `/jobs/{job_id}/result/info` | Result file metadata without downloading |
| `DELETE` | `/jobs/{job_id}` | Delete job and its files (`409` if still processing, or if the job belongs to a batch — delete the batch instead) |

### Usage Example

```bash
# Submit a transcription job
JOB_ID=$(curl -s -F "file=@song.flac" -F "language=de" http://localhost:3334/transcribe | jq -r '.job_id')

# Poll until completed
curl http://localhost:3334/jobs/$JOB_ID

# Download result
curl -o lyrics.lrc http://localhost:3334/jobs/$JOB_ID/result
```

## Batch Processing

Transcribe a whole album — or any list of songs — as **one batch** that runs on the server, independent of the client. The transcriber reads the songs directly from storage mounted into the container, so nothing is uploaded.

### Setup

1. Mount your music library into the container, e.g. `-v /srv/music:/music:ro`.
2. Set `TRANSCRIBER_ALLOWED_ROOTS=/music`. Batch requests may only read inside these directories (comma-separated); every path is fully resolved (`..`, symlinks) before the check. Without this variable the batch endpoints answer `403 batch_disabled`.
3. Choose where results go with `TRANSCRIBER_BATCH_OUTPUT`:
   - `download` (default) — results stay in the transcriber and are downloaded per song or as one ZIP. A read-only mount is enough.
   - `alongside` — additionally write `<song>.lrc` / `<song>.txt` next to each song. Needs a writable mount. Existing lyrics files are never overwritten while `skip_existing` is on (the default); such songs are reported as `skipped`.

### Starting a batch

```bash
# A folder (one album); add -F recursive=true to include subfolders
curl -s -F "directory=/music/Artist/Album" http://localhost:3334/batch
# -> {"batch_id": "3f9c1a2b7d4e", "total": 12}

# A CSV with one path per line (optional "path" header; quote paths containing commas)
cat > songs.csv <<'EOF'
path
/music/A/01 Song.flac
"/music/B/02 Illegal, legal.flac"
EOF
curl -s -F "csv=@songs.csv" http://localhost:3334/batch
```

`POST /batch` accepts the same options as `/transcribe` (`format`, `no_separation`, `demucs_model`, `whisper_model`, `language`, `no_correction`, `correction_min_match`) plus `directory` **or** `csv`, `recursive` and `skip_existing`. Artist and title for Genius come from each file's tags. Songs run one after another, sharing the GPU queue with single jobs. A CSV with any invalid line is rejected as a whole, listing every bad line.

### Following and collecting a batch

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/batches` | All batches, newest first, with per-status counts |
| `GET` | `/batches/{batch_id}` | Batch status and per-song `status` (`pending`, `processing`, `completed`, `failed`, `skipped`, `cancelled`), progress, warnings, result files and the song's `job_id` |
| `GET` | `/batches/{batch_id}/result` | ZIP of all finished results, one folder per song (works while the batch is running; `204` if nothing yet) |
| `POST` | `/batches/{batch_id}/cancel` | Stop the batch but keep it: waiting songs are cancelled, a running song finishes, finished results stay downloadable |
| `DELETE` | `/batches/{batch_id}` | Cancel and remove the batch and its results (`202` with `status: cancelling` while a song is still running). Files written next to songs are kept |

Each song is also a regular job, so `GET /jobs/{job_id}` and `/jobs/{job_id}/result` work for single songs. Batches expire like jobs, `JOB_TTL_SECONDS` after they end.

### Batch errors

Batch errors carry a machine-readable code: `{"detail": {"code": "...", "message": "...", ...}}`.

| Status | Code | Meaning |
| --- | --- | --- |
| 403 | `batch_disabled` | `TRANSCRIBER_ALLOWED_ROOTS` is not set |
| 403 | `path_not_allowed` | The path resolves outside the allowed roots |
| 404 | `path_not_found` | The directory doesn't exist |
| 404 | `batch_not_found` | Unknown, deleted or expired batch |
| 409 | `batch_finished` | Cancel on a batch that has already ended |
| 409 | `job_in_batch` | `DELETE /jobs/{id}` on a song of a batch |
| 422 | `invalid_batch_request` | Neither or both of `directory`/`csv`, no audio files found, or more than 5000 songs |
| 422 | `invalid_csv` | Invalid CSV lines, listed in `errors: [{line, path, reason}]` |

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
| `DEMUCS_MODEL` | `htdemucs` | Demucs model name for vocal separation (e.g. `htdemucs` for faster/lower-latency single-pass separation) |
| `DEMUCS_SEGMENT_SECONDS` | model default | Override the chunk length `apply_model()` uses to split long tracks; lower it if you hit GPU OOM on very long tracks |
| `JOBS_DIR` | `/app/jobs` | Base directory for job input/output files |
| `GPU_BACKEND` | auto-detect | Usually leave unset. `nvidia-legacy` forces the Maxwell/Pascal/Volta environment and `cpu` forces CPU; `nvidia`/`cuda` (or unset) let the `nvidia` image pick by GPU generation. Outside Docker it also accepts `xpu`/`intel` and `rocm`/`amd`. With Docker Compose, set `RUNTIME_GPU_BACKEND` instead, because compose uses `GPU_BACKEND` to choose which image variant to build |
| `TRANSCRIBER_ALLOWED_ROOTS` | *(empty)* | Comma-separated directories batch requests may read from. Empty disables batch processing (see [Batch Processing](#batch-processing)) |
| `TRANSCRIBER_BATCH_OUTPUT` | `download` | Where batch results go: `download` (kept for download only) or `alongside` (also written next to each song; needs write access) |
| `CORRECTION_MIN_MATCH` | `0` | Genius correction threshold in percent: below it the Genius lyrics are discarded and the Whisper transcription is kept. `0` always applies the correction |
| `PUID` | `1000` | UID the container process runs as (container runs as a non-root `appuser`) |
| `PGID` | `1000` | GID the container process runs as |

### Docker Compose

Compose files live in `deploy/compose/`. The base `docker-compose.yml` contains no GPU reservation; per-backend override files handle device passthrough:

- **Port**: `3334`
- **Volume**: `model-cache` persists downloaded model weights (~5 GB) across container restarts
- **Restart**: `unless-stopped`
- **Build arg**: `GPU_BACKEND` selects the image variant (`nvidia`, `intel`, `amd`, `cpu`)

#### GPU Device Passthrough

| Backend | Override file |
| --- | --- |
| NVIDIA (all generations) | `deploy/compose/docker-compose.nvidia.yml` |
| Intel | `deploy/compose/docker-compose.intel.yml` |
| AMD | `deploy/compose/docker-compose.amd.yml` |
| CPU | none |

See [Run with Docker Compose](#2-run-with-docker-compose) for the full commands.

### Kubernetes

Generic Kustomize examples (base + NVIDIA overlay) and notes on GPU → CPU failover are in [`deploy/k8s/`](deploy/k8s/README.md).

## Local Development (Without Docker)

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The project needs Python 3.14 on Linux x86_64; on Windows, use WSL or Docker. Each hardware target is an extra, and you pick exactly one:

```bash
uv sync --extra nvidia          # NVIDIA Turing (RTX 20xx) and newer
uv sync --extra nvidia-legacy   # NVIDIA Maxwell/Pascal/Volta
uv sync --extra intel           # Intel Arc
uv sync --extra amd             # AMD Radeon
uv sync --extra cpu             # CPU only
```

Start the server:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 3334
```

Run the tests (no GPU or models needed; they use the `cpu` environment):

```bash
uv run --extra cpu python -m unittest discover -v
```

With the `nvidia-legacy` extra, also set `GPU_BACKEND=nvidia-legacy` so the app uses the OpenAI Whisper engine. Inside Docker the entrypoint does this for you. `GPU_BACKEND=cpu` forces CPU.

## Project Structure

```text
├── Dockerfile              # Multi-stage build, GPU_BACKEND arg selects the variant
├── pyproject.toml          # Dependencies; one extra per hardware target
├── uv.lock                 # Locked versions for every extra
├── docker/
│   ├── entrypoint.sh       # PUID/PGID remap, GPU groups, drops root
│   └── select-profile.sh   # Picks the runtime environment (NVIDIA modern/legacy/CPU)
├── deploy/
│   ├── compose/            # docker-compose.yml + nvidia/intel/amd overrides
│   └── k8s/                # Generic Kustomize examples
├── .env                    # Genius API token (not committed)
└── app/
    ├── main.py             # FastAPI app, route handlers, startup logic
    ├── version.py          # Version/build/commit reported by /health
    ├── runtime_profile.py  # Runtime profile chosen by the entrypoint
    ├── config.py           # Batch/threshold settings from the environment
    ├── paths.py            # TRANSCRIBER_ALLOWED_ROOTS path checks
    ├── batch_input.py      # Song lists from a directory or a CSV
    ├── batch_manager.py    # Batches: song queue, cancel/delete, alongside output, ZIP
    ├── batch_routes.py     # /batch and /batches endpoints
    ├── api_docs.py         # OpenAPI texts shared by the endpoints
    ├── models.py           # Pydantic request/response models
    ├── job_manager.py      # Async job queue with threading, TTL eviction
    ├── pipeline.py         # Orchestrates separation → transcription → correction
    ├── separation.py       # Demucs (htdemucs) vocal separation
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

**Correction threshold:** the match rate is the share of aligned words that Whisper and Genius agree on exactly. When Genius returns lyrics for a different version of the song — a dub in another language, a live version, a remix — the rate drops, and applying them would rewrite correct lines with the wrong text. Set `CORRECTION_MIN_MATCH` (or `correction_min_match` per request) to a percentage such as `30`: below it the Genius lyrics are discarded, the Whisper transcription is kept, and the job gets a warning like `Genius lyrics matched only 10.5 % (threshold 30 %): kept the Whisper transcription`. The default `0` always applies the correction.

## Tech Stack

| Component | Technology |
| --- | --- |
| HTTP Server | FastAPI + Uvicorn |
| Vocal Separation | Demucs v4 (`htdemucs` by default, via the `demucs` package) |
| Speech-to-Text | faster-whisper (CUDA/CPU), OpenAI Whisper (XPU/ROCm) |
| Lyrics Reference | Genius API (`lyricsgenius`) |
| Audio Metadata | mutagen |
| GPU Support | NVIDIA CUDA (cu130, plus cu126 for Maxwell/Pascal/Volta), Intel XPU, AMD ROCm 7.2, CPU fallback |
| Python | 3.14, dependencies managed with uv |
| Container Runtime | Docker, multi-stage build on a slim Debian base (Ubuntu 26.04 for Intel) |

## License

[MIT](LICENSE) — Copyright (c) 2026 TXCJulian
