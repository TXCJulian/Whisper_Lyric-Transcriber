from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    lrc = "lrc"
    txt = "txt"
    all = "all"


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ItemStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"
    cancelled = "cancelled"


class BatchStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus = Field(description="`pending`, `processing`, `completed` or `failed`.")
    progress: str | None = Field(None, description="Latest progress message, e.g. `Transcribing audio`.")
    error: str | None = Field(None, description="Error message when `status` is `failed`.")
    warning: str | None = Field(None, description="The most recent warning (kept for compatibility; see `warnings`).")
    warnings: list[str] = Field([], description="Every warning of the job, e.g. no vocals detected or a Genius match below the threshold.")
    created_at: str
    completed_at: str | None = None


class JobResultFile(BaseModel):
    filename: str
    format: str


class JobResultResponse(BaseModel):
    job_id: str
    files: list[JobResultFile]


class ErrorDetail(BaseModel):
    code: str
    message: str
    path: str | None = None
    batch_id: str | None = None
    errors: list[dict] | None = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail | str | list


class BatchCreatedResponse(BaseModel):
    batch_id: str
    total: int


class BatchCounts(BaseModel):
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    cancelled: int = 0


class BatchItemResponse(BaseModel):
    path: str = Field(description="Source file of this song.")
    status: ItemStatus = Field(description="`pending`, `processing`, `completed`, `failed`, `skipped` (lyrics already existed next to the song) or `cancelled`.")
    job_id: str | None = Field(None, description="The song's job; usable with `GET /jobs/{job_id}` and its result endpoints.")
    progress: str | None = None
    error: str | None = None
    warnings: list[str] = []
    outputs: list[str] = Field([], description="Result file names, downloadable via the job or the batch ZIP.")
    written_alongside: list[str] = Field([], description="Files written next to the song (`alongside` mode).")


class BatchSummary(BaseModel):
    batch_id: str
    status: BatchStatus
    created_at: str
    completed_at: str | None = None
    total: int
    counts: BatchCounts


class BatchStatusResponse(BatchSummary):
    output_mode: str = Field(description="`download` or `alongside`, fixed when the batch was created.")
    items: list[BatchItemResponse]


class BatchDeleteResponse(BaseModel):
    status: Literal["deleted", "cancelling"]
    batch_id: str


class VersionInfo(BaseModel):
    version: str = Field(description="Release version (from pyproject.toml).", examples=["2.0.0"])
    build: str = Field(description="CI build number of this version; `beta` for branch builds, `dev` locally.", examples=["3"])
    commit: str = Field(description="Git commit the image was built from.")
    image_variant: str = Field(description="Image variant: `nvidia`, `intel`, `amd`, `cpu`, or `local`.", examples=["nvidia"])


class BatchInfo(BaseModel):
    enabled: bool = Field(description="Whether batch endpoints accept requests (`TRANSCRIBER_ALLOWED_ROOTS` is set).")
    allowed_roots: list[str] = Field(description="Directories batch requests may read from.")
    output_mode: str = Field(description="`download` or `alongside` (`TRANSCRIBER_BATCH_OUTPUT`).")


class HealthResponse(BaseModel):
    status: str = Field(description="Always `ok` when the service answers.", examples=["ok"])
    version: VersionInfo
    runtime_profile: str | None = Field(
        description="Environment chosen at container start: `nvidia`, `nvidia-legacy`, `intel`, `amd` or `cpu`."
    )
    gpu_backend: str = Field(description="Active compute backend: `cuda`, `xpu`, `rocm` or `cpu`.", examples=["cuda"])
    gpu_name: str = Field(description="GPU name, or `CPU`.", examples=["NVIDIA GeForce RTX 3060 Ti"])
    transcription_engine: str = Field(description="`faster-whisper` or `openai-whisper`.")
    vram_total_mb: int | None = Field(description="Total GPU memory in MB; null on CPU.")
    whisper_model_fit: dict[str, bool] | None = Field(
        description="Per Whisper model, whether it fits into the GPU's memory; null on CPU (no limit)."
    )
    correction_min_match: float = Field(description="Default Genius correction threshold in percent (`CORRECTION_MIN_MATCH`).")
    batch: BatchInfo


class DeleteResponse(BaseModel):
    status: str = Field(examples=["deleted"])
    job_id: str
