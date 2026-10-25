from enum import Enum
from typing import Literal

from pydantic import BaseModel


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
    status: JobStatus
    progress: str | None = None
    error: str | None = None
    warning: str | None = None
    warnings: list[str] = []
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
    path: str
    status: ItemStatus
    job_id: str | None = None
    progress: str | None = None
    error: str | None = None
    warnings: list[str] = []
    outputs: list[str] = []
    written_alongside: list[str] = []


class BatchSummary(BaseModel):
    batch_id: str
    status: BatchStatus
    created_at: str
    completed_at: str | None = None
    total: int
    counts: BatchCounts


class BatchStatusResponse(BatchSummary):
    output_mode: str
    items: list[BatchItemResponse]


class BatchDeleteResponse(BaseModel):
    status: Literal["deleted", "cancelling"]
    batch_id: str
