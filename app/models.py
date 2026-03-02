from enum import Enum
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


class JobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: str | None = None
    error: str | None = None
    warning: str | None = None
    created_at: str
    completed_at: str | None = None


class JobResultFile(BaseModel):
    filename: str
    format: str


class JobResultResponse(BaseModel):
    job_id: str
    files: list[JobResultFile]
