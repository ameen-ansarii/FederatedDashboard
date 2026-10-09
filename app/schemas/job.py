from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict
from app.models.domain import JobStatus


class JobResponse(BaseModel):
    id: str
    type: str
    status: JobStatus
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
