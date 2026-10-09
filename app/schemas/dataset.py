from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DatasetResponse(BaseModel):
    id: int
    name: str
    filename: str
    row_count: Optional[int] = None
    feature_count: Optional[int] = None
    target_column: Optional[str] = "Target"
    checksum: Optional[str] = None
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetDetailResponse(DatasetResponse):
    columns: List[str] = []
    sample_rows: List[Dict[str, Any]] = []
    summary_stats: Dict[str, Any] = {}
