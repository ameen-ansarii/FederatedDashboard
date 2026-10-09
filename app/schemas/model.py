from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict


class ModelRegistryResponse(BaseModel):
    id: int
    name: str
    version: str
    framework: str
    metrics: Dict[str, Any]
    checksum: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelDetailResponse(ModelRegistryResponse):
    lineage: Dict[str, Any]
    user_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)
