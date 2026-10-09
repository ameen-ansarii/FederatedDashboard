from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.models.domain import RiskLevel


class SinglePredictionRequest(BaseModel):
    model_id: int = Field(..., description="ID of the registered model to use for inference")
    features: Dict[str, Any] = Field(..., description="Key-value mapping of raw student feature inputs")
    save_to_history: bool = Field(default=True, description="Whether to persist this prediction to history")


class SinglePredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = Field(None, description="Prediction history ID if saved")
    model_id: int = Field(..., description="ID of the model used")
    model_version: str = Field(..., description="Version of the model used")
    probability: float = Field(..., description="Predicted dropout probability between 0.0 and 1.0")
    prediction_class: int = Field(..., description="Binary prediction class: 1 for Dropout, 0 for Not Dropout")
    prediction_label: str = Field(..., description="Human-readable prediction label: Dropout or Not Dropout")
    risk_level: RiskLevel = Field(..., description="Risk category: LOW (<0.40), MEDIUM (0.40-0.70), HIGH (>=0.70)")
    created_at: datetime = Field(..., description="Timestamp of inference")


class BatchPredictionRequest(BaseModel):
    model_id: int = Field(..., description="ID of the registered model to use for inference")
    records: List[Dict[str, Any]] = Field(..., min_length=1, description="List of student feature dictionaries")
    save_to_history: bool = Field(default=False, description="Whether to persist each record to prediction history")


class BatchPredictionItemResult(BaseModel):
    index: int = Field(..., description="Original zero-indexed position in batch")
    probability: float = Field(..., description="Predicted dropout probability")
    prediction_class: int = Field(..., description="Binary class (0 or 1)")
    prediction_label: str = Field(..., description="Binary label (Dropout or Not Dropout)")
    risk_level: RiskLevel = Field(..., description="Risk band")


class BatchPredictionSummary(BaseModel):
    total_samples: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    mean_probability: float


class BatchPredictionResultResponse(BaseModel):
    job_id: str
    status: str
    model_id: int
    model_version: str
    total_samples: int
    summary: BatchPredictionSummary
    predictions: List[BatchPredictionItemResult]


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: Optional[int]
    model_version: Optional[str] = None
    probability: float
    prediction_class: Optional[int]
    prediction_label: Optional[str]
    risk_level: RiskLevel
    feature_values: Dict[str, Any]
    batch_job_id: Optional[str] = None
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
    total: int
    skip: int
    limit: int
