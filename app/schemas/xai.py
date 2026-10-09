from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.models.domain import RiskLevel


class FeatureAttribution(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    feature_name: str = Field(..., description="Transformed feature name (e.g. Tuition_Paid_Yes, Age)")
    original_feature_name: Optional[str] = Field(None, description="Original raw feature name (e.g. Tuition_Paid)")
    feature_value: Optional[Any] = Field(None, description="Input value for this feature")
    attribution: float = Field(..., description="Attribution/importance score (positive pushes to Dropout, negative away)")
    direction: str = Field(..., description="'positive' (pushes toward Dropout) or 'negative' (pushes away)")
    rank: int = Field(..., description="1-based importance rank sorted by absolute attribution magnitude")


class XAIRequest(BaseModel):
    prediction_id: Optional[int] = Field(None, description="ID of an existing saved prediction to explain")
    model_id: Optional[int] = Field(None, description="Model ID (required if providing direct features without prediction_id)")
    features: Optional[Dict[str, Any]] = Field(None, description="Direct feature values to explain (if not referencing prediction_id)")


class GlobalSHAPRequest(BaseModel):
    model_id: int = Field(..., description="Model ID to compute global feature importance for")
    max_samples: int = Field(default=50, ge=10, le=500, description="Number of background/validation samples to evaluate")


class GlobalFeatureImportance(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    feature_name: str = Field(..., description="Transformed feature name")
    mean_abs_shap: float = Field(..., description="Mean absolute SHAP value across evaluated dataset samples")
    rank: int = Field(..., description="Importance rank (1 is most impactful globally)")


class XAIResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    method: str = Field(..., description="XAI method used: SHAP, LIME, or INTEGRATED_GRADIENTS")
    prediction_id: Optional[int] = Field(None, description="ID of explained prediction record if applicable")
    model_id: int = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")
    probability: float = Field(..., description="Predicted dropout probability")
    prediction_class: int = Field(..., description="Binary class: 1 (Dropout) or 0 (Not Dropout)")
    prediction_label: str = Field(..., description="Binary label")
    risk_level: RiskLevel = Field(..., description="Risk category: LOW, MEDIUM, or HIGH")
    features: List[FeatureAttribution] = Field(..., description="Sorted list of feature attributions")
    created_at: datetime = Field(..., description="Timestamp of explanation generation")


class GlobalSHAPResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: int
    model_version: str
    samples_evaluated: int
    feature_importance: List[GlobalFeatureImportance]


class UnifiedXAIResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prediction_id: int
    model_id: int
    model_version: str
    prediction: Dict[str, Any]
    explanations: Dict[str, Any]
