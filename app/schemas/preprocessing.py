from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PreprocessingConfig(BaseModel):
    target_column: str = Field(default="Target", description="Column to map as target variable")
    positive_class_val: Optional[Any] = Field(default=None, description="Value representing Dropout (class 1)")
    drop_columns: List[str] = Field(default_factory=list, description="Columns to exclude from features")
    impute_strategy: str = Field(default="mean", description="Strategy for missing numerical features: mean, median, most_frequent")
    scale_features: bool = Field(default=True, description="Whether to apply StandardScaler to numerical features")
    handle_duplicates: bool = Field(default=True, description="Whether to drop duplicate rows")
    test_size: float = Field(default=0.15, description="Validation split ratio")
    val_size: float = Field(default=0.15, description="Test split ratio")
    random_state: int = Field(default=42, description="Seed for reproducible train/val/test split")


class PreprocessingArtifactResponse(BaseModel):
    id: int
    dataset_id: int
    version: str
    artifact_path: str
    feature_schema: Dict[str, Any]
    target_mapping: Optional[Dict[str, Any]] = None
    split_summary: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
