from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.models.domain import JobStatus


class CentralizedTrainingConfig(BaseModel):
    artifact_id: int = Field(..., description="ID of the Phase 2 Preprocessing Artifact to train on")
    name: str = Field(default="Centralized Dropout MLP", description="Descriptive name for the trained model")
    hidden_dims: List[int] = Field(default_factory=lambda: [64, 32], description="List of hidden layer output sizes")
    dropout_rate: float = Field(default=0.2, description="Dropout probability for regularization")
    learning_rate: float = Field(default=0.001, description="Learning rate for optimization")
    batch_size: int = Field(default=32, description="Mini-batch size for DataLoader")
    epochs: int = Field(default=20, description="Number of training epochs")
    optimizer: str = Field(default="adam", description="Optimizer choice: adam, adamw, sgd")
    random_state: int = Field(default=42, description="Derived seed for this training sub-run")
    base_random_state: Optional[int] = Field(
        default=None,
        description="Root experiment seed when random_state is a derived child seed",
    )


class TrainingRunResponse(BaseModel):
    id: str
    type: str
    status: JobStatus
    hyperparameters: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    history: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
