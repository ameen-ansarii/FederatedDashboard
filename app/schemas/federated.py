from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FedAvgTrainingConfig(BaseModel):
    artifact_id: int = Field(..., description="ID of the Phase 2 Preprocessing Artifact to train on")
    name: str = Field(default="FedAvg Student Dropout Classifier", description="Name for the registered model")
    num_clients: int = Field(default=5, description="Number of federated client partitions")
    partitioning: str = Field(default="iid", description="Data partitioning strategy: iid or non_iid")
    dirichlet_alpha: float = Field(default=0.5, description="Dirichlet alpha parameter for Non-IID label skew (lower = higher heterogeneity)")
    rounds: int = Field(default=5, description="Number of federated aggregation rounds")
    local_epochs: int = Field(default=3, description="Local client training epochs per round")
    learning_rate: float = Field(default=0.001, description="Learning rate for local optimization")
    batch_size: int = Field(default=32, description="Local batch size")
    hidden_dims: List[int] = Field(default_factory=lambda: [64, 32], description="MLP hidden dimensions")
    dropout_rate: float = Field(default=0.2, description="Dropout rate")
    optimizer: str = Field(default="adam", description="Optimizer choice: adam, adamw, sgd")
    random_state: int = Field(default=42, description="Derived seed for this training sub-run")
    base_random_state: Optional[int] = Field(
        default=None,
        description="Root experiment seed when random_state is a derived child seed",
    )


class FedProxTrainingConfig(FedAvgTrainingConfig):
    name: str = Field(default="FedProx Student Dropout Classifier", description="Name for the registered FedProx model")
    mu: float = Field(default=0.01, description="Proximal term penalty coefficient (mu >= 0.0)")


class FederatedClientResponse(BaseModel):
    id: str
    name: str
    status: str
    ip_address: Optional[str] = None
    last_heartbeat: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
