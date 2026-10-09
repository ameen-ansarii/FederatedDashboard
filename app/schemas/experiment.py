from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkConfig(BaseModel):
    artifact_id: int = Field(..., description="ID of the PreprocessingArtifact to use for all runs")
    name: Optional[str] = Field(default="FL vs Centralized Benchmark", description="Name of the benchmark experiment")
    num_clients: int = Field(default=3, ge=1, description="Number of federated client partitions")
    rounds: int = Field(default=3, ge=1, description="Number of federated communication rounds")
    local_epochs: int = Field(default=1, ge=1, description="Number of local epochs per federated round")
    centralized_epochs: int = Field(default=5, ge=1, description="Number of epochs for centralized baseline")
    batch_size: int = Field(default=16, ge=1, description="Mini-batch size")
    learning_rate: float = Field(default=0.01, gt=0.0, description="Optimizer learning rate")
    hidden_dims: List[int] = Field(default=[32, 16], description="Hidden layer dimensions for DropoutMLP")
    dropout_rate: float = Field(default=0.1, ge=0.0, lt=1.0, description="Dropout regularization probability")
    dirichlet_alpha: float = Field(default=0.5, gt=0.0, description="Dirichlet concentration parameter for Non-IID partitioning")
    mu_values: List[float] = Field(default=[0.01, 0.1, 1.0], min_length=1, description="List of proximal penalty μ values for FedProx evaluation")
    random_state: int = Field(default=42, description="Random seed for reproducibility across all runs")


class ModelComparisonRequest(BaseModel):
    model_ids: List[int] = Field(..., min_length=1, description="List of registered model IDs to compare")
    artifact_id: Optional[int] = Field(None, description="Optional target PreprocessingArtifact ID for fresh test-set evaluation")


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    loss: float
    confusion_matrix: Optional[List[List[int]]] = None


class ModelEvaluationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: int
    model_version: str
    algorithm: str
    variant: str
    metrics: EvaluationMetrics
    delta_from_baseline: Optional[Dict[str, float]] = None


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    artifact_id: int
    config: Dict[str, Any]
    summary: Dict[str, Any]
    models: List[ModelEvaluationSummary]
    client_distributions: Optional[Dict[str, Any]] = None
    created_at: datetime


class ConvergenceTrajectoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment_id: str
    trajectories: Dict[str, Dict[str, List[float]]]


class MuSensitivityResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment_id: str
    mu_analysis: List[Dict[str, Any]]


class ExperimentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    artifact_id: int
    created_at: datetime
    model_count: int


class ExperimentListResponse(BaseModel):
    items: List[ExperimentListItem]
    total: int
    skip: int
    limit: int
