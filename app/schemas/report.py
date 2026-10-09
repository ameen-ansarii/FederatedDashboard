from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.models.domain import JobStatus


class ReportGenerationRequest(BaseModel):
    experiment_id: str = Field(..., description="ID of the Experiment or Multi-Seed Study to report on")
    name: Optional[str] = Field(default=None, description="Descriptive title for the publication report")
    primary_metric: str = Field(default="f1", description="Primary metric for model ranking: f1, accuracy, roc_auc, loss")
    include_xai: bool = Field(default=True, description="Whether to include Explainable AI synthesis (SHAP/LIME/IG)")
    include_visualizations: bool = Field(default=True, description="Whether to generate publication figures")
    include_raw_seed_results: bool = Field(default=True, description="Whether to include granular seed-by-seed breakdown")
    include_reproducibility_manifest: bool = Field(default=True, description="Whether to generate reproducibility manifest")
    generate_markdown: bool = Field(default=True, description="Generate Markdown report (report.md)")
    generate_json: bool = Field(default=True, description="Generate machine-readable JSON (report.json)")
    generate_csv: bool = Field(default=True, description="Generate CSV metric tables")
    generate_figures: bool = Field(default=True, description="Generate publication PNG figures")


class MetricStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float


class DeltaStats(BaseModel):
    delta_accuracy_mean: float
    delta_accuracy_std: float
    delta_f1_mean: float
    delta_f1_std: float
    delta_roc_auc_mean: float
    delta_roc_auc_std: float
    delta_loss_mean: float
    delta_loss_std: float


class SeedMetricItem(BaseModel):
    seed: int
    derived_seed: Optional[int] = None
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    loss: float
    confusion_matrix: Optional[List[List[int]]] = None
    model_checksum: Optional[str] = None


class ModelVariantReportSummary(BaseModel):
    algorithm: str
    variant: str
    metrics: Dict[str, MetricStats]
    deltas: Optional[DeltaStats] = None
    seed_runs: Optional[List[SeedMetricItem]] = None


class ModelRankingItem(BaseModel):
    rank: int
    algorithm: str
    variant: str
    primary_metric_name: str
    primary_metric_mean: float
    primary_metric_std: float
    f1_mean: float
    accuracy_mean: float
    roc_auc_mean: float
    loss_mean: float


class ModelRanking(BaseModel):
    primary_metric: str
    higher_is_better: bool
    rankings: List[ModelRankingItem]
    best_model_variant: str


class GlobalXAIImportanceItem(BaseModel):
    rank: int
    feature_name: str
    original_feature_name: Optional[str] = None
    mean_abs_attribution: float


class LocalXAIExplanationSummary(BaseModel):
    prediction_id: Optional[int] = None
    model_id: Optional[int] = None
    model_version: Optional[str] = None
    predicted_label: str
    probability: float
    risk_level: str
    top_positive_contributors: List[Dict[str, Any]]
    top_negative_contributors: List[Dict[str, Any]]


class XAIMethodOverlap(BaseModel):
    top_k: int
    shap_lime_overlap: int
    shap_ig_overlap: int
    lime_ig_overlap: int
    shap_lime_jaccard: float
    shap_ig_jaccard: float
    lime_ig_jaccard: float
    common_top_features: List[str]


class XAISynthesisSummary(BaseModel):
    global_shap_importance: List[GlobalXAIImportanceItem]
    local_explanations: List[LocalXAIExplanationSummary]
    cross_method_overlap: Optional[XAIMethodOverlap] = None


class ReproducibilityManifest(BaseModel):
    manifest_version: str = "1.0"
    generated_at: datetime
    experiment_id: str
    dataset_name: str
    dataset_id: Optional[int] = None
    preprocessing_artifact_version: str
    preprocessing_artifact_id: Optional[int] = None
    test_split_sample_count: int
    feature_dimension: int
    seeds: List[int]
    derived_seeds_by_seed: Dict[str, Dict[str, int]]
    model_checksums_by_seed: Dict[str, Dict[str, str]]
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    system_runtime: Dict[str, Any]


class PublicationArtifactItem(BaseModel):
    filename: str
    path_key: str
    artifact_type: str  # report_markdown, report_json, csv_table, figure_png, manifest_json
    checksum_sha256: str
    size_bytes: int


class PublicationArtifactResponse(BaseModel):
    report_id: str
    experiment_id: str
    storage_prefix: str
    artifacts: List[PublicationArtifactItem]


class ReportSummaryResponse(BaseModel):
    report_id: str
    experiment_id: str
    name: str
    status: JobStatus
    dataset_name: str
    preprocessing_artifact_version: str
    seeds: List[int]
    model_summaries: List[ModelVariantReportSummary]
    ranking: ModelRanking
    xai_synthesis: Optional[XAISynthesisSummary] = None
    created_at: datetime


class ReportResponse(BaseModel):
    id: str
    experiment_id: str
    name: str
    status: JobStatus
    configuration: Dict[str, Any]
    generated_artifact_path: Optional[str] = None
    report_metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportListItem(BaseModel):
    id: str
    experiment_id: str
    name: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportListResponse(BaseModel):
    items: List[ReportListItem]
    total: int
