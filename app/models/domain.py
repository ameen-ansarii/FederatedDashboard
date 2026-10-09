import enum
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship
from app.core.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    RESEARCHER = "RESEARCHER"
    CLIENT = "CLIENT"
    VIEWER = "VIEWER"


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RiskLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(Enum(UserRole), default=UserRole.RESEARCHER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    jobs = relationship("Job", back_populates="user")
    datasets = relationship("Dataset", back_populates="user")
    models = relationship("ModelRegistry", back_populates="user")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, index=True)  # UUID string
    type = Column(String(100), nullable=False)  # e.g., PREPROCESSING, CENTRALIZED_TRAINING, FEDERATED_TRAINING, PREDICTION, XAI
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    progress = Column(Float, default=0.0, nullable=False)  # 0 to 100
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    user = relationship("User", back_populates="jobs")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    target_column = Column(String(100), nullable=True, default="Target")
    checksum = Column(String(64), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    user = relationship("User", back_populates="datasets")
    preprocessing_artifacts = relationship("PreprocessingArtifact", back_populates="dataset")


class PreprocessingArtifact(Base):
    __tablename__ = "preprocessing_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version = Column(String(50), nullable=False)
    artifact_path = Column(String(512), nullable=False)
    feature_schema = Column(JSON, nullable=False)
    target_mapping = Column(JSON, nullable=True)
    split_summary = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    dataset = relationship("Dataset", back_populates="preprocessing_artifacts")


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False, unique=True, index=True)
    framework = Column(String(50), default="PyTorch", nullable=False)
    artifact_path = Column(String(512), nullable=False)
    metrics = Column(JSON, nullable=False)  # accuracy, precision, recall, f1, roc_auc
    lineage = Column(JSON, nullable=False)  # dataset_id, preprocessing_artifact_id, config, seed
    checksum = Column(String(64), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    user = relationship("User", back_populates="models")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    artifact_id = Column(Integer, ForeignKey("preprocessing_artifacts.id", ondelete="CASCADE"), nullable=False)
    config = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    artifact = relationship("PreprocessingArtifact")
    user = relationship("User")
    training_runs = relationship("TrainingRun", back_populates="experiment")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String(36), primary_key=True, index=True)
    type = Column(String(50), nullable=False)  # CENTRALIZED, FEDAVG, FEDPROX
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=True)
    history = Column(JSON, nullable=True)
    experiment_id = Column(String(36), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    experiment = relationship("Experiment", back_populates="training_runs")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model_registry.id", ondelete="SET NULL"), nullable=True)
    probability = Column(Float, nullable=False)
    prediction_class = Column(Integer, nullable=True)  # 0 or 1
    prediction_label = Column(String(50), nullable=True)  # e.g., "Dropout", "Not Dropout"
    risk_level = Column(Enum(RiskLevel), nullable=False)
    feature_values = Column(JSON, nullable=False)
    explanation = Column(JSON, nullable=True)
    batch_job_id = Column(String(36), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    model = relationship("ModelRegistry")
    explanations = relationship("XAIExplanation", back_populates="prediction", cascade="all, delete-orphan")


class XAIExplanation(Base):
    __tablename__ = "xai_explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=True, index=True)
    model_id = Column(Integer, ForeignKey("model_registry.id", ondelete="SET NULL"), nullable=True, index=True)
    method = Column(String(50), nullable=False)  # SHAP, LIME, INTEGRATED_GRADIENTS, GLOBAL_SHAP
    explanation_data = Column(JSON, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    prediction = relationship("Prediction", back_populates="explanations")
    model = relationship("ModelRegistry")



class FederatedClient(Base):
    __tablename__ = "federated_clients"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="IDLE", nullable=False)  # IDLE, TRAINING, OFFLINE
    ip_address = Column(String(50), nullable=True)
    last_heartbeat = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class PublicationReport(Base):
    __tablename__ = "publication_reports"

    id = Column(String(36), primary_key=True, index=True)  # UUID string
    experiment_id = Column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    configuration = Column(JSON, nullable=False)
    generated_artifact_path = Column(String(512), nullable=True)
    report_metadata = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    experiment = relationship("Experiment")
    user = relationship("User")

