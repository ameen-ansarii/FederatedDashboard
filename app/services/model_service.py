import hashlib
import io
import json
import uuid
import torch
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.errors import NotFoundException
from app.ml.models import DropoutMLP
from app.models.domain import ModelRegistry, PreprocessingArtifact, User
from app.storage.local import storage


class ModelService:
    @staticmethod
    def save_model(
        db: Session,
        name: str,
        model: DropoutMLP,
        artifact: PreprocessingArtifact,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        user: Optional[User] = None,
    ) -> ModelRegistry:
        """
        Persists a trained PyTorch model state dict and registers immutable model lineage in DB.
        """
        timestamp = int(pd.Timestamp.now(tz="UTC").timestamp())
        version_str = f"mod_v_{artifact.id}_{timestamp}_{uuid.uuid4().hex[:6]}"
        model_dir = f"artifacts/models/{version_str}"

        # 1. Save PyTorch state dict binary
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        weight_bytes = buffer.getvalue()
        checksum = hashlib.sha256(weight_bytes).hexdigest()

        weights_rel_path = f"{model_dir}/model.pt"
        storage.save_file(weights_rel_path, weight_bytes)

        # 2. Lineage details
        lineage = {
            "dataset_id": artifact.dataset_id,
            "preprocessing_artifact_id": artifact.id,
            "preprocessing_version": artifact.version,
            "feature_schema": artifact.feature_schema,
            "hyperparameters": hyperparameters,
            "split_summary": artifact.split_summary,
            "seed": hyperparameters.get("random_state", 42),
        }

        # 3. Save metadata JSON
        metadata = {
            "name": name,
            "version": version_str,
            "framework": "PyTorch",
            "checksum": checksum,
            "metrics": metrics,
            "lineage": lineage,
        }
        meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        storage.save_file(f"{model_dir}/metadata.json", meta_bytes)

        # 4. Save DB Record
        db_model = ModelRegistry(
            name=name,
            version=version_str,
            framework="PyTorch",
            artifact_path=model_dir,
            metrics=metrics,
            lineage=lineage,
            checksum=checksum,
            user_id=user.id if user else None,
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model

    @staticmethod
    def get_model(db: Session, model_id: int) -> ModelRegistry:
        model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
        if not model:
            raise NotFoundException(f"Model with ID {model_id} not found in model registry")
        return model

    @staticmethod
    def list_models(db: Session, skip: int = 0, limit: int = 100) -> List[ModelRegistry]:
        return db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def load_pytorch_model(model_record: ModelRegistry) -> DropoutMLP:
        """
        Loads PyTorch model architecture and populates saved state dict weights.
        """
        weights_bytes = storage.read_file(f"{model_record.artifact_path}/model.pt")
        lineage = model_record.lineage
        hyperparams = lineage.get("hyperparameters", {})
        feature_schema = lineage.get("feature_schema", {})
        input_dim = feature_schema.get("feature_count", 0)

        model = DropoutMLP(
            input_dim=input_dim,
            hidden_dims=hyperparams.get("hidden_dims", [64, 32]),
            dropout_rate=hyperparams.get("dropout_rate", 0.2),
        )
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
