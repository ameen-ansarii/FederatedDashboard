import hashlib
import io
import joblib
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.errors import BadRequestException, NotFoundException
from app.ml.models import DropoutMLP
from app.models.domain import ModelRegistry, Prediction, PreprocessingArtifact, RiskLevel, User
from app.schemas.prediction import (
    BatchPredictionItemResult,
    BatchPredictionSummary,
    SinglePredictionResponse,
)
from app.services.model_service import ModelService
from app.services.preprocessing_service import PreprocessingService
from app.storage.local import storage


class PredictionService:
    @staticmethod
    def classify_risk(probability: float) -> RiskLevel:
        """
        Maps probability to risk category:
        - LOW: probability < 0.40
        - MEDIUM: 0.40 <= probability < 0.70
        - HIGH: probability >= 0.70
        """
        if probability < 0.40:
            return RiskLevel.LOW
        elif probability < 0.70:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    @staticmethod
    def load_model_and_preprocessor(
        db: Session, model_id: int
    ) -> Tuple[ModelRegistry, DropoutMLP, Any, PreprocessingArtifact]:
        """
        Retrieves model, verifies its weight checksum, loads PyTorch architecture/weights,
        and retrieves its exact lineage-linked preprocessing pipeline.
        """
        model_record = ModelService.get_model(db, model_id)

        # 1. Verify weights integrity
        weights_bytes = storage.read_file(f"{model_record.artifact_path}/model.pt")
        actual_checksum = hashlib.sha256(weights_bytes).hexdigest()
        if actual_checksum != model_record.checksum:
            raise BadRequestException("Model weights failed cryptographic integrity check (checksum mismatch)")

        # 2. Load PyTorch model in evaluation mode
        model = ModelService.load_pytorch_model(model_record)
        model.eval()

        # 3. Retrieve exact PreprocessingArtifact from model lineage
        lineage = model_record.lineage or {}
        artifact_id = lineage.get("preprocessing_artifact_id")
        if not artifact_id:
            raise BadRequestException(f"Model ID {model_id} does not contain linked preprocessing lineage metadata")

        artifact = PreprocessingService.get_artifact(db, artifact_id)

        # 4. Load preprocessor transformer
        pipeline_bytes = storage.read_file(f"{artifact.artifact_path}/preprocessor.joblib")
        preprocessor = joblib.load(io.BytesIO(pipeline_bytes))

        return model_record, model, preprocessor, artifact

    @staticmethod
    def validate_and_preprocess_features(
        raw_df: pd.DataFrame, artifact: PreprocessingArtifact, preprocessor: Any
    ) -> np.ndarray:
        """
        Validates presence and schema of raw input features against the artifact's feature schema,
        then transforms features using the pre-fitted preprocessor.
        """
        if raw_df.empty:
            raise BadRequestException("Input feature payload contains no records")

        feature_schema = artifact.feature_schema or {}
        num_cols = feature_schema.get("numerical_features", [])
        cat_cols = feature_schema.get("categorical_features", [])
        expected_features = set(num_cols + cat_cols)

        # Check for missing feature columns
        missing = expected_features - set(raw_df.columns)
        if missing:
            raise BadRequestException(f"Input is missing required features: {sorted(list(missing))}")

        # Ensure correct column ordering for transformer
        ordered_df = raw_df[num_cols + cat_cols]

        try:
            transformed = preprocessor.transform(ordered_df)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            return np.asarray(transformed, dtype=np.float32)
        except Exception as e:
            raise BadRequestException(f"Feature transformation failed: {str(e)}")

    @classmethod
    def predict_single(
        cls,
        db: Session,
        model_id: int,
        features: Dict[str, Any],
        save_to_history: bool = True,
        user: Optional[User] = None,
    ) -> SinglePredictionResponse:
        """
        Performs synchronous single-instance student dropout inference.
        """
        if not features or not isinstance(features, dict):
            raise BadRequestException("Features must be a non-empty dictionary of feature values")

        model_record, model, preprocessor, artifact = cls.load_model_and_preprocessor(db, model_id)

        # Build single-row DataFrame
        df_input = pd.DataFrame([features])
        X_trans = cls.validate_and_preprocess_features(df_input, artifact, preprocessor)

        # PyTorch forward pass under torch.no_grad()
        with torch.no_grad():
            tensor_in = torch.tensor(X_trans, dtype=torch.float32)
            logits = model(tensor_in)
            prob = float(torch.sigmoid(logits).item())

        # Determine class, label, and risk band
        pred_class = 1 if prob >= 0.5 else 0
        pred_label = "Dropout" if pred_class == 1 else "Not Dropout"
        risk = cls.classify_risk(prob)
        created_at = datetime.now(timezone.utc)

        prediction_record_id: Optional[int] = None
        if save_to_history:
            # Data minimization: store sanitized copy of input features
            clean_features = {k: v for k, v in features.items() if not k.startswith("_")}
            pred_db = Prediction(
                model_id=model_record.id,
                probability=round(prob, 5),
                prediction_class=pred_class,
                prediction_label=pred_label,
                risk_level=risk,
                feature_values=clean_features,
                user_id=user.id if user else None,
                created_at=created_at,
            )
            db.add(pred_db)
            db.commit()
            db.refresh(pred_db)
            prediction_record_id = pred_db.id

        return SinglePredictionResponse(
            id=prediction_record_id,
            model_id=model_record.id,
            model_version=model_record.version,
            probability=round(prob, 5),
            prediction_class=pred_class,
            prediction_label=pred_label,
            risk_level=risk,
            created_at=created_at,
        )

    @classmethod
    def predict_batch(
        cls,
        db: Session,
        model_id: int,
        records: List[Dict[str, Any]],
        batch_job_id: Optional[str] = None,
        save_to_history: bool = False,
        user_id: Optional[int] = None,
    ) -> Tuple[BatchPredictionSummary, List[BatchPredictionItemResult]]:
        """
        Executes vectorized batch inference across multiple student records.
        """
        if not records:
            raise BadRequestException("Records list cannot be empty")

        model_record, model, preprocessor, artifact = cls.load_model_and_preprocessor(db, model_id)

        df_input = pd.DataFrame(records)
        X_trans = cls.validate_and_preprocess_features(df_input, artifact, preprocessor)

        with torch.no_grad():
            tensor_in = torch.tensor(X_trans, dtype=torch.float32)
            logits = model(tensor_in)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([float(probs)])

        results: List[BatchPredictionItemResult] = []
        low_count = 0
        med_count = 0
        high_count = 0
        db_records_to_insert = []
        now = datetime.now(timezone.utc)

        for i, prob_val in enumerate(probs):
            prob = float(prob_val)
            pred_class = 1 if prob >= 0.5 else 0
            pred_label = "Dropout" if pred_class == 1 else "Not Dropout"
            risk = cls.classify_risk(prob)

            if risk == RiskLevel.LOW:
                low_count += 1
            elif risk == RiskLevel.MEDIUM:
                med_count += 1
            else:
                high_count += 1

            results.append(
                BatchPredictionItemResult(
                    index=i,
                    probability=round(prob, 5),
                    prediction_class=pred_class,
                    prediction_label=pred_label,
                    risk_level=risk,
                )
            )

            if save_to_history:
                clean_features = {k: v for k, v in records[i].items() if not k.startswith("_")}
                db_records_to_insert.append(
                    Prediction(
                        model_id=model_record.id,
                        probability=round(prob, 5),
                        prediction_class=pred_class,
                        prediction_label=pred_label,
                        risk_level=risk,
                        feature_values=clean_features,
                        batch_job_id=batch_job_id,
                        user_id=user_id,
                        created_at=now,
                    )
                )

        if db_records_to_insert:
            db.bulk_save_objects(db_records_to_insert)
            db.commit()

        summary = BatchPredictionSummary(
            total_samples=len(results),
            low_risk_count=low_count,
            medium_risk_count=med_count,
            high_risk_count=high_count,
            mean_probability=round(float(np.mean(probs)), 5) if len(probs) > 0 else 0.0,
        )

        return summary, results

    @staticmethod
    def list_history(
        db: Session,
        skip: int = 0,
        limit: int = 50,
        model_id: Optional[int] = None,
        risk_level: Optional[RiskLevel] = None,
        user_id: Optional[int] = None,
    ) -> Tuple[List[Prediction], int]:
        """
        Queries paginated prediction history with optional filters.
        """
        query = db.query(Prediction)
        if model_id is not None:
            query = query.filter(Prediction.model_id == model_id)
        if risk_level is not None:
            query = query.filter(Prediction.risk_level == risk_level)
        if user_id is not None:
            query = query.filter(Prediction.user_id == user_id)

        total = query.count()
        items = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
        return items, total

    @staticmethod
    def get_prediction_by_id(db: Session, prediction_id: int) -> Prediction:
        pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not pred:
            raise NotFoundException(f"Prediction record with ID {prediction_id} not found")
        return pred
