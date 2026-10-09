import io
import shap
import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from datetime import datetime, timezone
from lime.lime_tabular import LimeTabularExplainer
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.errors import BadRequestException, ForbiddenException, NotFoundException
from app.models.domain import ModelRegistry, Prediction, PreprocessingArtifact, RiskLevel, User, UserRole, XAIExplanation
from app.schemas.xai import (
    FeatureAttribution,
    GlobalFeatureImportance,
    GlobalSHAPResponse,
    UnifiedXAIResponse,
    XAIResponse,
)
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService
from app.storage.local import storage


class XAIService:
    @staticmethod
    def _resolve_prediction_and_features(
        db: Session,
        prediction_id: Optional[int] = None,
        model_id: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
        user: Optional[User] = None,
    ) -> Tuple[int, Dict[str, Any], Optional[Prediction]]:
        """
        Resolves model ID and feature dictionary from either a saved prediction record
        or explicit model_id and features payload. Enforces user access control.
        """
        prediction: Optional[Prediction] = None

        if prediction_id is not None:
            prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
            if not prediction:
                raise NotFoundException(f"Prediction with ID {prediction_id} not found")

            # Authorization check
            if user and user.role != UserRole.ADMIN and prediction.user_id is not None:
                if prediction.user_id != user.id:
                    raise ForbiddenException("You do not have permission to access explanations for this prediction")

            resolved_model_id = prediction.model_id
            resolved_features = prediction.feature_values
        elif model_id is not None and features is not None:
            resolved_model_id = model_id
            resolved_features = features
        else:
            raise BadRequestException("Either prediction_id or both model_id and features must be provided")

        if not resolved_features or not isinstance(resolved_features, dict):
            raise BadRequestException("Resolved feature payload is invalid or empty")

        return resolved_model_id, resolved_features, prediction

    @staticmethod
    def _get_background_training_data(artifact: PreprocessingArtifact) -> Tuple[np.ndarray, List[str]]:
        """
        Loads the exact training splits from the model's lineage preprocessing artifact.
        """
        splits_bytes = storage.read_file(f"{artifact.artifact_path}/splits.npz")
        npz = np.load(io.BytesIO(splits_bytes), allow_pickle=True)
        X_train = npz["X_train"]
        feature_names = list(npz["feature_names"])
        return X_train, feature_names

    @staticmethod
    def _map_original_feature_name(
        transformed_name: str, num_cols: List[str], cat_cols: List[str]
    ) -> str:
        if transformed_name in num_cols:
            return transformed_name
        for col in cat_cols:
            if transformed_name.startswith(f"{col}_") or transformed_name == col:
                return col
        return transformed_name

    @classmethod
    def explain_integrated_gradients(
        cls,
        db: Session,
        prediction_id: Optional[int] = None,
        model_id: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
        user: Optional[User] = None,
    ) -> XAIResponse:
        """
        Generates feature attributions using Captum's Integrated Gradients on the PyTorch model.
        """
        res_model_id, res_features, pred_record = cls._resolve_prediction_and_features(
            db, prediction_id, model_id, features, user
        )

        model_record, model, preprocessor, artifact = PredictionService.load_model_and_preprocessor(
            db, res_model_id
        )

        df_in = pd.DataFrame([res_features])
        X_trans = PredictionService.validate_and_preprocess_features(df_in, artifact, preprocessor)
        transformed_names = artifact.feature_schema.get("feature_names", [])
        num_cols = artifact.feature_schema.get("numerical_features", [])
        cat_cols = artifact.feature_schema.get("categorical_features", [])

        # Forward function returning sigmoid probability for Class 1 (Dropout)
        def forward_func(inputs: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(model(inputs))

        model.eval()
        x_tensor = torch.tensor(X_trans, dtype=torch.float32, requires_grad=True)
        baseline = torch.zeros_like(x_tensor)

        ig = IntegratedGradients(forward_func)
        attributions, _ = ig.attribute(
            x_tensor, baselines=baseline, return_convergence_delta=True, n_steps=50
        )
        attr_vals = attributions.detach().cpu().numpy().flatten()

        with torch.no_grad():
            prob = float(torch.sigmoid(model(x_tensor)).item())

        pred_class = 1 if prob >= 0.5 else 0
        pred_label = "Dropout" if pred_class == 1 else "Not Dropout"
        risk = PredictionService.classify_risk(prob)

        # Build and rank feature attributions
        raw_items = []
        for i, val_attr in enumerate(attr_vals):
            t_name = transformed_names[i] if i < len(transformed_names) else f"feature_{i}"
            orig_name = cls._map_original_feature_name(t_name, num_cols, cat_cols)
            val_in = res_features.get(orig_name, None)
            score = float(val_attr)
            direction = "positive" if score > 0 else "negative"

            raw_items.append(
                {
                    "feature_name": t_name,
                    "original_feature_name": orig_name,
                    "feature_value": val_in,
                    "attribution": round(score, 5),
                    "direction": direction,
                    "abs_val": abs(score),
                }
            )

        # Sort descending by absolute attribution magnitude
        raw_items.sort(key=lambda x: x["abs_val"], reverse=True)
        feature_attributions: List[FeatureAttribution] = []
        for rank_idx, item in enumerate(raw_items, start=1):
            feature_attributions.append(
                FeatureAttribution(
                    feature_name=item["feature_name"],
                    original_feature_name=item["original_feature_name"],
                    feature_value=item["feature_value"],
                    attribution=item["attribution"],
                    direction=item["direction"],
                    rank=rank_idx,
                )
            )

        now = datetime.now(timezone.utc)

        # Persist explanation if linked to a saved prediction
        if pred_record:
            cls._save_explanation_record(
                db=db,
                prediction_id=pred_record.id,
                model_id=model_record.id,
                method="INTEGRATED_GRADIENTS",
                explanation_data=[f.model_dump() for f in feature_attributions],
                user_id=user.id if user else None,
            )

        return XAIResponse(
            method="INTEGRATED_GRADIENTS",
            prediction_id=pred_record.id if pred_record else None,
            model_id=model_record.id,
            model_version=model_record.version,
            probability=round(prob, 5),
            prediction_class=pred_class,
            prediction_label=pred_label,
            risk_level=risk,
            features=feature_attributions,
            created_at=now,
        )

    @classmethod
    def explain_shap(
        cls,
        db: Session,
        prediction_id: Optional[int] = None,
        model_id: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
        user: Optional[User] = None,
    ) -> XAIResponse:
        """
        Generates local SHAP explanations using KernelExplainer with background distribution
        sampled directly from the training split.
        """
        res_model_id, res_features, pred_record = cls._resolve_prediction_and_features(
            db, prediction_id, model_id, features, user
        )

        model_record, model, preprocessor, artifact = PredictionService.load_model_and_preprocessor(
            db, res_model_id
        )

        df_in = pd.DataFrame([res_features])
        X_trans = PredictionService.validate_and_preprocess_features(df_in, artifact, preprocessor)
        transformed_names = artifact.feature_schema.get("feature_names", [])
        num_cols = artifact.feature_schema.get("numerical_features", [])
        cat_cols = artifact.feature_schema.get("categorical_features", [])

        # Background dataset from model training splits
        X_train, _ = cls._get_background_training_data(artifact)
        background = shap.sample(X_train, min(50, len(X_train)), random_state=42)

        def shap_predict_fn(numpy_arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor_in = torch.tensor(numpy_arr, dtype=torch.float32)
                probs = torch.sigmoid(model(tensor_in)).cpu().numpy().flatten()
                return probs

        model.eval()
        explainer = shap.KernelExplainer(shap_predict_fn, background)
        shap_vals = explainer.shap_values(X_trans, nsamples=100, l1_reg="auto")
        shap_vals_arr = np.asarray(shap_vals).flatten()

        with torch.no_grad():
            prob = float(torch.sigmoid(model(torch.tensor(X_trans, dtype=torch.float32))).item())

        pred_class = 1 if prob >= 0.5 else 0
        pred_label = "Dropout" if pred_class == 1 else "Not Dropout"
        risk = PredictionService.classify_risk(prob)

        raw_items = []
        for i, s_val in enumerate(shap_vals_arr):
            t_name = transformed_names[i] if i < len(transformed_names) else f"feature_{i}"
            orig_name = cls._map_original_feature_name(t_name, num_cols, cat_cols)
            val_in = res_features.get(orig_name, None)
            score = float(s_val)
            direction = "positive" if score > 0 else "negative"

            raw_items.append(
                {
                    "feature_name": t_name,
                    "original_feature_name": orig_name,
                    "feature_value": val_in,
                    "attribution": round(score, 5),
                    "direction": direction,
                    "abs_val": abs(score),
                }
            )

        raw_items.sort(key=lambda x: x["abs_val"], reverse=True)
        feature_attributions: List[FeatureAttribution] = []
        for rank_idx, item in enumerate(raw_items, start=1):
            feature_attributions.append(
                FeatureAttribution(
                    feature_name=item["feature_name"],
                    original_feature_name=item["original_feature_name"],
                    feature_value=item["feature_value"],
                    attribution=item["attribution"],
                    direction=item["direction"],
                    rank=rank_idx,
                )
            )

        now = datetime.now(timezone.utc)

        if pred_record:
            cls._save_explanation_record(
                db=db,
                prediction_id=pred_record.id,
                model_id=model_record.id,
                method="SHAP",
                explanation_data=[f.model_dump() for f in feature_attributions],
                user_id=user.id if user else None,
            )

        return XAIResponse(
            method="SHAP",
            prediction_id=pred_record.id if pred_record else None,
            model_id=model_record.id,
            model_version=model_record.version,
            probability=round(prob, 5),
            prediction_class=pred_class,
            prediction_label=pred_label,
            risk_level=risk,
            features=feature_attributions,
            created_at=now,
        )

    @classmethod
    def explain_global_shap(
        cls,
        db: Session,
        model_id: int,
        max_samples: int = 50,
        user: Optional[User] = None,
    ) -> GlobalSHAPResponse:
        """
        Computes global feature importance by averaging absolute SHAP values across
        actual validation/training split records.
        """
        model_record, model, preprocessor, artifact = PredictionService.load_model_and_preprocessor(
            db, model_id
        )

        X_train, feature_names = cls._get_background_training_data(artifact)
        background = shap.sample(X_train, min(30, len(X_train)), random_state=42)
        eval_samples = X_train[: min(max_samples, len(X_train))]

        def shap_predict_fn(numpy_arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor_in = torch.tensor(numpy_arr, dtype=torch.float32)
                return torch.sigmoid(model(tensor_in)).cpu().numpy().flatten()

        model.eval()
        explainer = shap.KernelExplainer(shap_predict_fn, background)
        shap_matrix = explainer.shap_values(eval_samples, nsamples=50, l1_reg="auto")
        shap_matrix = np.asarray(shap_matrix)

        mean_abs_scores = np.mean(np.abs(shap_matrix), axis=0)

        importance_list = []
        for i, score in enumerate(mean_abs_scores):
            f_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            importance_list.append(
                {"feature_name": f_name, "mean_abs_shap": round(float(score), 5)}
            )

        importance_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        ranked_importance: List[GlobalFeatureImportance] = []
        for rank_idx, item in enumerate(importance_list, start=1):
            ranked_importance.append(
                GlobalFeatureImportance(
                    feature_name=item["feature_name"],
                    mean_abs_shap=item["mean_abs_shap"],
                    rank=rank_idx,
                )
            )

        return GlobalSHAPResponse(
            model_id=model_record.id,
            model_version=model_record.version,
            samples_evaluated=len(eval_samples),
            feature_importance=ranked_importance,
        )

    @classmethod
    def explain_lime(
        cls,
        db: Session,
        prediction_id: Optional[int] = None,
        model_id: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
        user: Optional[User] = None,
    ) -> XAIResponse:
        """
        Generates local LIME feature weights using LimeTabularExplainer on transformed input space.
        """
        res_model_id, res_features, pred_record = cls._resolve_prediction_and_features(
            db, prediction_id, model_id, features, user
        )

        model_record, model, preprocessor, artifact = PredictionService.load_model_and_preprocessor(
            db, res_model_id
        )

        df_in = pd.DataFrame([res_features])
        X_trans = PredictionService.validate_and_preprocess_features(df_in, artifact, preprocessor)
        transformed_names = artifact.feature_schema.get("feature_names", [])
        num_cols = artifact.feature_schema.get("numerical_features", [])
        cat_cols = artifact.feature_schema.get("categorical_features", [])

        X_train, _ = cls._get_background_training_data(artifact)

        def lime_predict_fn(numpy_arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor_in = torch.tensor(numpy_arr, dtype=torch.float32)
                p = torch.sigmoid(model(tensor_in)).cpu().numpy()
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                return np.hstack([1.0 - p, p])

        model.eval()
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=transformed_names,
            class_names=["Not Dropout", "Dropout"],
            mode="classification",
            random_state=42,
        )

        exp = explainer.explain_instance(
            data_row=X_trans.flatten(),
            predict_fn=lime_predict_fn,
            num_features=len(transformed_names),
        )

        with torch.no_grad():
            prob = float(torch.sigmoid(model(torch.tensor(X_trans, dtype=torch.float32))).item())

        pred_class = 1 if prob >= 0.5 else 0
        pred_label = "Dropout" if pred_class == 1 else "Not Dropout"
        risk = PredictionService.classify_risk(prob)

        # Extract map for class 1 (Dropout)
        exp_map = exp.as_map().get(1, [])
        raw_items = []

        for feat_idx, weight in exp_map:
            t_name = (
                transformed_names[feat_idx]
                if feat_idx < len(transformed_names)
                else f"feature_{feat_idx}"
            )
            orig_name = cls._map_original_feature_name(t_name, num_cols, cat_cols)
            val_in = res_features.get(orig_name, None)
            score = float(weight)
            direction = "positive" if score > 0 else "negative"

            raw_items.append(
                {
                    "feature_name": t_name,
                    "original_feature_name": orig_name,
                    "feature_value": val_in,
                    "attribution": round(score, 5),
                    "direction": direction,
                    "abs_val": abs(score),
                }
            )

        raw_items.sort(key=lambda x: x["abs_val"], reverse=True)
        feature_attributions: List[FeatureAttribution] = []
        for rank_idx, item in enumerate(raw_items, start=1):
            feature_attributions.append(
                FeatureAttribution(
                    feature_name=item["feature_name"],
                    original_feature_name=item["original_feature_name"],
                    feature_value=item["feature_value"],
                    attribution=item["attribution"],
                    direction=item["direction"],
                    rank=rank_idx,
                )
            )

        now = datetime.now(timezone.utc)

        if pred_record:
            cls._save_explanation_record(
                db=db,
                prediction_id=pred_record.id,
                model_id=model_record.id,
                method="LIME",
                explanation_data=[f.model_dump() for f in feature_attributions],
                user_id=user.id if user else None,
            )

        return XAIResponse(
            method="LIME",
            prediction_id=pred_record.id if pred_record else None,
            model_id=model_record.id,
            model_version=model_record.version,
            probability=round(prob, 5),
            prediction_class=pred_class,
            prediction_label=pred_label,
            risk_level=risk,
            features=feature_attributions,
            created_at=now,
        )

    @classmethod
    def get_unified_explanation(
        cls,
        db: Session,
        prediction_id: int,
        user: Optional[User] = None,
    ) -> UnifiedXAIResponse:
        """
        Retrieves unified multi-method explainability suite (SHAP, LIME, Integrated Gradients)
        for a saved student dropout prediction.
        """
        _, _, pred_record = cls._resolve_prediction_and_features(
            db, prediction_id=prediction_id, user=user
        )

        model_record = ModelService.get_model(db, pred_record.model_id)

        # Run all three explanation methods
        ig_res = cls.explain_integrated_gradients(db, prediction_id=prediction_id, user=user)
        shap_res = cls.explain_shap(db, prediction_id=prediction_id, user=user)
        lime_res = cls.explain_lime(db, prediction_id=prediction_id, user=user)

        # Update consolidated explanation in Prediction model
        pred_record.explanation = {
            "shap_top_features": [f.model_dump() for f in shap_res.features[:5]],
            "lime_top_features": [f.model_dump() for f in lime_res.features[:5]],
            "ig_top_features": [f.model_dump() for f in ig_res.features[:5]],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        db.commit()

        prediction_meta = {
            "probability": pred_record.probability,
            "prediction_class": pred_record.prediction_class,
            "prediction_label": pred_record.prediction_label,
            "risk_level": pred_record.risk_level.value if pred_record.risk_level else None,
            "created_at": pred_record.created_at.isoformat() if pred_record.created_at else None,
        }

        return UnifiedXAIResponse(
            prediction_id=pred_record.id,
            model_id=model_record.id,
            model_version=model_record.version,
            prediction=prediction_meta,
            explanations={
                "shap": shap_res.model_dump(),
                "lime": lime_res.model_dump(),
                "integrated_gradients": ig_res.model_dump(),
            },
        )

    @staticmethod
    def _save_explanation_record(
        db: Session,
        prediction_id: int,
        model_id: int,
        method: str,
        explanation_data: Any,
        user_id: Optional[int] = None,
    ) -> XAIExplanation:
        """
        Persists explanation artifact record in database.
        """
        existing = (
            db.query(XAIExplanation)
            .filter(
                XAIExplanation.prediction_id == prediction_id,
                XAIExplanation.method == method,
            )
            .first()
        )

        if existing:
            existing.explanation_data = explanation_data
            existing.created_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(existing)
            return existing

        explanation = XAIExplanation(
            prediction_id=prediction_id,
            model_id=model_id,
            method=method,
            explanation_data=explanation_data,
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
        )
        db.add(explanation)
        db.commit()
        db.refresh(explanation)
        return explanation
