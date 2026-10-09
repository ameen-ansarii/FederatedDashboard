from typing import Any, Dict, List, Optional, Set
from sqlalchemy.orm import Session
import numpy as np

from app.models.domain import ModelRegistry, Prediction, PreprocessingArtifact, User
from app.schemas.report import (
    GlobalXAIImportanceItem,
    LocalXAIExplanationSummary,
    XAIMethodOverlap,
    XAISynthesisSummary,
)
from app.services.model_service import ModelService
from app.services.xai_service import XAIService


class XAIReportService:
    """
    Synthesizes Global and Local Explainable AI metrics (SHAP, LIME, Integrated Gradients)
    for publication reports and calculates empirical cross-method feature ranking overlap.
    """

    @classmethod
    def synthesize_xai(
        cls,
        db: Session,
        model_id: int,
        artifact: PreprocessingArtifact,
        user: Optional[User] = None,
        top_k: int = 5,
    ) -> XAISynthesisSummary:
        """
        Builds a comprehensive XAI synthesis for a chosen registered model:
        1. Computes Global SHAP feature importances across the background training split.
        2. Retrieves or generates representative Local explanations (SHAP, LIME, IG).
        3. Computes cross-method feature ranking overlap and Jaccard similarity.
        """
        # 1. Global SHAP Feature Importance
        global_shap_res = XAIService.explain_global_shap(
            db=db,
            model_id=model_id,
            max_samples=25,
            user=user,
        )

        global_items: List[GlobalXAIImportanceItem] = []
        for item in global_shap_res.feature_importance[:10]:
            global_items.append(
                GlobalXAIImportanceItem(
                    rank=item.rank,
                    feature_name=item.feature_name,
                    original_feature_name=getattr(item, "original_feature_name", None),
                    mean_abs_attribution=item.mean_abs_shap,
                )
            )

        # 2. Local Explanations for available or representative predictions
        local_summaries: List[LocalXAIExplanationSummary] = []
        preds = db.query(Prediction).filter(Prediction.model_id == model_id).limit(3).all()

        # If no prediction exists for this model in DB, create one on a test sample for synthesis
        if not preds:
            from app.services.prediction_service import PredictionService
            from app.schemas.prediction import SinglePredictionRequest

            # Use realistic sample values from artifact
            num_cols = artifact.feature_schema.get("numerical_features", [])
            cat_cols = artifact.feature_schema.get("categorical_features", [])
            sample_feats = {
                "total_vle_clicks": 45,
                "avg_assessment_score": 35.0,
                "studied_credits": 60,
                "num_of_prev_attempts": 1,
                "code_module": "BBB",
                "code_presentation": "2014J",
                "gender": "M",
                "highest_education": "A Level or Equivalent",
                "imd_band": "20-30%",
                "age_band": "0-35",
                "disability": "N",
            }
            pred_res = PredictionService.predict_single(db=db, model_id=model_id, features=sample_feats, user=user)
            p_rec = db.query(Prediction).filter(Prediction.id == pred_res.id).first()
            if p_rec:
                preds = [p_rec]

        overlap: Optional[XAIMethodOverlap] = None

        for p in preds:
            # Generate / fetch local explanations using unified XAI
            unified = XAIService.get_unified_explanation(db=db, prediction_id=p.id, user=user)
            shap_feats = unified.explanations.get("shap", {}).get("features", [])
            lime_feats = unified.explanations.get("lime", {}).get("features", [])
            ig_feats = unified.explanations.get("integrated_gradients", {}).get("features", [])

            # Extract positive and negative attributions
            pos_contribs = []
            neg_contribs = []
            for f in shap_feats:
                entry = {
                    "feature": f.get("feature_name"),
                    "original_feature": f.get("original_feature_name"),
                    "attribution": f.get("attribution"),
                    "direction": f.get("direction"),
                }
                if f.get("direction") == "positive":
                    pos_contribs.append(entry)
                else:
                    neg_contribs.append(entry)

            local_summaries.append(
                LocalXAIExplanationSummary(
                    prediction_id=p.id,
                    model_id=model_id,
                    model_version=unified.model_version,
                    predicted_label=p.prediction_label or ("Dropout" if p.probability >= 0.5 else "Not Dropout"),
                    probability=round(p.probability, 4),
                    risk_level=p.risk_level.value if p.risk_level else "MEDIUM",
                    top_positive_contributors=pos_contribs[:3],
                    top_negative_contributors=neg_contribs[:3],
                )
            )

            # Compute cross-method overlap for this prediction if not already computed
            if overlap is None and shap_feats and lime_feats and ig_feats:
                overlap = cls.calculate_cross_method_overlap(
                    shap_features=shap_feats,
                    lime_features=lime_feats,
                    ig_features=ig_feats,
                    top_k=top_k,
                )

        return XAISynthesisSummary(
            global_shap_importance=global_items,
            local_explanations=local_summaries,
            cross_method_overlap=overlap,
        )

    @classmethod
    def calculate_cross_method_overlap(
        cls,
        shap_features: List[Dict[str, Any]],
        lime_features: List[Dict[str, Any]],
        ig_features: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> XAIMethodOverlap:
        """
        Computes the empirical feature-ranking overlap and Jaccard similarity
        between the top-k attributed features of SHAP, LIME, and Integrated Gradients.

        Jaccard Similarity: J(A, B) = |A ∩ B| / |A ∪ B|
        """
        shap_top_set: Set[str] = {f["feature_name"] for f in shap_features[:top_k]}
        lime_top_set: Set[str] = {f["feature_name"] for f in lime_features[:top_k]}
        ig_top_set: Set[str] = {f["feature_name"] for f in ig_features[:top_k]}

        def jaccard(s1: Set[str], s2: Set[str]) -> float:
            union = s1.union(s2)
            if not union:
                return 0.0
            return round(len(s1.intersection(s2)) / len(union), 4)

        common_all = sorted(list(shap_top_set.intersection(lime_top_set).intersection(ig_top_set)))

        return XAIMethodOverlap(
            top_k=top_k,
            shap_lime_overlap=len(shap_top_set.intersection(lime_top_set)),
            shap_ig_overlap=len(shap_top_set.intersection(ig_top_set)),
            lime_ig_overlap=len(lime_top_set.intersection(ig_top_set)),
            shap_lime_jaccard=jaccard(shap_top_set, lime_top_set),
            shap_ig_jaccard=jaccard(shap_top_set, ig_top_set),
            lime_ig_jaccard=jaccard(lime_top_set, ig_top_set),
            common_top_features=common_all,
        )
