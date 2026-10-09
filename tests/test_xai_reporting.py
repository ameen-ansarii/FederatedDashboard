import io
import pytest
import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.models.domain import ModelRegistry, PreprocessingArtifact, User
from app.schemas.report import XAIMethodOverlap
from app.services.xai_report_service import XAIReportService
from app.storage.local import storage


def create_mock_model_and_artifact(db: Session, client: TestClient):
    # 1. Upload dataset
    df = pd.DataFrame({
        "total_vle_clicks": np.random.randint(10, 500, size=60),
        "avg_assessment_score": np.random.uniform(20.0, 95.0, size=60),
        "studied_credits": np.random.choice([30, 60, 90, 120], size=60),
        "num_of_prev_attempts": np.random.choice([0, 1, 2], size=60),
        "code_module": np.random.choice(["AAA", "BBB"], size=60),
        "code_presentation": np.random.choice(["2013J", "2014J"], size=60),
        "gender": np.random.choice(["M", "F"], size=60),
        "highest_education": np.random.choice(["A Level", "HE Qualification"], size=60),
        "imd_band": np.random.choice(["10-20%", "20-30%"], size=60),
        "age_band": np.random.choice(["0-35", "35-55"], size=60),
        "disability": np.random.choice(["N", "Y"], size=60),
        "final_result": np.random.choice(["Dropout", "Pass"], size=60, p=[0.4, 0.6]),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)

    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "OULAD XAI Benchmark", "target_column": "final_result"},
        files={"file": ("oulad_xai.csv", buf.getvalue(), "text/csv")},
    )
    dataset_id = up_res.json()["id"]

    client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        json={
            "target_column": "final_result",
            "positive_class_val": "Dropout",
            "scale_features": True,
            "val_size": 0.20,
            "test_size": 0.20,
            "random_state": 42,
        },
    )
    artifacts = client.get(f"/api/v1/datasets/{dataset_id}/artifacts").json()
    artifact_id = artifacts[0]["id"]
    artifact = db.query(PreprocessingArtifact).filter(PreprocessingArtifact.id == artifact_id).first()

    # Create & register a PyTorch model
    from app.ml.models import DropoutMLP
    import hashlib
    feat_dim = artifact.feature_schema.get("feature_count", len(artifact.feature_schema.get("feature_names", [])))
    model = DropoutMLP(input_dim=feat_dim, hidden_dims=[16, 8], dropout_rate=0.1)
    
    # Save model weights to storage (directory/model.pt convention)
    m_buf = io.BytesIO()
    torch.save(model.state_dict(), m_buf)
    model_bytes = m_buf.getvalue()
    model_dir = f"models/xai_test_model_{artifact_id}"
    storage.save_file(f"{model_dir}/model.pt", model_bytes)
    model_checksum = hashlib.sha256(model_bytes).hexdigest()

    reg_model = ModelRegistry(
        name="DropoutMLP_XAI_Test",
        version=f"v_xai_test_{artifact_id}",
        framework="PyTorch",
        artifact_path=model_dir,
        metrics={"accuracy": 0.65, "f1": 0.50, "precision": 0.45, "recall": 0.55, "roc_auc": 0.68},
        lineage={
            "dataset_id": dataset_id,
            "preprocessing_artifact_id": artifact_id,
            "seed": 42,
            "feature_schema": {"feature_count": feat_dim},
            "hyperparameters": {"hidden_dims": [16, 8], "dropout_rate": 0.1},
        },
        checksum=model_checksum,
    )
    db.add(reg_model)
    db.commit()
    db.refresh(reg_model)

    return reg_model.id, artifact


def test_cross_method_overlap_calculation():
    """
    TEST 8: Cross-Method Feature Overlap & Jaccard Similarity Calculation
    """
    shap_f = [
        {"feature_name": "total_vle_clicks", "attribution": 0.35},
        {"feature_name": "avg_assessment_score", "attribution": 0.25},
        {"feature_name": "studied_credits", "attribution": -0.15},
        {"feature_name": "gender_M", "attribution": 0.10},
        {"feature_name": "age_band_0-35", "attribution": -0.05},
    ]
    lime_f = [
        {"feature_name": "total_vle_clicks", "attribution": 0.40},
        {"feature_name": "avg_assessment_score", "attribution": 0.20},
        {"feature_name": "studied_credits", "attribution": -0.12},
        {"feature_name": "imd_band_10-20%", "attribution": 0.08},
        {"feature_name": "code_module_BBB", "attribution": -0.04},
    ]
    ig_f = [
        {"feature_name": "total_vle_clicks", "attribution": 0.30},
        {"feature_name": "avg_assessment_score", "attribution": 0.22},
        {"feature_name": "studied_credits", "attribution": -0.18},
        {"feature_name": "gender_M", "attribution": 0.09},
        {"feature_name": "imd_band_10-20%", "attribution": 0.07},
    ]

    overlap = XAIReportService.calculate_cross_method_overlap(
        shap_features=shap_f,
        lime_features=lime_f,
        ig_features=ig_f,
        top_k=5,
    )

    assert overlap.top_k == 5
    # Common top features across all 3: total_vle_clicks, avg_assessment_score, studied_credits (3)
    assert set(overlap.common_top_features) == {"total_vle_clicks", "avg_assessment_score", "studied_credits"}
    # SHAP and LIME overlap: 3 features
    assert overlap.shap_lime_overlap == 3
    # Jaccard SHAP & LIME: 3 / (5 + 5 - 3) = 3/7 ≈ 0.4286
    assert abs(overlap.shap_lime_jaccard - round(3/7, 4)) < 1e-4
    # SHAP and IG overlap: 4 features (total_vle_clicks, avg_assessment_score, studied_credits, gender_M)
    assert overlap.shap_ig_overlap == 4
    # Jaccard SHAP & IG: 4 / (5 + 5 - 4) = 4/6 ≈ 0.6667
    assert abs(overlap.shap_ig_jaccard - round(4/6, 4)) < 1e-4


def test_xai_synthesis_end_to_end(db: Session, client: TestClient):
    """
    TEST 7: XAI Synthesis with Global SHAP and Local Explanations
    """
    model_id, artifact = create_mock_model_and_artifact(db, client)

    synthesis = XAIReportService.synthesize_xai(
        db=db,
        model_id=model_id,
        artifact=artifact,
        top_k=5,
    )

    assert len(synthesis.global_shap_importance) > 0
    assert synthesis.global_shap_importance[0].rank == 1
    assert synthesis.global_shap_importance[0].mean_abs_attribution >= 0.0

    assert len(synthesis.local_explanations) > 0
    local_exp = synthesis.local_explanations[0]
    assert local_exp.model_id == model_id
    assert local_exp.predicted_label in ["Dropout", "Not Dropout"]
    assert 0.0 <= local_exp.probability <= 1.0
    assert local_exp.risk_level in ["LOW", "MEDIUM", "HIGH"]

    if synthesis.cross_method_overlap:
        assert synthesis.cross_method_overlap.top_k == 5
        assert 0.0 <= synthesis.cross_method_overlap.shap_lime_jaccard <= 1.0
