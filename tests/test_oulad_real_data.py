import io
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.domain import JobStatus, ModelRegistry, PreprocessingArtifact
from app.services.model_service import ModelService


def create_oulad_derived_student_csv() -> bytes:
    """
    Generates a realistic student-level feature dataset modeled on OULAD
    (Open University Learning Analytics Dataset).
    """
    np.random.seed(123)
    n = 200

    code_modules = np.random.choice(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"], size=n)
    code_presentations = np.random.choice(["2013J", "2013B", "2014J", "2014B"], size=n)
    genders = np.random.choice(["M", "F"], size=n)
    highest_education = np.random.choice(
        ["HE Qualification", "A Level or Equivalent", "Lower Than A Level", "Post Graduate Qualification"],
        size=n,
    )
    imd_bands = np.random.choice(["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "90-100%"], size=n)
    age_bands = np.random.choice(["0-35", "35-55", "55<="], size=n)
    num_prev_attempts = np.random.choice([0, 1, 2, 3], size=n, p=[0.75, 0.15, 0.07, 0.03])
    studied_credits = np.random.choice([30, 60, 90, 120, 150], size=n)
    disabilities = np.random.choice(["N", "Y"], size=n, p=[0.9, 0.1])

    # Engagement & performance
    total_vle_clicks = np.random.exponential(scale=500, size=n).astype(int)
    avg_assessment_score = np.round(np.random.uniform(20.0, 100.0, size=n), 1)

    # OULAD Target: Withdrawn (Dropout), Pass, Distinction, Fail
    # Students with low clicks & low assessment score have high chance of Withdrawn
    outcomes = []
    for i in range(n):
        if total_vle_clicks[i] < 150 or avg_assessment_score[i] < 40:
            outcomes.append(np.random.choice(["Withdrawn", "Fail"], p=[0.7, 0.3]))
        else:
            outcomes.append(np.random.choice(["Pass", "Distinction", "Withdrawn"], p=[0.6, 0.3, 0.1]))

    df = pd.DataFrame({
        "code_module": code_modules,
        "code_presentation": code_presentations,
        "gender": genders,
        "highest_education": highest_education,
        "imd_band": imd_bands,
        "age_band": age_bands,
        "num_of_prev_attempts": num_prev_attempts,
        "studied_credits": studied_credits,
        "disability": disabilities,
        "total_vle_clicks": total_vle_clicks,
        "avg_assessment_score": avg_assessment_score,
        "final_result": outcomes,
    })

    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def test_oulad_real_dataset_preprocessing_and_training(client: TestClient, db: Session):
    csv_bytes = create_oulad_derived_student_csv()

    # 1. Upload OULAD-derived Dataset
    upload_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "OULAD Student Dropout Benchmark", "target_column": "final_result"},
        files={"file": ("oulad_students.csv", csv_bytes, "text/csv")}
    )
    assert upload_res.status_code == 201
    dataset_data = upload_res.json()
    assert dataset_data["name"] == "OULAD Student Dropout Benchmark"
    dataset_id = dataset_data["id"]

    # 2. Preprocess with OULAD Target "Withdrawn" (mapped to 1=Dropout, others=0=Not Dropout)
    preprocess_res = client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        json={
            "target_column": "final_result",
            "positive_class_val": "Withdrawn",
            "scale_features": True,
            "handle_duplicates": True,
            "val_size": 0.20,
            "test_size": 0.20,
            "random_state": 42
        }
    )
    assert preprocess_res.status_code == 202

    artifact = db.query(PreprocessingArtifact).filter(PreprocessingArtifact.dataset_id == dataset_id).first()
    assert artifact is not None
    assert "code_module" in artifact.feature_schema["categorical_features"]
    assert "total_vle_clicks" in artifact.feature_schema["numerical_features"]

    # Target mapping verification: Withdrawn -> 1, Pass/Distinction/Fail -> 0
    assert artifact.target_mapping["Withdrawn"] == 1
    assert artifact.target_mapping["Pass"] == 0

    # 3. Train Centralized Model on OULAD preprocessed data
    train_res = client.post(
        "/api/v1/training/centralized",
        json={
            "artifact_id": artifact.id,
            "name": "OULAD Centralized PyTorch Classifier",
            "hidden_dims": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 15,
            "batch_size": 32,
            "random_state": 42
        }
    )
    assert train_res.status_code == 202
    job_data = train_res.json()
    assert job_data["status"] == JobStatus.SUCCESS
    job_id = job_data["id"]

    # 4. Verify Training Run & Real Scikit-Learn Metrics
    run_res = client.get(f"/api/v1/training/runs/{job_id}")
    assert run_res.status_code == 200
    metrics = run_res.json()["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2

    # 5. Verify Model Registry Details & Lineage
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    model_id = models_res.json()[0]["id"]

    detail_res = client.get(f"/api/v1/models/{model_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["lineage"]["dataset_id"] == dataset_id
    assert detail["lineage"]["preprocessing_artifact_id"] == artifact.id
    assert len(detail["checksum"]) == 64
