import io
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.domain import Dataset, JobStatus, PreprocessingArtifact
from app.services.preprocessing_service import PreprocessingService


def create_sample_student_csv() -> bytes:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Age": np.random.randint(18, 30, size=n),
        "GPA": np.round(np.random.uniform(1.5, 4.0, size=n), 2),
        "Attendance": np.random.uniform(50.0, 100.0, size=n),
        "Tuition_Paid": np.random.choice(["Yes", "No"], size=n),
        "Target": np.random.choice(["Dropout", "Graduate"], size=n, p=[0.3, 0.7])
    })
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def test_dataset_upload_and_detail(client: TestClient):
    csv_bytes = create_sample_student_csv()

    response = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Synthetic Student Dropout Dataset", "target_column": "Target"},
        files={"file": ("student_data.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Synthetic Student Dropout Dataset"
    assert data["row_count"] == 100
    assert data["feature_count"] == 4
    dataset_id = data["id"]

    # Test GET dataset detail
    detail_res = client.get(f"/api/v1/datasets/{dataset_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert len(detail_data["columns"]) == 5
    assert len(detail_data["sample_rows"]) == 5
    assert "GPA" in detail_data["summary_stats"]


def test_preprocessing_execution_and_leakage_prevention(client: TestClient, db: Session):
    csv_bytes = create_sample_student_csv()

    upload_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Leakage Test Dataset", "target_column": "Target"},
        files={"file": ("student_data.csv", csv_bytes, "text/csv")}
    )
    dataset_id = upload_res.json()["id"]

    # Trigger preprocessing
    preprocess_res = client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        json={
            "target_column": "Target",
            "positive_class_val": "Dropout",
            "scale_features": True,
            "val_size": 0.20,
            "test_size": 0.20,
            "random_state": 42
        }
    )
    assert preprocess_res.status_code == 202
    job_data = preprocess_res.json()
    assert job_data["type"] == "PREPROCESSING"
    assert job_data["status"] in [JobStatus.SUCCESS, JobStatus.PENDING, JobStatus.RUNNING]

    # Fetch preprocessing artifact from DB
    artifact = db.query(PreprocessingArtifact).filter(PreprocessingArtifact.dataset_id == dataset_id).first()
    assert artifact is not None
    assert artifact.split_summary["train_samples"] == 60
    assert artifact.split_summary["val_samples"] == 20
    assert artifact.split_summary["test_samples"] == 20

    # DATA LEAKAGE PREVENTION VERIFICATION
    # Load the preprocessor fitted by PreprocessingService
    preprocessor, splits = PreprocessingService.load_preprocessor_and_splits(artifact)
    num_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # Manually compute mean of training set raw values
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    from sklearn.model_selection import train_test_split
    X_raw = df_raw.drop(columns=["Target"])
    target_map = {"Graduate": 0, "Dropout": 1}
    y_mapped = df_raw["Target"].map(target_map).values
    num_cols = ["Age", "GPA", "Attendance"]

    X_train_raw, X_temp_raw, _, _ = train_test_split(
        X_raw, y_mapped, test_size=0.40, random_state=42, stratify=y_mapped
    )
    train_num_means = X_train_raw[num_cols].mean().values
    whole_num_means = X_raw[num_cols].mean().values

    # Scaler mean must match training set mean, NOT whole set mean!
    np.testing.assert_allclose(num_scaler.mean_, train_num_means, rtol=1e-5)
    assert not np.allclose(num_scaler.mean_, whole_num_means)

    # Test artifacts endpoints
    list_artifacts_res = client.get(f"/api/v1/datasets/{dataset_id}/artifacts")
    assert list_artifacts_res.status_code == 200
    assert len(list_artifacts_res.json()) == 1

    schema_res = client.get(f"/api/v1/datasets/artifacts/{artifact.id}/schema")
    assert schema_res.status_code == 200
    assert schema_res.json()["version"] == artifact.version
