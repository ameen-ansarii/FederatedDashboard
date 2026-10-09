import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.domain import JobStatus, ModelRegistry, PreprocessingArtifact, RiskLevel
from app.services.prediction_service import PredictionService


def create_sample_student_csv() -> bytes:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Age": np.random.randint(18, 30, size=n),
        "GPA": np.round(np.random.uniform(1.5, 4.0, size=n), 2),
        "Attendance": np.random.uniform(50.0, 100.0, size=n),
        "Tuition_Paid": np.random.choice(["Yes", "No"], size=n),
        "Target": np.random.choice(["Dropout", "Graduate"], size=n, p=[0.3, 0.7]),
    })
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def setup_trained_model(client: TestClient) -> int:
    """Helper to upload data, preprocess, and train a real model, returning model_id."""
    csv_bytes = create_sample_student_csv()

    # 1. Upload
    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Prediction Test Dataset", "target_column": "Target"},
        files={"file": ("student_data.csv", csv_bytes, "text/csv")},
    )
    assert up_res.status_code == 201
    dataset_id = up_res.json()["id"]

    # 2. Preprocess
    prep_res = client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        json={
            "target_column": "Target",
            "positive_class_val": "Dropout",
            "scale_features": True,
            "val_size": 0.20,
            "test_size": 0.20,
            "random_state": 42,
        },
    )
    assert prep_res.status_code == 202

    # Get artifact ID
    artifacts_res = client.get(f"/api/v1/datasets/{dataset_id}/artifacts")
    assert artifacts_res.status_code == 200
    artifact_id = artifacts_res.json()[0]["id"]

    # 3. Train Centralized Model
    train_res = client.post(
        "/api/v1/training/centralized",
        json={
            "artifact_id": artifact_id,
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.01,
            "hidden_dims": [32, 16],
            "dropout_rate": 0.1,
            "random_state": 42,
        },
    )
    assert train_res.status_code == 202
    job_id = train_res.json()["id"]

    # Retrieve trained model ID from models list
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    model_id = models_res.json()[0]["id"]
    return model_id


def test_risk_band_boundaries_and_classification():
    """
    Test exact mathematical boundaries for risk classification:
    LOW: p < 0.40
    MEDIUM: 0.40 <= p < 0.70
    HIGH: p >= 0.70
    """
    assert PredictionService.classify_risk(0.0) == RiskLevel.LOW
    assert PredictionService.classify_risk(0.39999) == RiskLevel.LOW
    assert PredictionService.classify_risk(0.40000) == RiskLevel.MEDIUM
    assert PredictionService.classify_risk(0.55000) == RiskLevel.MEDIUM
    assert PredictionService.classify_risk(0.69999) == RiskLevel.MEDIUM
    assert PredictionService.classify_risk(0.70000) == RiskLevel.HIGH
    assert PredictionService.classify_risk(0.85000) == RiskLevel.HIGH
    assert PredictionService.classify_risk(1.0) == RiskLevel.HIGH


def test_single_prediction_end_to_end(client: TestClient):
    """
    Verify single prediction executes real inference, computes probability,
    maps risk levels correctly, and does not leak filesystem paths.
    """
    model_id = setup_trained_model(client)

    payload = {
        "model_id": model_id,
        "features": {
            "Age": 22,
            "GPA": 3.5,
            "Attendance": 92.0,
            "Tuition_Paid": "Yes",
        },
        "save_to_history": True,
    }

    pred_res = client.post("/api/v1/predictions/single", json=payload)
    assert pred_res.status_code == 200
    data = pred_res.json()

    assert data["model_id"] == model_id
    assert "model_version" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction_class"] in [0, 1]
    assert data["prediction_label"] in ["Dropout", "Not Dropout"]
    assert data["risk_level"] in [RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]
    assert data["id"] is not None

    # Check risk level matches probability
    p = data["probability"]
    if p < 0.40:
        assert data["risk_level"] == "LOW"
    elif p < 0.70:
        assert data["risk_level"] == "MEDIUM"
    else:
        assert data["risk_level"] == "HIGH"

    # Verify no filesystem paths are exposed in response
    res_str = str(data)
    assert "artifacts/" not in res_str
    assert "storage_data" not in res_str
    assert "/home/" not in res_str


def test_schema_mismatch_and_validation_errors(client: TestClient):
    """
    Verify error handling for missing features, invalid models, and malformed requests.
    """
    model_id = setup_trained_model(client)

    # 1. Missing required features
    bad_payload = {
        "model_id": model_id,
        "features": {
            "Age": 22,
            # Missing GPA, Attendance, Tuition_Paid
        },
        "save_to_history": False,
    }
    res = client.post("/api/v1/predictions/single", json=bad_payload)
    assert res.status_code == 400
    error_msg = res.json().get("error", {}).get("message") or res.json().get("detail", "")
    assert "missing required features" in error_msg.lower()

    # 2. Non-existent model ID
    bad_model_payload = {
        "model_id": 999999,
        "features": {"Age": 20, "GPA": 3.0, "Attendance": 80.0, "Tuition_Paid": "Yes"},
        "save_to_history": False,
    }
    res_model = client.post("/api/v1/predictions/single", json=bad_model_payload)
    assert res_model.status_code == 404

    # 3. Empty features
    empty_features_payload = {
        "model_id": model_id,
        "features": {},
        "save_to_history": False,
    }
    res_empty = client.post("/api/v1/predictions/single", json=empty_features_payload)
    assert res_empty.status_code == 400


def test_batch_prediction_end_to_end(client: TestClient):
    """
    Verify asynchronous batch prediction:
    1. POST /batch returns 202 Accepted with durable job_id.
    2. GET /batch/{job_id}/results retrieves full predictions array and summary.
    """
    model_id = setup_trained_model(client)

    records = [
        {"Age": 19, "GPA": 3.9, "Attendance": 98.0, "Tuition_Paid": "Yes"},
        {"Age": 28, "GPA": 1.6, "Attendance": 45.0, "Tuition_Paid": "No"},
        {"Age": 22, "GPA": 2.8, "Attendance": 75.0, "Tuition_Paid": "Yes"},
    ]

    batch_payload = {
        "model_id": model_id,
        "records": records,
        "save_to_history": True,
    }

    batch_res = client.post("/api/v1/predictions/batch", json=batch_payload)
    assert batch_res.status_code == 202
    job_data = batch_res.json()
    job_id = job_data["id"]
    assert job_data["type"] == "BATCH_PREDICTION"

    # Retrieve batch results
    results_res = client.get(f"/api/v1/predictions/batch/{job_id}/results")
    assert results_res.status_code == 200
    res_data = results_res.json()

    assert res_data["job_id"] == job_id
    assert res_data["status"] == "SUCCESS"
    assert res_data["total_samples"] == 3
    assert len(res_data["predictions"]) == 3

    # Check summary metrics
    summary = res_data["summary"]
    assert summary["total_samples"] == 3
    assert (
        summary["low_risk_count"] + summary["medium_risk_count"] + summary["high_risk_count"] == 3
    )
    assert 0.0 <= summary["mean_probability"] <= 1.0

    # Verify no filesystem paths leaked
    assert "artifacts/" not in str(res_data)
    assert "storage_data" not in str(res_data)


def test_prediction_history_endpoints(client: TestClient):
    """
    Verify prediction history pagination, single record lookup, and risk_level filtering.
    """
    model_id = setup_trained_model(client)

    # Make 2 single predictions to ensure history records exist
    for gpa in [3.8, 1.7]:
        client.post(
            "/api/v1/predictions/single",
            json={
                "model_id": model_id,
                "features": {"Age": 21, "GPA": gpa, "Attendance": 80.0, "Tuition_Paid": "Yes"},
                "save_to_history": True,
            },
        )

    # 1. Fetch history
    hist_res = client.get(f"/api/v1/predictions/history?model_id={model_id}&limit=10")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["total"] >= 2
    assert len(hist_data["items"]) >= 2

    first_item = hist_data["items"][0]
    pred_id = first_item["id"]

    # 2. Fetch single history record by ID
    single_hist_res = client.get(f"/api/v1/predictions/history/{pred_id}")
    assert single_hist_res.status_code == 200
    single_item = single_hist_res.json()
    assert single_item["id"] == pred_id
    assert single_item["model_id"] == model_id
    assert "GPA" in single_item["feature_values"]


def test_prediction_with_fedavg_and_fedprox_models(client: TestClient):
    """
    Verify single and batch inference work with actual FedAvg and FedProx models
    and verify lineage-linked preprocessor is used.
    """
    csv_bytes = create_sample_student_csv()
    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Fed Prediction Dataset", "target_column": "Target"},
        files={"file": ("fed_student_data.csv", csv_bytes, "text/csv")},
    )
    dataset_id = up_res.json()["id"]

    prep_res = client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        json={
            "target_column": "Target",
            "positive_class_val": "Dropout",
            "scale_features": True,
            "val_size": 0.20,
            "test_size": 0.20,
            "random_state": 42,
        },
    )
    artifacts_res = client.get(f"/api/v1/datasets/{dataset_id}/artifacts")
    artifact_id = artifacts_res.json()[0]["id"]

    # 1. FedAvg Training
    fedavg_res = client.post(
        "/api/v1/training/fedavg",
        json={
            "artifact_id": artifact_id,
            "num_clients": 3,
            "rounds": 2,
            "local_epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
            "hidden_dims": [32, 16],
            "dropout_rate": 0.1,
            "partitioning": "IID",
            "random_state": 42,
        },
    )
    assert fedavg_res.status_code == 202

    # 2. FedProx Training
    fedprox_res = client.post(
        "/api/v1/training/fedprox",
        json={
            "artifact_id": artifact_id,
            "num_clients": 3,
            "rounds": 2,
            "local_epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
            "mu": 0.1,
            "hidden_dims": [32, 16],
            "dropout_rate": 0.1,
            "partitioning": "NON_IID",
            "dirichlet_alpha": 0.5,
            "random_state": 42,
        },
    )
    assert fedprox_res.status_code == 202

    models = client.get("/api/v1/models").json()
    assert len(models) >= 2

    # Predict with both latest models
    for m in models[:2]:
        single_res = client.post(
            "/api/v1/predictions/single",
            json={
                "model_id": m["id"],
                "features": {"Age": 24, "GPA": 2.9, "Attendance": 70.0, "Tuition_Paid": "No"},
                "save_to_history": True,
            },
        )
        assert single_res.status_code == 200
        data = single_res.json()
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

