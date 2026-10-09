import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.models.domain import RiskLevel


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


def setup_trained_model_and_prediction(client: TestClient):
    """Helper to upload data, preprocess, train model, and create a single prediction."""
    csv_bytes = create_sample_student_csv()

    # 1. Upload dataset
    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "XAI Test Dataset", "target_column": "Target"},
        files={"file": ("xai_student_data.csv", csv_bytes, "text/csv")},
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

    artifacts_res = client.get(f"/api/v1/datasets/{dataset_id}/artifacts")
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

    models_res = client.get("/api/v1/models")
    model_id = models_res.json()[0]["id"]

    # 4. Create a Prediction record
    pred_res = client.post(
        "/api/v1/predictions/single",
        json={
            "model_id": model_id,
            "features": {"Age": 22, "GPA": 3.2, "Attendance": 85.0, "Tuition_Paid": "Yes"},
            "save_to_history": True,
        },
    )
    assert pred_res.status_code == 200
    prediction_id = pred_res.json()["id"]

    return model_id, artifact_id, prediction_id


def test_shap_single_prediction(client: TestClient):
    """
    Verify SHAP generates local feature contributions, meaningful feature names,
    valid direction and numeric attribution scores.
    """
    model_id, artifact_id, pred_id = setup_trained_model_and_prediction(client)

    # 1. Explain via prediction_id
    shap_res = client.post("/api/v1/xai/shap", json={"prediction_id": pred_id})
    assert shap_res.status_code == 200
    data = shap_res.json()

    assert data["method"] == "SHAP"
    assert data["prediction_id"] == pred_id
    assert data["model_id"] == model_id
    assert 0.0 <= data["probability"] <= 1.0
    assert len(data["features"]) >= 4

    for f in data["features"]:
        assert isinstance(f["feature_name"], str)
        assert not f["feature_name"].startswith("feature_")  # Meaningful names preserved
        assert isinstance(f["attribution"], float)
        assert not np.isnan(f["attribution"])
        assert f["direction"] in ["positive", "negative"]
        assert f["rank"] >= 1

    # Ranks must be sorted 1, 2, 3...
    ranks = [f["rank"] for f in data["features"]]
    assert ranks == list(range(1, len(ranks) + 1))

    # 2. Explain via direct features
    shap_direct_res = client.post(
        "/api/v1/xai/shap",
        json={
            "model_id": model_id,
            "features": {"Age": 26, "GPA": 2.1, "Attendance": 55.0, "Tuition_Paid": "No"},
        },
    )
    assert shap_direct_res.status_code == 200
    assert len(shap_direct_res.json()["features"]) >= 4

    # No filesystem paths in response
    assert "artifacts/" not in str(data)
    assert "storage_data" not in str(data)


def test_lime_single_prediction(client: TestClient):
    """
    Verify LIME generates local feature weights and directions.
    """
    model_id, _, pred_id = setup_trained_model_and_prediction(client)

    lime_res = client.post("/api/v1/xai/lime", json={"prediction_id": pred_id})
    assert lime_res.status_code == 200
    data = lime_res.json()

    assert data["method"] == "LIME"
    assert data["prediction_id"] == pred_id
    assert len(data["features"]) >= 4

    for f in data["features"]:
        assert isinstance(f["attribution"], float)
        assert not np.isnan(f["attribution"])
        assert f["direction"] in ["positive", "negative"]

    assert "artifacts/" not in str(data)


def test_integrated_gradients_prediction(client: TestClient):
    """
    Verify Captum Integrated Gradients generates gradient attributions against PyTorch model.
    """
    model_id, _, pred_id = setup_trained_model_and_prediction(client)

    ig_res = client.post("/api/v1/xai/integrated-gradients", json={"prediction_id": pred_id})
    assert ig_res.status_code == 200
    data = ig_res.json()

    assert data["method"] == "INTEGRATED_GRADIENTS"
    assert data["prediction_id"] == pred_id
    assert len(data["features"]) >= 4

    for f in data["features"]:
        assert isinstance(f["attribution"], float)
        assert not np.isnan(f["attribution"])
        assert f["direction"] in ["positive", "negative"]

    assert "artifacts/" not in str(data)


def test_global_shap_importance(client: TestClient):
    """
    Verify global SHAP computes mean absolute importance across training/validation samples.
    """
    model_id, _, _ = setup_trained_model_and_prediction(client)

    global_res = client.post(
        "/api/v1/xai/shap/global",
        json={"model_id": model_id, "max_samples": 20},
    )
    assert global_res.status_code == 200
    data = global_res.json()

    assert data["model_id"] == model_id
    assert data["samples_evaluated"] <= 20
    assert len(data["feature_importance"]) >= 4

    # Verify descending ordering of mean_abs_shap
    scores = [item["mean_abs_shap"] for item in data["feature_importance"]]
    assert scores == sorted(scores, reverse=True)


def test_unified_xai_prediction_endpoint(client: TestClient):
    """
    Verify GET /api/v1/xai/predictions/{prediction_id} produces unified SHAP, LIME, and IG outputs.
    """
    _, _, pred_id = setup_trained_model_and_prediction(client)

    res = client.get(f"/api/v1/xai/predictions/{pred_id}")
    assert res.status_code == 200
    data = res.json()

    assert data["prediction_id"] == pred_id
    assert "prediction" in data
    assert "probability" in data["prediction"]
    assert "risk_level" in data["prediction"]

    explanations = data["explanations"]
    assert "shap" in explanations
    assert "lime" in explanations
    assert "integrated_gradients" in explanations

    assert len(explanations["shap"]["features"]) >= 4
    assert len(explanations["lime"]["features"]) >= 4
    assert len(explanations["integrated_gradients"]["features"]) >= 4


def test_cross_model_xai_centralized_fedavg_fedprox(client: TestClient):
    """
    Verify all 3 XAI methods (SHAP, LIME, Integrated Gradients) succeed on
    Centralized, FedAvg, and FedProx models.
    """
    csv_bytes = create_sample_student_csv()
    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Cross Model XAI Dataset", "target_column": "Target"},
        files={"file": ("cross_student_data.csv", csv_bytes, "text/csv")},
    )
    dataset_id = up_res.json()["id"]

    prep_res = client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        json={"target_column": "Target", "val_size": 0.20, "test_size": 0.20, "random_state": 42},
    )
    artifact_id = client.get(f"/api/v1/datasets/{dataset_id}/artifacts").json()[0]["id"]

    # 1. FedAvg Model
    client.post(
        "/api/v1/training/fedavg",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 1,
            "local_epochs": 1,
            "batch_size": 16,
            "hidden_dims": [16, 8],
            "partitioning": "IID",
            "random_state": 42,
        },
    )
    # 2. FedProx Model
    client.post(
        "/api/v1/training/fedprox",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 1,
            "local_epochs": 1,
            "batch_size": 16,
            "mu": 0.05,
            "hidden_dims": [16, 8],
            "partitioning": "IID",
            "random_state": 42,
        },
    )

    models = client.get("/api/v1/models").json()
    assert len(models) >= 2

    # Test each model with SHAP, LIME, and IG
    for m in models[:2]:
        m_id = m["id"]
        features = {"Age": 23, "GPA": 3.1, "Attendance": 80.0, "Tuition_Paid": "Yes"}

        # SHAP
        shap_res = client.post("/api/v1/xai/shap", json={"model_id": m_id, "features": features})
        assert shap_res.status_code == 200
        assert len(shap_res.json()["features"]) >= 4

        # LIME
        lime_res = client.post("/api/v1/xai/lime", json={"model_id": m_id, "features": features})
        assert lime_res.status_code == 200
        assert len(lime_res.json()["features"]) >= 4

        # Integrated Gradients
        ig_res = client.post(
            "/api/v1/xai/integrated-gradients", json={"model_id": m_id, "features": features}
        )
        assert ig_res.status_code == 200
        assert len(ig_res.json()["features"]) >= 4


def test_xai_error_handling_and_validation(client: TestClient):
    """
    Verify error responses for missing predictions, invalid models, and malformed inputs.
    """
    # 1. Non-existent prediction
    res_pred = client.get("/api/v1/xai/predictions/999999")
    assert res_pred.status_code == 404

    # 2. Missing both prediction_id and model_id+features
    res_empty = client.post("/api/v1/xai/shap", json={})
    assert res_empty.status_code == 400

    # 3. Non-existent model ID
    res_model = client.post(
        "/api/v1/xai/shap",
        json={
            "model_id": 999999,
            "features": {"Age": 20, "GPA": 3.0, "Attendance": 80.0, "Tuition_Paid": "Yes"},
        },
    )
    assert res_model.status_code == 404
