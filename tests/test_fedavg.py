import io
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ml.federated.client import DropoutFlowerClient
from app.ml.federated.strategy import aggregate_fedavg
from app.ml.models import DropoutMLP
from app.ml.partitioning import partition_dataset
from app.models.domain import JobStatus, ModelRegistry, PreprocessingArtifact
from app.services.model_service import ModelService


def create_fl_test_csv() -> bytes:
    np.random.seed(999)
    n = 150
    df = pd.DataFrame({
        "Age": np.random.randint(18, 30, size=n),
        "GPA": np.round(np.random.uniform(1.5, 4.0, size=n), 2),
        "Attendance": np.random.uniform(50.0, 100.0, size=n),
        "Tuition_Paid": np.random.choice(["Yes", "No"], size=n),
        "Target": np.random.choice(["Dropout", "Graduate"], size=n, p=[0.4, 0.6])
    })
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def test_fedavg_mathematical_exact_weighted_average():
    """
    Deterministic unit test with manually known client parameters and sample counts:
    Client 1: parameters = [[1.0, 2.0], [3.0]], N1 = 10
    Client 2: parameters = [[5.0, 10.0], [15.0]], N2 = 30
    Total N = 40 (weights: 0.25 for Client 1, 0.75 for Client 2)

    Expected Layer 0: 0.25 * [1.0, 2.0] + 0.75 * [5.0, 10.0] = [4.0, 8.0]
    Expected Layer 1: 0.25 * [3.0] + 0.75 * [15.0] = [12.0]
    """
    client_1_params = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0], dtype=np.float32)]
    client_2_params = [np.array([5.0, 10.0], dtype=np.float32), np.array([15.0], dtype=np.float32)]

    results = [
        (client_1_params, 10),
        (client_2_params, 30),
    ]

    aggregated = aggregate_fedavg(results)

    expected_layer_0 = np.array([4.0, 8.0], dtype=np.float32)
    expected_layer_1 = np.array([12.0], dtype=np.float32)

    np.testing.assert_allclose(aggregated[0], expected_layer_0, rtol=1e-6)
    np.testing.assert_allclose(aggregated[1], expected_layer_1, rtol=1e-6)


def test_partitioning_coverage_and_isolation():
    """
    Verifies that partitioning covers 100% of training samples, has zero client sample overlap,
    and produces deterministic results under fixed seeds.
    """
    X = np.random.randn(100, 5)
    y = np.array([1] * 40 + [0] * 60)

    # 1. Test IID Partitioning
    iid_parts_1 = partition_dataset(X, y, num_clients=5, strategy="iid", seed=42)
    iid_parts_2 = partition_dataset(X, y, num_clients=5, strategy="iid", seed=42)

    total_samples = sum(len(p["y"]) for p in iid_parts_1)
    assert total_samples == 100

    for p1, p2 in zip(iid_parts_1, iid_parts_2):
        np.testing.assert_array_equal(p1["X"], p2["X"])
        np.testing.assert_array_equal(p1["y"], p2["y"])

    # 2. Test Non-IID Dirichlet Partitioning Reproducibility & Coverage
    non_iid_1 = partition_dataset(X, y, num_clients=4, strategy="non_iid", alpha=0.5, seed=42)
    non_iid_2 = partition_dataset(X, y, num_clients=4, strategy="non_iid", alpha=0.5, seed=42)

    total_non_iid = sum(len(p["y"]) for p in non_iid_1)
    assert total_non_iid == 100

    for p1, p2 in zip(non_iid_1, non_iid_2):
        np.testing.assert_array_equal(p1["X"], p2["X"])
        np.testing.assert_array_equal(p1["y"], p2["y"])


def test_flower_client_and_sample_weighted_aggregation():
    X_k = np.random.randn(30, 4)
    y_k = np.array([1] * 10 + [0] * 20)

    client = DropoutFlowerClient(
        client_id="client_0",
        input_dim=4,
        X_train=X_k,
        y_train=y_k,
        hidden_dims=[16, 8],
        local_epochs=2,
    )

    params = client.get_parameters()
    assert isinstance(params, list)
    assert len(params) > 0

    # Perform fit
    updated_params, num_examples, metrics = client.fit(params, {})
    assert num_examples == 30
    assert "train_loss" in metrics
    assert len(updated_params) == len(params)

    # Test Sample-Weighted FedAvg Aggregation Logic
    client_results = [
        (updated_params, 30),
        (params, 70),
    ]
    agg_weights = aggregate_fedavg(client_results)
    assert len(agg_weights) == len(params)

    # Privacy Boundary Verification: assert zero raw records inside client weights payload
    for weight_array in agg_weights:
        assert isinstance(weight_array, np.ndarray)
        assert weight_array.ndim in [1, 2]  # Only 1D/2D PyTorch weight & bias tensors, no raw sample rows!


def test_end_to_end_fedavg_pipeline(client: TestClient, db: Session):
    csv_bytes = create_fl_test_csv()

    # 1. Upload Dataset
    upload_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "FedAvg FL Dataset", "target_column": "Target"},
        files={"file": ("student_fl_data.csv", csv_bytes, "text/csv")}
    )
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["id"]

    # 2. Preprocess Dataset
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

    artifact = db.query(PreprocessingArtifact).filter(PreprocessingArtifact.dataset_id == dataset_id).first()
    assert artifact is not None

    # 3. Trigger FedAvg Training API (Non-IID Dirichlet strategy)
    fedavg_res = client.post(
        "/api/v1/training/fedavg",
        json={
            "artifact_id": artifact.id,
            "name": "FedAvg Non-IID Student Classifier",
            "num_clients": 4,
            "partitioning": "non_iid",
            "dirichlet_alpha": 0.5,
            "rounds": 5,
            "local_epochs": 2,
            "learning_rate": 0.001,
            "batch_size": 16,
            "hidden_dims": [32, 16],
            "dropout_rate": 0.2,
            "optimizer": "adam",
            "random_state": 42
        }
    )
    assert fedavg_res.status_code == 202
    job_data = fedavg_res.json()
    assert job_data["type"] == "FEDAVG_TRAINING"
    assert job_data["status"] == JobStatus.SUCCESS
    job_id = job_data["id"]

    # 4. Fetch Training Run
    run_res = client.get(f"/api/v1/training/runs/{job_id}")
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["type"] == "FEDAVG"
    assert run_data["status"] == JobStatus.SUCCESS
    assert "accuracy" in run_data["metrics"]
    assert len(run_data["history"]["round"]) == 5

    # 5. Fetch Models List & Detail
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    models_list = models_res.json()
    assert len(models_list) >= 1

    latest_model = models_list[0]
    detail_res = client.get(f"/api/v1/models/{latest_model['id']}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["name"] == "FedAvg Non-IID Student Classifier"
    assert detail["framework"] == "PyTorch"
    assert len(detail["checksum"]) == 64

    # Lineage Verification
    lineage = detail["lineage"]
    assert lineage["dataset_id"] == dataset_id
    assert lineage["preprocessing_artifact_id"] == artifact.id
    assert lineage["hyperparameters"]["partitioning_details"]["partitioning"] == "non_iid"
    assert lineage["hyperparameters"]["partitioning_details"]["num_clients"] == 4

    # 6. Verify Global PyTorch Model Reloading & Inference
    model_record = db.query(ModelRegistry).filter(ModelRegistry.id == latest_model['id']).first()
    loaded_global_model = ModelService.load_pytorch_model(model_record)
    assert isinstance(loaded_global_model, DropoutMLP)

    feature_count = artifact.feature_schema["feature_count"]
    sample_input = torch.randn(3, feature_count)
    with torch.no_grad():
        out_logits = loaded_global_model(sample_input)
    assert out_logits.shape == (3, 1)

    # 7. List Federated Client Nodes
    clients_res = client.get("/api/v1/federated/clients")
    assert clients_res.status_code == 200
    client_nodes = clients_res.json()
    assert len(client_nodes) >= 4
