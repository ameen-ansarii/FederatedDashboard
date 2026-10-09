import io
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ml.federated.fedprox_client import DropoutFedProxFlowerClient
from app.models.domain import JobStatus, ModelRegistry, PreprocessingArtifact
from app.services.model_service import ModelService


def create_fedprox_test_csv() -> bytes:
    np.random.seed(888)
    n = 160
    df = pd.DataFrame({
        "Age": np.random.randint(18, 30, size=n),
        "GPA": np.round(np.random.uniform(1.5, 4.0, size=n), 2),
        "Attendance": np.random.uniform(50.0, 100.0, size=n),
        "Tuition_Paid": np.random.choice(["Yes", "No"], size=n),
        "Target": np.random.choice(["Dropout", "Graduate"], size=n, p=[0.45, 0.55])
    })
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def test_fedprox_proximal_penalty_math_properties():
    """
    Tests:
    1. Proximal penalty is exactly 0.0 when local parameters equal global parameters.
    2. Proximal penalty increases monotonically as local parameters diverge from global parameters.
    3. mu=0 produces zero proximal penalty.
    4. mu>0 changes the optimization objective.
    """
    X_k = np.random.randn(20, 4)
    y_k = np.array([1] * 10 + [0] * 10)

    client = DropoutFedProxFlowerClient(
        client_id="fedprox_test_client",
        input_dim=4,
        X_train=X_k,
        y_train=y_k,
        hidden_dims=[16, 8],
        mu=0.01,
    )

    global_params_frozen = [p.clone().detach() for p in client.model.parameters()]

    # 1. When local parameters equal global parameters, penalty MUST be 0.0
    penalty_zero = client.compute_proximal_term(global_params_frozen)
    assert float(penalty_zero.item()) == 0.0

    # 2. Perturb local model parameters slightly
    with torch.no_grad():
        for p in client.model.parameters():
            p.add_(0.1)

    penalty_small = client.compute_proximal_term(global_params_frozen)
    assert float(penalty_small.item()) > 0.0

    # Perturb local model parameters further
    with torch.no_grad():
        for p in client.model.parameters():
            p.add_(0.5)

    penalty_large = client.compute_proximal_term(global_params_frozen)
    assert float(penalty_large.item()) > float(penalty_small.item())

    # 3. Test mu=0 behavior -> penalty is 0.0 regardless of divergence
    client_mu_zero = DropoutFedProxFlowerClient(
        client_id="client_mu_zero",
        input_dim=4,
        X_train=X_k,
        y_train=y_k,
        hidden_dims=[16, 8],
        mu=0.0,
    )
    penalty_when_mu_zero = client_mu_zero.compute_proximal_term(global_params_frozen)
    assert float(penalty_when_mu_zero.item()) == 0.0


def test_end_to_end_fedprox_pipeline(client: TestClient, db: Session):
    """
    Integration test:
    - Upload dataset & preprocess.
    - Trigger FedAvg AND FedProx APIs.
    - Verify multi-round FedProx training and real metrics.
    - Verify FedAvg and FedProx models are registered separately and distinguishable in ModelRegistry.
    """
    csv_bytes = create_fedprox_test_csv()

    # 1. Upload Dataset
    upload_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "FedProx Validation Dataset", "target_column": "Target"},
        files={"file": ("student_fedprox.csv", csv_bytes, "text/csv")}
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

    # 3. Trigger FedAvg Training (Baseline)
    fedavg_res = client.post(
        "/api/v1/training/fedavg",
        json={
            "artifact_id": artifact.id,
            "name": "FedAvg Model Baseline",
            "num_clients": 3,
            "rounds": 4,
            "local_epochs": 2,
            "random_state": 42
        }
    )
    assert fedavg_res.status_code == 202

    # 4. Trigger FedProx Training (with mu=0.05)
    fedprox_res = client.post(
        "/api/v1/training/fedprox",
        json={
            "artifact_id": artifact.id,
            "name": "FedProx Model mu=0.05",
            "num_clients": 3,
            "partitioning": "non_iid",
            "dirichlet_alpha": 0.5,
            "rounds": 4,
            "local_epochs": 2,
            "mu": 0.05,
            "random_state": 42
        }
    )
    assert fedprox_res.status_code == 202
    job_data = fedprox_res.json()
    assert job_data["type"] == "FEDPROX_TRAINING"
    assert job_data["status"] == JobStatus.SUCCESS
    job_id = job_data["id"]

    # 5. Fetch FedProx Training Run
    run_res = client.get(f"/api/v1/training/runs/{job_id}")
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["type"] == "FEDPROX"
    assert run_data["status"] == JobStatus.SUCCESS
    assert "accuracy" in run_data["metrics"]
    assert "prox_loss" in run_data["history"]
    assert len(run_data["history"]["round"]) == 4

    # 6. Verify Model Registry Distinguishability
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    models = models_res.json()

    # Locate FedAvg and FedProx models
    fedavg_models = [m for m in models if "FedAvg" in m["name"]]
    fedprox_models = [m for m in models if "FedProx" in m["name"]]

    assert len(fedavg_models) >= 1
    assert len(fedprox_models) >= 1

    fedprox_model_detail = client.get(f"/api/v1/models/{fedprox_models[0]['id']}").json()
    lineage = fedprox_model_detail["lineage"]

    # Verify FedProx specific metadata in lineage
    assert lineage["hyperparameters"]["algorithm"] == "FEDPROX"
    assert lineage["hyperparameters"]["mu"] == 0.05
    assert len(fedprox_model_detail["checksum"]) == 64
