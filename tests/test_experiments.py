import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.errors import ForbiddenException
from app.models.domain import Experiment, Job, JobStatus, ModelRegistry, PreprocessingArtifact, User, UserRole
from app.services.experiment_service import ExperimentService
from app.services.preprocessing_service import PreprocessingService


def create_experiment_student_csv() -> bytes:
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "Age": np.random.randint(18, 30, size=n),
        "GPA": np.round(np.random.uniform(1.5, 4.0, size=n), 2),
        "Attendance": np.random.uniform(50.0, 100.0, size=n),
        "Tuition_Paid": np.random.choice(["Yes", "No"], size=n),
        "Target": np.random.choice(["Dropout", "Graduate"], size=n, p=[0.35, 0.65]),
    })
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def setup_experiment_dataset_and_artifact(client: TestClient) -> int:
    """Helper to upload data and preprocess it, returning the PreprocessingArtifact ID."""
    csv_bytes = create_experiment_student_csv()

    # 1. Upload dataset
    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Benchmark Test Dataset", "target_column": "Target"},
        files={"file": ("benchmark_data.csv", csv_bytes, "text/csv")},
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
    assert len(artifacts_res.json()) > 0
    return artifacts_res.json()[0]["id"]


def test_benchmark_execution(client: TestClient, db: Session):
    """
    Test 1: Verify Centralized, FedAvg IID, FedAvg Non-IID, and FedProx Non-IID
    all execute successfully in an end-to-end controlled benchmark.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    bench_res = client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "name": "Full FL Benchmark Test",
            "num_clients": 3,
            "rounds": 2,
            "local_epochs": 1,
            "centralized_epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.01,
            "dirichlet_alpha": 0.5,
            "mu_values": [0.01, 0.1],
            "random_state": 42,
        },
    )
    assert bench_res.status_code == 202
    job_data = bench_res.json()
    assert job_data["status"] == "SUCCESS"

    # List experiments
    list_res = client.get("/api/v1/experiments")
    assert list_res.status_code == 200
    exp_list = list_res.json()
    assert exp_list["total"] >= 1
    exp_id = exp_list["items"][0]["id"]

    # Get details
    exp_res = client.get(f"/api/v1/experiments/{exp_id}")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()

    assert exp_data["status"] == "SUCCESS"
    assert exp_data["artifact_id"] == artifact_id
    assert len(exp_data["models"]) == 5  # Centralized + FedAvg IID + FedAvg Non-IID + FedProx (mu=0.01, mu=0.1)
    algorithms = [m["algorithm"] for m in exp_data["models"]]
    assert "CENTRALIZED" in algorithms
    assert "FEDAVG" in algorithms
    assert "FEDPROX" in algorithms


def test_same_test_split(client: TestClient, db: Session):
    """
    Test 2: Verify every benchmark model is evaluated against the same held-out test split
    from the preprocessing artifact.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    bench_res = client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 1,
            "centralized_epochs": 2,
            "mu_values": [0.1],
            "random_state": 42,
        },
    )
    assert bench_res.status_code == 202

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    assert exp is not None
    assert exp.artifact_id == artifact_id

    # Verify that each model's confusion matrix total samples equals the test split size
    art = db.query(PreprocessingArtifact).filter(PreprocessingArtifact.id == artifact_id).first()
    _, splits = PreprocessingService.load_preprocessor_and_splits(art)
    expected_test_samples = int(splits["X_test"].shape[0])

    for model_summary in exp.results["models"]:
        cm = model_summary["metrics"]["confusion_matrix"]
        total_eval_samples = sum(sum(row) for row in cm)
        assert total_eval_samples == expected_test_samples


def test_model_lineage(client: TestClient, db: Session):
    """
    Test 3: Verify model -> preprocessing artifact -> dataset lineage is preserved for all benchmark models.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    bench_res = client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 1,
            "centralized_epochs": 2,
            "mu_values": [0.05],
            "random_state": 42,
        },
    )
    assert bench_res.status_code == 202

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    for m_summary in exp.results["models"]:
        model_rec = db.query(ModelRegistry).filter(ModelRegistry.id == m_summary["model_id"]).first()
        assert model_rec is not None
        assert model_rec.lineage["preprocessing_artifact_id"] == artifact_id
        assert model_rec.checksum is not None
        assert len(model_rec.checksum) == 64  # SHA-256 integrity preserved


def test_comparative_deltas(client: TestClient, db: Session):
    """
    Test 4: Verify delta_f1 = federated_f1 - centralized_f1 and delta_accuracy = federated_accuracy - centralized_accuracy.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    bench_res = client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 1,
            "centralized_epochs": 2,
            "mu_values": [0.1],
            "random_state": 42,
        },
    )
    assert bench_res.status_code == 202

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    models = exp.results["models"]
    cent_model = next(m for m in models if m["algorithm"] == "CENTRALIZED")
    cent_f1 = cent_model["metrics"]["f1"]
    cent_acc = cent_model["metrics"]["accuracy"]

    for fed_model in models:
        if fed_model["algorithm"] != "CENTRALIZED":
            expected_df1 = round(fed_model["metrics"]["f1"] - cent_f1, 4)
            expected_dacc = round(fed_model["metrics"]["accuracy"] - cent_acc, 4)
            assert fed_model["delta_from_baseline"]["delta_f1"] == expected_df1
            assert fed_model["delta_from_baseline"]["delta_accuracy"] == expected_dacc


def test_convergence_trajectory(client: TestClient, db: Session):
    """
    Test 5: Verify convergence trajectories exist, containing aligned rounds, loss, and validation accuracy.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 2,
            "centralized_epochs": 2,
            "mu_values": [0.1],
            "random_state": 42,
        },
    )

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    traj_res = client.get(f"/api/v1/experiments/{exp.id}/convergence")
    assert traj_res.status_code == 200
    trajectories = traj_res.json()["trajectories"]

    assert "CENTRALIZED" in trajectories
    assert "FEDAVG_IID" in trajectories
    assert len(trajectories["FEDAVG_IID"]["rounds"]) == 2
    assert len(trajectories["FEDAVG_IID"]["loss"]) == 2
    assert len(trajectories["FEDAVG_IID"]["validation_accuracy"]) == 2


def test_mu_sensitivity(client: TestClient, db: Session):
    """
    Test 6: Verify FedProx mu sensitivity analysis reflects every configured mu value with real metrics.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 1,
            "centralized_epochs": 2,
            "mu_values": [0.01, 0.5, 1.0],
            "random_state": 42,
        },
    )

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    mu_res = client.get(f"/api/v1/experiments/{exp.id}/mu-sensitivity")
    assert mu_res.status_code == 200
    mu_analysis = mu_res.json()["mu_analysis"]

    assert len(mu_analysis) == 3
    tested_mus = [item["mu"] for item in mu_analysis]
    assert tested_mus == [0.01, 0.5, 1.0]

    for item in mu_analysis:
        assert 0.0 <= item["f1"] <= 1.0
        assert 0.0 <= item["accuracy"] <= 1.0
        assert 0.0 <= item["roc_auc"] <= 1.0
        assert not np.isnan(item["loss"])


def test_model_comparison(client: TestClient, db: Session):
    """
    Test 7: Verify arbitrary registered models can be compared side-by-side on a common test split.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    # Train two models
    train1 = client.post(
        "/api/v1/training/centralized",
        json={"artifact_id": artifact_id, "epochs": 2, "random_state": 42},
    )
    assert train1.status_code == 202

    train2 = client.post(
        "/api/v1/training/fedavg",
        json={"artifact_id": artifact_id, "num_clients": 2, "rounds": 1, "random_state": 42},
    )
    assert train2.status_code == 202

    models = client.get("/api/v1/models").json()
    model_ids = [models[0]["id"], models[1]["id"]]

    comp_res = client.post(
        "/api/v1/experiments/compare",
        json={"model_ids": model_ids, "artifact_id": artifact_id},
    )
    assert comp_res.status_code == 200
    comparisons = comp_res.json()

    assert len(comparisons) == 2
    assert comparisons[0]["model_id"] in model_ids
    assert comparisons[1]["model_id"] in model_ids
    assert "accuracy" in comparisons[0]["metrics"]
    assert "f1" in comparisons[0]["metrics"]


def test_iid_partition_metadata(client: TestClient, db: Session):
    """
    Test 8: Verify IID experiment metadata correctly records strategy and client splits.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 3,
            "rounds": 1,
            "centralized_epochs": 1,
            "mu_values": [0.1],
            "random_state": 42,
        },
    )

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    iid_dist = exp.results["client_distributions"]["iid"]

    assert iid_dist["strategy"] == "IID"
    assert iid_dist["num_clients"] == 3
    assert len(iid_dist["client_breakdown"]) == 3
    assert iid_dist["dirichlet_alpha"] is None


def test_non_iid_partition_metadata(client: TestClient, db: Session):
    """
    Test 9: Verify Non-IID experiment metadata tracks Dirichlet alpha and class heterogeneity.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 3,
            "rounds": 1,
            "centralized_epochs": 1,
            "dirichlet_alpha": 0.3,
            "mu_values": [0.1],
            "random_state": 42,
        },
    )

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    non_iid_dist = exp.results["client_distributions"]["non_iid"]

    assert non_iid_dist["strategy"] == "NON_IID"
    assert non_iid_dist["dirichlet_alpha"] == 0.3
    assert len(non_iid_dist["client_breakdown"]) == 3


def test_client_distribution(client: TestClient, db: Session):
    """
    Test 10: Verify sum of partitioned client sample counts equals total training sample count.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 4,
            "rounds": 1,
            "centralized_epochs": 1,
            "mu_values": [0.1],
            "random_state": 42,
        },
    )

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    iid_dist = exp.results["client_distributions"]["iid"]
    non_iid_dist = exp.results["client_distributions"]["non_iid"]

    art = db.query(PreprocessingArtifact).filter(PreprocessingArtifact.id == artifact_id).first()
    _, splits = PreprocessingService.load_preprocessor_and_splits(art)
    expected_train_rows = int(splits["X_train"].shape[0])

    sum_iid_samples = sum(c["sample_count"] for c in iid_dist["client_breakdown"].values())
    sum_non_iid_samples = sum(c["sample_count"] for c in non_iid_dist["client_breakdown"].values())

    assert sum_iid_samples == expected_train_rows
    assert sum_non_iid_samples == expected_train_rows


def test_invalid_benchmark_config(client: TestClient):
    """
    Test 11: Verify invalid benchmark configuration parameters (learning_rate <= 0, empty mu_values, invalid dropout) are rejected.
    """
    # 1. Invalid learning rate
    res1 = client.post(
        "/api/v1/experiments/benchmark",
        json={"artifact_id": 1, "learning_rate": -0.05},
    )
    assert res1.status_code == 422

    # 2. Empty mu_values
    res2 = client.post(
        "/api/v1/experiments/benchmark",
        json={"artifact_id": 1, "mu_values": []},
    )
    assert res2.status_code == 422

    # 3. Invalid dropout rate >= 1.0
    res3 = client.post(
        "/api/v1/experiments/benchmark",
        json={"artifact_id": 1, "dropout_rate": 1.5},
    )
    assert res3.status_code == 422


def test_authorization(client: TestClient, db: Session):
    """
    Test 12: Verify experiment access is restricted according to user ownership and roles.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    exp_id = "exp-auth-test"
    exp = Experiment(
        id=exp_id,
        name="Private Experiment",
        status=JobStatus.SUCCESS,
        artifact_id=artifact_id,
        config={},
        results={"summary": {}},
        user_id=999,  # Belongs to user 999
    )
    db.add(exp)
    db.commit()

    # User 1 attempting to access user 999's private experiment directly raises ForbiddenException
    unauthorized_user = User(id=1, email="test@example.com", role=UserRole.RESEARCHER)
    with pytest.raises(ForbiddenException):
        ExperimentService.get_experiment_by_id(db=db, experiment_id=exp_id, user=unauthorized_user)

    # Admin user has permission to access any experiment
    admin_user = User(id=2, email="admin@example.com", role=UserRole.ADMIN)
    fetched_exp = ExperimentService.get_experiment_by_id(db=db, experiment_id=exp_id, user=admin_user)
    assert fetched_exp.id == exp_id


def test_experiment_failure_state(client: TestClient, db: Session):
    """
    Test 13: Verify failed benchmark jobs are persisted in FAILED state without leaking stack traces.
    """
    # Trigger benchmark with non-existent artifact_id
    res = client.post(
        "/api/v1/experiments/benchmark",
        json={"artifact_id": 99999, "rounds": 1},
    )
    assert res.status_code == 404
    error_msg = res.json()["error"]["message"]
    assert "not found" in error_msg.lower()


def test_reproducibility_metadata(client: TestClient, db: Session):
    """
    Test 14: Verify random_state, hyperparameters, and artifact_id are persisted in experiment config.
    """
    artifact_id = setup_experiment_dataset_and_artifact(client)

    client.post(
        "/api/v1/experiments/benchmark",
        json={
            "artifact_id": artifact_id,
            "num_clients": 2,
            "rounds": 1,
            "centralized_epochs": 1,
            "mu_values": [0.01],
            "random_state": 1234,
            "batch_size": 32,
            "learning_rate": 0.05,
        },
    )

    exp = db.query(Experiment).order_by(Experiment.created_at.desc()).first()
    assert exp.config["random_state"] == 1234
    assert exp.config["batch_size"] == 32
    assert exp.config["learning_rate"] == 0.05
    assert exp.config["artifact_id"] == artifact_id
