import io
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ml.models import DropoutMLP
from app.ml.trainer import CentralizedTrainer
from app.models.domain import JobStatus, ModelRegistry, PreprocessingArtifact, TrainingRun
from app.services.model_service import ModelService
from app.services.preprocessing_service import PreprocessingService


def create_sample_training_csv() -> bytes:
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "Age": np.random.randint(18, 30, size=n),
        "GPA": np.round(np.random.uniform(1.5, 4.0, size=n), 2),
        "Attendance": np.random.uniform(50.0, 100.0, size=n),
        "Tuition_Paid": np.random.choice(["Yes", "No"], size=n),
        "Target": np.random.choice(["Dropout", "Graduate"], size=n, p=[0.35, 0.65])
    })
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def test_dropout_mlp_architecture():
    model = DropoutMLP(input_dim=5, hidden_dims=[32, 16], dropout_rate=0.1)
    dummy_input = torch.randn(10, 5)
    logits = model(dummy_input)

    assert logits.shape == (10, 1)
    assert model.input_dim == 5
    assert model.hidden_dims == [32, 16]


def test_trainer_weighted_loss_and_metrics():
    # Synthetic splits with class imbalance (10 pos, 40 neg)
    X_train = np.random.randn(50, 4)
    y_train = np.array([1] * 10 + [0] * 40)

    X_val = np.random.randn(20, 4)
    y_val = np.array([1] * 5 + [0] * 15)

    X_test = np.random.randn(20, 4)
    y_test = np.array([1] * 5 + [0] * 15)

    splits = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": ["f1", "f2", "f3", "f4"],
    }

    trainer = CentralizedTrainer(
        input_dim=4,
        hidden_dims=[16, 8],
        epochs=5,
        batch_size=10,
        random_state=42,
    )
    model, metrics, history = trainer.train_and_evaluate(splits)

    # Class weight on y_train should be 40 / 10 = 4.0
    assert metrics["class_weight_pos"] == 4.0
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert len(history["epoch"]) == 5
    assert len(history["train_loss"]) == 5


def test_end_to_end_centralized_training_pipeline(client: TestClient, db: Session):
    csv_bytes = create_sample_training_csv()

    # 1. Upload Dataset
    upload_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Centralized ML Training Dataset", "target_column": "Target"},
        files={"file": ("student_data.csv", csv_bytes, "text/csv")}
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

    # 3. Trigger Centralized Training API
    train_res = client.post(
        "/api/v1/training/centralized",
        json={
            "artifact_id": artifact.id,
            "name": "Centralized Dropout MLP v1",
            "hidden_dims": [32, 16],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 10,
            "optimizer": "adam",
            "random_state": 42
        }
    )
    assert train_res.status_code == 202
    job_data = train_res.json()
    assert job_data["type"] == "CENTRALIZED_TRAINING"
    assert job_data["status"] == JobStatus.SUCCESS
    job_id = job_data["id"]

    # 4. Fetch Training Run
    run_res = client.get(f"/api/v1/training/runs/{job_id}")
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["status"] == JobStatus.SUCCESS
    assert "accuracy" in run_data["metrics"]
    assert "confusion_matrix" in run_data["metrics"]
    assert len(run_data["history"]["epoch"]) == 10

    # 5. Fetch Models List and Model Detail
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    models_list = models_res.json()
    assert len(models_list) >= 1
    model_id = models_list[0]["id"]

    detail_res = client.get(f"/api/v1/models/{model_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["name"] == "Centralized Dropout MLP v1"
    assert detail_data["framework"] == "PyTorch"
    assert "checksum" in detail_data
    assert len(detail_data["checksum"]) == 64  # SHA256 hex string

    # Lineage Verification
    lineage = detail_data["lineage"]
    assert lineage["dataset_id"] == dataset_id
    assert lineage["preprocessing_artifact_id"] == artifact.id
    assert lineage["preprocessing_version"] == artifact.version
    assert lineage["hyperparameters"]["epochs"] == 10

    # 6. Verify Model Load & Inference via ModelService
    model_db_record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    loaded_model = ModelService.load_pytorch_model(model_db_record)
    assert isinstance(loaded_model, DropoutMLP)

    # Perform sample inference
    feature_count = artifact.feature_schema["feature_count"]
    sample_input = torch.randn(2, feature_count)
    with torch.no_grad():
        out_logits = loaded_model(sample_input)
    assert out_logits.shape == (2, 1)
