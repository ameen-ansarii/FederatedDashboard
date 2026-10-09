"""
Phase 9B — Experiment reproducibility and seed isolation tests.

Verifies deterministic sub-run seeds, execution-order independence,
μ-order independence, partition reproducibility, and model initialization isolation.
"""

import io
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.reproducibility import (
    derive_benchmark_run_seed,
    derive_client_round_seed,
    derive_seed,
    model_state_checksum,
)
from app.ml.federated.client import DropoutFlowerClient
from app.ml.federated.strategy import aggregate_fedavg
from app.ml.models import DropoutMLP
from app.ml.partitioning import partition_dataset
from app.schemas.experiment import BenchmarkConfig
from app.schemas.federated import FedAvgTrainingConfig, FedProxTrainingConfig
from app.schemas.training import CentralizedTrainingConfig
from app.services.experiment_service import ExperimentService
from app.services.fedavg_service import FedAvgService
from app.services.fedprox_service import FedProxService
from app.services.model_service import ModelService
from app.services.training_service import TrainingService


def _create_repro_csv() -> bytes:
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "Age": rng.integers(18, 30, size=n),
            "GPA": np.round(rng.uniform(1.5, 4.0, size=n), 2),
            "Attendance": rng.uniform(50.0, 100.0, size=n),
            "Tuition_Paid": rng.choice(["Yes", "No"], size=n),
            "Target": rng.choice(["Dropout", "Graduate"], size=n, p=[0.35, 0.65]),
        }
    )
    output = io.BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


def _setup_artifact(client: TestClient) -> int:
    csv_bytes = _create_repro_csv()
    up_res = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Repro Test Dataset", "target_column": "Target"},
        files={"file": ("repro_data.csv", csv_bytes, "text/csv")},
    )
    assert up_res.status_code == 201
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
    assert prep_res.status_code == 202

    artifacts_res = client.get(f"/api/v1/datasets/{dataset_id}/artifacts")
    return artifacts_res.json()[0]["id"]


def _tiny_benchmark_config(artifact_id: int, seed: int = 42) -> BenchmarkConfig:
    return BenchmarkConfig(
        artifact_id=artifact_id,
        name="Repro Tiny Benchmark",
        num_clients=2,
        rounds=1,
        local_epochs=1,
        centralized_epochs=1,
        batch_size=16,
        learning_rate=0.01,
        hidden_dims=[16, 8],
        dropout_rate=0.1,
        dirichlet_alpha=0.5,
        mu_values=[0.001, 0.01],
        random_state=seed,
    )


import uuid


def _run_fedavg_variant(
    db: Session, artifact_id: int, base_seed: int, partitioning: str = "IID", run_suffix: str = ""
) -> dict:
    derived = derive_benchmark_run_seed(
        base_seed,
        "FEDAVG",
        partitioning,
        dirichlet_alpha=0.5 if partitioning == "NON_IID" else None,
    )
    cfg = FedAvgTrainingConfig(
        artifact_id=artifact_id,
        num_clients=2,
        rounds=1,
        local_epochs=1,
        batch_size=16,
        learning_rate=0.01,
        hidden_dims=[16, 8],
        dropout_rate=0.1,
        partitioning=partitioning,
        dirichlet_alpha=0.5,
        random_state=derived,
        base_random_state=base_seed,
    )
    run_id = f"repro-fedavg-{partitioning}-{base_seed}-{uuid.uuid4().hex[:8]}"
    model_rec, run = FedAvgService.run_fedavg_training(
        db=db, run_id=run_id, config=cfg
    )
    pytorch_model = ModelService.load_pytorch_model(model_rec)
    return {
        "metrics": run.metrics,
        "history": run.history,
        "checksum": model_state_checksum(pytorch_model),
        "derived_seed": derived,
    }


def _run_fedprox_variant(db: Session, artifact_id: int, base_seed: int, mu: float, run_suffix: str = "") -> dict:
    derived = derive_benchmark_run_seed(
        base_seed, "FEDPROX", "NON_IID", dirichlet_alpha=0.5, mu=mu
    )
    cfg = FedProxTrainingConfig(
        artifact_id=artifact_id,
        num_clients=2,
        rounds=1,
        local_epochs=1,
        batch_size=16,
        learning_rate=0.01,
        hidden_dims=[16, 8],
        dropout_rate=0.1,
        partitioning="NON_IID",
        dirichlet_alpha=0.5,
        mu=mu,
        random_state=derived,
        base_random_state=base_seed,
    )
    run_id = f"repro-fedprox-mu{mu}-{base_seed}-{uuid.uuid4().hex[:8]}"
    model_rec, run = FedProxService.run_fedprox_training(
        db=db, run_id=run_id, config=cfg
    )
    pytorch_model = ModelService.load_pytorch_model(model_rec)
    return {
        "metrics": run.metrics,
        "history": run.history,
        "checksum": model_state_checksum(pytorch_model),
        "derived_seed": derived,
    }


def test_same_seed_same_config_deterministic(client: TestClient, db: Session):
    """TEST 1: Identical seed + config produces identical benchmark outputs."""
    artifact_id = _setup_artifact(client)
    config = _tiny_benchmark_config(artifact_id, seed=42)

    exp_a = ExperimentService.run_benchmark(db, "repro-exp-a", config)
    exp_b = ExperimentService.run_benchmark(db, "repro-exp-b", config)

    assert exp_a.results["reproducibility"]["derived_seeds"] == exp_b.results["reproducibility"]["derived_seeds"]

    for model_a, model_b in zip(exp_a.results["models"], exp_b.results["models"]):
        assert model_a["metrics"] == model_b["metrics"]
        assert model_a["algorithm"] == model_b["algorithm"]

    assert exp_a.results["trajectories"] == exp_b.results["trajectories"]
    assert exp_a.results["client_distributions"] == exp_b.results["client_distributions"]


def test_different_seed_produces_different_derived_seeds(client: TestClient, db: Session):
    """TEST 2: Different base seeds produce independently derived experiment seeds."""
    artifact_id = _setup_artifact(client)

    exp_42 = ExperimentService.run_benchmark(
        db, "repro-seed-42", _tiny_benchmark_config(artifact_id, seed=42)
    )
    exp_123 = ExperimentService.run_benchmark(
        db, "repro-seed-123", _tiny_benchmark_config(artifact_id, seed=123)
    )

    seeds_42 = exp_42.results["reproducibility"]["derived_seeds"]
    seeds_123 = exp_123.results["reproducibility"]["derived_seeds"]

    assert seeds_42 != seeds_123
    assert exp_42.results["reproducibility"]["base_random_state"] == 42
    assert exp_123.results["reproducibility"]["base_random_state"] == 123


def test_execution_order_independence(client: TestClient, db: Session):
    """TEST 3: Sub-run results are identical regardless of prior sub-run execution."""
    artifact_id = _setup_artifact(client)
    base_seed = 42

    # Order A: Centralized context then FedAvg IID (simulate by running centralized first)
    cent_seed = derive_benchmark_run_seed(base_seed, "CENTRALIZED", "BASELINE")
    cent_cfg = CentralizedTrainingConfig(
        artifact_id=artifact_id,
        epochs=1,
        batch_size=16,
        learning_rate=0.01,
        hidden_dims=[16, 8],
        dropout_rate=0.1,
        random_state=cent_seed,
        base_random_state=base_seed,
    )
    TrainingService.run_centralized_training(db, f"repro-order-cent-{uuid.uuid4().hex[:8]}", cent_cfg)
    fedavg_after_cent = _run_fedavg_variant(db, artifact_id, base_seed, "IID")

    # Order B: FedAvg IID alone (fresh DB transaction state, isolated seed)
    fedavg_alone = _run_fedavg_variant(db, artifact_id, base_seed, "IID")

    assert fedavg_after_cent["metrics"] == fedavg_alone["metrics"]
    assert fedavg_after_cent["checksum"] == fedavg_alone["checksum"]
    assert fedavg_after_cent["history"] == fedavg_alone["history"]

    # FedProx μ=0.01: after μ=0.001 vs alone
    _run_fedprox_variant(db, artifact_id, base_seed, mu=0.001)
    prox_after = _run_fedprox_variant(db, artifact_id, base_seed, mu=0.01)
    prox_alone = _run_fedprox_variant(db, artifact_id, base_seed, mu=0.01)

    assert prox_after["metrics"] == prox_alone["metrics"]
    assert prox_after["checksum"] == prox_alone["checksum"]


def test_mu_order_independence(client: TestClient, db: Session):
    """TEST 4: Each μ value produces the same result regardless of evaluation order."""
    artifact_id = _setup_artifact(client)
    base_seed = 42
    mu_values = [0.001, 0.01, 0.1]

    forward_results = {}
    for mu in mu_values:
        forward_results[mu] = _run_fedprox_variant(db, artifact_id, base_seed, mu)

    reverse_results = {}
    for mu in reversed(mu_values):
        reverse_results[mu] = _run_fedprox_variant(db, artifact_id, base_seed, mu)

    for mu in mu_values:
        assert forward_results[mu]["metrics"] == reverse_results[mu]["metrics"]
        assert forward_results[mu]["checksum"] == reverse_results[mu]["checksum"]


def test_client_seed_order_independence():
    """TEST 5: Client-round seeds depend on client_id, not iteration order."""
    experiment_seed = derive_benchmark_run_seed(42, "FEDAVG", "IID")

    seeds_ordered = [
        derive_client_round_seed(experiment_seed, f"client_{i}", round_num=1)
        for i in range(3)
    ]
    seeds_reversed = [
        derive_client_round_seed(experiment_seed, f"client_{i}", round_num=1)
        for i in reversed(range(3))
    ]

    assert seeds_ordered == list(reversed(seeds_reversed))

    # Simulate one federated round with clients processed in opposite orders
    X = np.random.default_rng(0).random((60, 4)).astype(np.float32)
    y = np.array([1] * 20 + [0] * 40, dtype=np.float32)
    partitions = partition_dataset(X, y, num_clients=2, strategy="iid", seed=experiment_seed)

    global_model = DropoutMLP(input_dim=4, hidden_dims=[8, 4], dropout_rate=0.1)
    global_weights = [v.cpu().numpy() for v in global_model.state_dict().values()]

    def _run_round(client_order):
        clients = [
            DropoutFlowerClient(
                client_id=f"client_{i}",
                input_dim=4,
                X_train=partitions[i]["X"],
                y_train=partitions[i]["y"],
                hidden_dims=[8, 4],
                local_epochs=1,
                batch_size=16,
                random_state=experiment_seed,
            )
            for i in client_order
        ]
        results = []
        for fc in clients:
            w, n, _ = fc.fit(global_weights, {"round": 1})
            results.append((w, n))
        return aggregate_fedavg(results)

    agg_forward = _run_round([0, 1])
    agg_reverse = _run_round([1, 0])

    for a, b in zip(agg_forward, agg_reverse):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_partition_reproducibility():
    """TEST 6: Same seed + α yields identical partitions; different seeds differ."""
    X = np.random.default_rng(7).random((80, 5)).astype(np.float32)
    y = np.array([1] * 30 + [0] * 50, dtype=np.float32)

    part_a = partition_dataset(X, y, num_clients=3, strategy="non_iid", alpha=0.5, seed=42)
    part_b = partition_dataset(X, y, num_clients=3, strategy="non_iid", alpha=0.5, seed=42)
    part_other = partition_dataset(X, y, num_clients=3, strategy="non_iid", alpha=0.5, seed=99)

    for pa, pb in zip(part_a, part_b):
        np.testing.assert_array_equal(pa["X"], pb["X"])
        np.testing.assert_array_equal(pa["y"], pb["y"])

    assert any(
        not np.array_equal(pa["X"], po["X"])
        for pa, po in zip(part_a, part_other)
    )


def test_model_initialization_isolation():
    """TEST 7: Benchmark variants use independent derived seeds and fresh models."""
    base_seed = 42
    seeds = {
        "fedavg_iid": derive_benchmark_run_seed(base_seed, "FEDAVG", "IID"),
        "fedavg_non_iid": derive_benchmark_run_seed(
            base_seed, "FEDAVG", "NON_IID", dirichlet_alpha=0.5
        ),
        "fedprox": derive_benchmark_run_seed(
            base_seed, "FEDPROX", "NON_IID", dirichlet_alpha=0.5, mu=0.01
        ),
    }

    assert len(set(seeds.values())) == 3

    from app.core.reproducibility import seed_everything

    checksums = {}
    for name, seed in seeds.items():
        seed_everything(seed)
        model = DropoutMLP(input_dim=10, hidden_dims=[16, 8], dropout_rate=0.1)
        checksums[name] = model_state_checksum(model)

    assert len(set(checksums.values())) == 3


def test_derive_seed_properties():
    """Unit checks for the seed derivation contract."""
    assert derive_seed(42, "A") == derive_seed(42, "A")
    assert derive_seed(42, "A") != derive_seed(42, "B")
    assert derive_seed(42, "A") != derive_seed(123, "A")

    assert derive_benchmark_run_seed(42, "FEDPROX", "NON_IID", dirichlet_alpha=0.5, mu=0.001) != \
        derive_benchmark_run_seed(42, "FEDPROX", "NON_IID", dirichlet_alpha=0.5, mu=0.01)
