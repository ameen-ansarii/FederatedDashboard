"""
Phase 9C — OULAD Pilot Experiment Runner.

Executes a single controlled pilot experiment on the audited OULAD dataset & artifact (v_1_1786778298).
Evaluates:
    1. Centralized Baseline (10 epochs)
    2. FedAvg IID (5 rounds, 2 local epochs)
    3. FedAvg Non-IID (Dirichlet alpha=0.5, 5 rounds, 2 local epochs)
    4. FedProx Non-IID mu=0.001
    5. FedProx Non-IID mu=0.01
    6. FedProx Non-IID mu=0.1

Extracts all metrics, trajectories, client distributions, seeds, and model checksums.
"""

import json
import os
import sys
import numpy as np
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Setup in-memory SQLite DB
from app.core.database import Base
from app.core.reproducibility import model_state_checksum
from app.models.domain import Dataset, PreprocessingArtifact, ModelRegistry, TrainingRun, Experiment
from app.schemas.experiment import BenchmarkConfig
from app.services.experiment_service import ExperimentService
from app.services.model_service import ModelService

def main():
    print("=================================================================")
    print("PHASE 9C — OULAD PILOT EXPERIMENT EXECUTION")
    print("=================================================================")

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()

    # 1. Verify and register OULAD Preprocessing Artifact v_1_1786778298 in SQLite DB
    artifact_version = "v_1_1786778298"
    artifact_rel_path = f"artifacts/preprocessing/{artifact_version}"
    splits_path = os.path.join("storage_data", artifact_rel_path, "splits.npz")
    preprocessor_path = os.path.join("storage_data", artifact_rel_path, "preprocessor.joblib")

    if not os.path.exists(splits_path) or not os.path.exists(preprocessor_path):
        print(f"ERROR: Artifact files not found in {artifact_rel_path}")
        sys.exit(1)

    splits_data = np.load(splits_path)
    X_train = splits_data["X_train"]
    y_train = splits_data["y_train"]
    X_val = splits_data["X_val"]
    y_val = splits_data["y_val"]
    X_test = splits_data["X_test"]
    y_test = splits_data["y_test"]

    print(f"Verified Artifact: {artifact_version}")
    print(f"Train Shape: {X_train.shape}, Labels: {np.bincount(y_train)}")
    print(f"Val Shape:   {X_val.shape}, Labels: {np.bincount(y_val)}")
    print(f"Test Shape:  {X_test.shape}, Labels: {np.bincount(y_test)}")
    print(f"Feature Count: {X_train.shape[1]}")

    # Register Dataset & Artifact in DB
    dataset_rec = Dataset(
        id=1,
        name="OULAD Student Dropout Benchmark",
        filename="oulad_students.csv",
        file_path="storage_data/datasets/oulad_students.csv",
        row_count=200,
        feature_count=11,
        target_column="final_result",
    )
    db.add(dataset_rec)
    db.commit()

    artifact_rec = PreprocessingArtifact(
        id=1,
        dataset_id=1,
        version=artifact_version,
        artifact_path=artifact_rel_path,
        feature_schema={
            "feature_count": int(X_train.shape[1]),
            "categorical_features": ["code_module", "code_presentation", "gender", "highest_education", "imd_band", "age_band", "disability"],
            "numerical_features": ["num_of_prev_attempts", "studied_credits", "total_vle_clicks", "avg_assessment_score"],
        },
        target_mapping={"Withdrawn": 1, "Pass": 0, "Distinction": 0, "Fail": 0},
    )
    db.add(artifact_rec)
    db.commit()
    db.refresh(artifact_rec)

    # 2. Define BenchmarkConfig matching Phase 9C requirements
    benchmark_cfg = BenchmarkConfig(
        artifact_id=artifact_rec.id,
        name="OULAD Pilot Experiment Phase 9C",
        num_clients=3,
        rounds=5,
        local_epochs=2,
        centralized_epochs=10,
        batch_size=16,
        learning_rate=0.01,
        hidden_dims=[32, 16],
        dropout_rate=0.1,
        dirichlet_alpha=0.5,
        mu_values=[0.001, 0.01, 0.1],
        random_state=42,
    )

    print("\nExecuting Benchmark via ExperimentService.run_benchmark...")
    experiment = ExperimentService.run_benchmark(
        db=db,
        experiment_id="oulad-pilot-phase9c-run1",
        config=benchmark_cfg,
    )
    print("Benchmark Execution Complete!\n")

    results = experiment.results
    models_summary = results["models"]
    trajectories = results["trajectories"]
    mu_analysis = results["mu_analysis"]
    client_dists = results["client_distributions"]
    reproducibility = results["reproducibility"]

    # Compute model checksums for all registered models in this experiment
    model_records = db.query(ModelRegistry).all()
    checksums = {}
    for m in model_records:
        py_model = ModelService.load_pytorch_model(m)
        csum = model_state_checksum(py_model)
        checksums[m.id] = csum

    # Output detailed report sections
    print("=== MODEL RESULTS ===")
    print(f"{'Model':<35} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'Loss':<8} | {'Checksum'}")
    print("-" * 125)
    for m in models_summary:
        algo = m["algorithm"]
        var = m["variant"]
        metrics = m["metrics"]
        mid = m["model_id"]
        csum = checksums.get(mid, "N/A")[:16] + "..."
        name = f"{algo} ({var})"
        print(f"{name:<35} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<9.4f} | {metrics['recall']:<8.4f} | {metrics['f1']:<8.4f} | {metrics['roc_auc']:<8.4f} | {metrics['loss']:<8.4f} | {csum}")

    print("\n=== BASELINE DELTAS (relative to Centralized) ===")
    cent_metrics = models_summary[0]["metrics"]
    print(f"{'Model':<35} | {'Δ Accuracy':<12} | {'Δ F1':<12} | {'Δ ROC-AUC':<12} | {'Δ Loss':<12}")
    print("-" * 90)
    for m in models_summary:
        name = f"{m['algorithm']} ({m['variant']})"
        acc_d = m["metrics"]["accuracy"] - cent_metrics["accuracy"]
        f1_d = m["metrics"]["f1"] - cent_metrics["f1"]
        auc_d = m["metrics"]["roc_auc"] - cent_metrics["roc_auc"]
        loss_d = m["metrics"]["loss"] - cent_metrics["loss"]
        print(f"{name:<35} | {acc_d:<+12.4f} | {f1_d:<+12.4f} | {auc_d:<+12.4f} | {loss_d:<+12.4f}")

    print("\n=== CONVERGENCE TRAJECTORIES ===")
    for k, traj in trajectories.items():
        print(f"\nModel: {k}")
        if "epochs" in traj:
            print(f"  Epochs:   {traj['epochs']}")
        if "rounds" in traj:
            print(f"  Rounds:   {traj['rounds']}")
        print(f"  Val Loss: {traj['loss']}")
        print(f"  Val Acc:  {traj['validation_accuracy']}")

    print("\n=== CLIENT DISTRIBUTIONS (Non-IID alpha=0.5) ===")
    non_iid_breakdown = client_dists["non_iid"]["client_breakdown"]
    for c_id, stats in non_iid_breakdown.items():
        print(f"  {c_id}: Total={stats['sample_count']}, Class 0 (Non-Dropout)={stats['class_0_count']} ({stats['class_0_ratio']*100:.1f}%), Class 1 (Dropout)={stats['class_1_count']} ({stats['class_1_ratio']*100:.1f}%)")

    print("\n=== CLIENT DISTRIBUTIONS (IID) ===")
    iid_breakdown = client_dists["iid"]["client_breakdown"]
    for c_id, stats in iid_breakdown.items():
        print(f"  {c_id}: Total={stats['sample_count']}, Class 0 (Non-Dropout)={stats['class_0_count']} ({stats['class_0_ratio']*100:.1f}%), Class 1 (Dropout)={stats['class_1_count']} ({stats['class_1_ratio']*100:.1f}%)")

    print("\n=== FEDPROX MU SENSITIVITY ===")
    print(f"{'μ Value':<10} | {'Accuracy':<10} | {'F1':<10} | {'ROC-AUC':<10} | {'Loss':<10} | {'Δ F1 vs Centralized':<20} | {'Δ F1 vs FedAvg Non-IID':<22}")
    print("-" * 105)
    for row in mu_analysis:
        print(f"{row['mu']:<10} | {row['accuracy']:<10.4f} | {row['f1']:<10.4f} | {row['roc_auc']:<10.4f} | {row['loss']:<10.4f} | {row['delta_f1_vs_centralized']:<+20.4f} | {row['delta_f1_vs_fedavg_non_iid']:<+22.4f}")

    print("\n=== REPRODUCIBILITY SEEDS & METADATA ===")
    print(f"Base Random State: {reproducibility['base_random_state']}")
    for k, s in reproducibility["derived_seeds"].items():
        print(f"  {k:<20}: {s}")

    # Save full JSON report
    out_payload = {
        "benchmark_config": benchmark_cfg.model_dump(),
        "models": models_summary,
        "checksums": checksums,
        "trajectories": trajectories,
        "client_distributions": client_dists,
        "mu_analysis": mu_analysis,
        "reproducibility": reproducibility,
    }
    with open("storage_data/phase9c_pilot_results.json", "w") as f:
        json.dump(out_payload, f, indent=2)
    print("\nSaved full results to storage_data/phase9c_pilot_results.json")

if __name__ == "__main__":
    main()
