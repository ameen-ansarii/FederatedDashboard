"""
Phase 9D — Multi-Seed Scientific Evaluation Runner.

Executes a 5-seed controlled scientific benchmark on the audited OULAD dataset
and preprocessing artifact (v_1_1786778298).

Seeds evaluated: [42, 123, 456, 789, 1024]
Variants per seed:
    1. Centralized Baseline (10 epochs)
    2. FedAvg IID (5 rounds, 2 local epochs)
    3. FedAvg Non-IID (Dirichlet alpha=0.5, 5 rounds, 2 local epochs)
    4. FedProx Non-IID mu=0.001 (5 rounds, 2 local epochs)
    5. FedProx Non-IID mu=0.01 (5 rounds, 2 local epochs)
    6. FedProx Non-IID mu=0.1 (5 rounds, 2 local epochs)

All models evaluate on the exact same 40-sample test split from artifact v_1_1786778298.
"""

import json
import os
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.core.reproducibility import model_state_checksum
from app.models.domain import Dataset, PreprocessingArtifact, ModelRegistry
from app.schemas.experiment import BenchmarkConfig
from app.services.experiment_service import ExperimentService
from app.services.model_service import ModelService

SEEDS = [42, 123, 456, 789, 1024]
ARTIFACT_VERSION = "v_1_1786778298"

def main():
    print("=" * 80)
    print(f"PHASE 9D — MULTI-SEED SCIENTIFIC EVALUATION ({len(SEEDS)} SEEDS)")
    print(f"Seeds: {SEEDS}")
    print(f"Audited Artifact: {ARTIFACT_VERSION}")
    print("=" * 80)

    # Verify artifact files exist
    artifact_rel_path = f"artifacts/preprocessing/{ARTIFACT_VERSION}"
    splits_path = os.path.join("storage_data", artifact_rel_path, "splits.npz")
    preprocessor_path = os.path.join("storage_data", artifact_rel_path, "preprocessor.joblib")

    if not os.path.exists(splits_path) or not os.path.exists(preprocessor_path):
        print(f"ERROR: Artifact files not found in storage_data/{artifact_rel_path}")
        sys.exit(1)

    splits_data = np.load(splits_path)
    X_train = splits_data["X_train"]
    y_train = splits_data["y_train"]
    X_val = splits_data["X_val"]
    y_val = splits_data["y_val"]
    X_test = splits_data["X_test"]
    y_test = splits_data["y_test"]

    print(f"\nDataset Verification:")
    print(f"  Train samples: {len(y_train)} (Class 0: {int(np.sum(y_train==0))}, Class 1: {int(np.sum(y_train==1))})")
    print(f"  Val samples:   {len(y_val)} (Class 0: {int(np.sum(y_val==0))}, Class 1: {int(np.sum(y_val==1))})")
    print(f"  Test samples:  {len(y_test)} (Class 0: {int(np.sum(y_test==0))}, Class 1: {int(np.sum(y_test==1))})")
    print(f"  Feature count: {X_train.shape[1]}")

    # Initialize in-memory SQLite DB
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()

    # Register Dataset and PreprocessingArtifact in DB
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
        version=ARTIFACT_VERSION,
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

    all_seed_results: Dict[int, Dict[str, Any]] = {}
    model_checksums_by_seed: Dict[int, Dict[str, str]] = {}

    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\n[{seed_idx}/{len(SEEDS)}] Running Benchmark for Base Seed = {seed}...")

        benchmark_cfg = BenchmarkConfig(
            artifact_id=artifact_rec.id,
            name=f"OULAD Multi-Seed Study (Seed {seed})",
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
            random_state=seed,
        )

        exp_id = f"oulad-phase9d-seed-{seed}"
        experiment = ExperimentService.run_benchmark(
            db=db,
            experiment_id=exp_id,
            config=benchmark_cfg,
        )

        # Collect checksums for models generated in this experiment run
        seed_csums = {}
        for m_summary in experiment.results["models"]:
            mid = m_summary["model_id"]
            m_rec = db.query(ModelRegistry).filter(ModelRegistry.id == mid).first()
            if m_rec:
                py_model = ModelService.load_pytorch_model(m_rec)
                seed_csums[m_summary["variant"]] = model_state_checksum(py_model)
        model_checksums_by_seed[seed] = seed_csums

        all_seed_results[seed] = experiment.results
        print(f"  ✓ Seed {seed} completed (6 models trained and evaluated on held-out test split)")

    # Aggregate Statistics across all 5 seeds
    variants = [
        ("CENTRALIZED", "BASELINE"),
        ("FEDAVG", "IID"),
        ("FEDAVG", "NON_IID_ALPHA_0.5"),
        ("FEDPROX", "NON_IID_ALPHA_0.5_MU_0.001"),
        ("FEDPROX", "NON_IID_ALPHA_0.5_MU_0.01"),
        ("FEDPROX", "NON_IID_ALPHA_0.5_MU_0.1"),
    ]

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "loss"]
    aggregated_stats: Dict[str, Dict[str, Any]] = {}

    for algo, var in variants:
        key = f"{algo} ({var})"
        var_data: Dict[str, List[float]] = {m: [] for m in metric_names}
        var_deltas: Dict[str, List[float]] = {"delta_accuracy": [], "delta_f1": [], "delta_roc_auc": [], "delta_loss": []}
        seed_runs = []

        for seed in SEEDS:
            res = all_seed_results[seed]
            # Find matching model in experiment results
            match = None
            for m in res["models"]:
                if m["algorithm"] == algo and (m["variant"] == var or var in m["variant"]):
                    match = m
                    break
            if not match:
                continue

            for m in metric_names:
                var_data[m].append(match["metrics"][m])

            if "delta_from_baseline" in match:
                for d_k in var_deltas:
                    var_deltas[d_k].append(match["delta_from_baseline"][d_k])

            seed_runs.append({
                "seed": seed,
                "metrics": match["metrics"],
                "checksum": model_checksums_by_seed[seed].get(match["variant"], "N/A"),
                "derived_seed": res["reproducibility"]["derived_seeds"].get(
                    "centralized" if algo == "CENTRALIZED" else
                    "fedavg_iid" if var == "IID" else
                    "fedavg_non_iid" if var == "NON_IID_ALPHA_0.5" else
                    f"fedprox_mu_{match['variant'].split('MU_')[-1]}"
                ),
            })

        summary = {}
        for m in metric_names:
            vals = np.array(var_data[m])
            summary[f"{m}_mean"] = float(np.mean(vals))
            summary[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[f"{m}_min"] = float(np.min(vals))
            summary[f"{m}_max"] = float(np.max(vals))

        delta_summary = {}
        for d_k in var_deltas:
            vals = np.array(var_deltas[d_k])
            delta_summary[f"{d_k}_mean"] = float(np.mean(vals))
            delta_summary[f"{d_k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        aggregated_stats[key] = {
            "algorithm": algo,
            "variant": var,
            "metrics_summary": summary,
            "deltas_summary": delta_summary,
            "seed_runs": seed_runs,
        }

    # Print Formatted Multi-Seed Summary Table
    print("\n" + "=" * 120)
    print(f"{'Model Variant':<38} | {'Accuracy (Mean±Std)':<20} | {'F1 (Mean±Std)':<18} | {'ROC-AUC (Mean±Std)':<20} | {'Loss (Mean±Std)':<18}")
    print("-" * 120)
    for key, data in aggregated_stats.items():
        s = data["metrics_summary"]
        acc_str = f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}"
        f1_str = f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
        auc_str = f"{s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f}"
        loss_str = f"{s['loss_mean']:.4f} ± {s['loss_std']:.4f}"
        print(f"{key:<38} | {acc_str:<20} | {f1_str:<18} | {auc_str:<20} | {loss_str:<18}")

    print("\n" + "=" * 120)
    print("DETAILED SEED-BY-SEED METRICS (F1 / Accuracy / ROC-AUC):")
    print(f"{'Model Variant':<38} | " + " | ".join([f"Seed {s} (F1/Acc)" for s in SEEDS]))
    print("-" * 120)
    for key, data in aggregated_stats.items():
        seed_cells = []
        for run in data["seed_runs"]:
            f1_val = run["metrics"]["f1"]
            acc_val = run["metrics"]["accuracy"]
            seed_cells.append(f"{f1_val:.4f} / {acc_val:.4f}")
        print(f"{key:<38} | " + " | ".join(seed_cells))

    # Save complete Phase 9D JSON
    final_output = {
        "dataset": "OULAD Student Dropout Benchmark",
        "preprocessing_artifact": ARTIFACT_VERSION,
        "seeds": SEEDS,
        "benchmark_parameters": {
            "num_clients": 3,
            "rounds": 5,
            "local_epochs": 2,
            "centralized_epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.01,
            "hidden_dims": [32, 16],
            "dropout_rate": 0.1,
            "dirichlet_alpha": 0.5,
            "mu_values": [0.001, 0.01, 0.1],
        },
        "aggregated_statistics": aggregated_stats,
        "per_seed_experiments": all_seed_results,
    }

    out_path = "storage_data/phase9d_multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved complete multi-seed results to {out_path}")

if __name__ == "__main__":
    main()
