import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.errors import BadRequestException, ForbiddenException, NotFoundException
from app.core.reproducibility import derive_benchmark_run_seed
from app.ml.models import DropoutMLP
from app.ml.partitioning import partition_dataset
from app.models.domain import (
    Experiment,
    JobStatus,
    ModelRegistry,
    PreprocessingArtifact,
    TrainingRun,
    User,
    UserRole,
)
from app.schemas.experiment import (
    BenchmarkConfig,
    ConvergenceTrajectoryResponse,
    EvaluationMetrics,
    ExperimentListItem,
    ExperimentListResponse,
    ExperimentResponse,
    ModelComparisonRequest,
    ModelEvaluationSummary,
    MuSensitivityResponse,
)
from app.schemas.federated import FedAvgTrainingConfig, FedProxTrainingConfig
from app.schemas.training import CentralizedTrainingConfig
from app.services.fedavg_service import FedAvgService
from app.services.fedprox_service import FedProxService
from app.services.model_service import ModelService
from app.services.preprocessing_service import PreprocessingService
from app.services.training_service import TrainingService


class ExperimentService:
    @staticmethod
    def evaluate_model_on_test_split(
        model: DropoutMLP, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates a PyTorch DropoutMLP model on held-out test data.
        Returns accuracy, precision, recall, f1, roc_auc, loss, and confusion matrix.
        """
        model.eval()
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            tensor_X = torch.tensor(X_test, dtype=torch.float32)
            tensor_y = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            logits = model(tensor_X)
            loss = float(criterion(logits, tensor_y).item())
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

        y_int = y_test.astype(int)
        accuracy = float(accuracy_score(y_int, preds))
        precision = float(precision_score(y_int, preds, zero_division=0))
        recall = float(recall_score(y_int, preds, zero_division=0))
        f1 = float(f1_score(y_int, preds, zero_division=0))
        roc_auc = float(roc_auc_score(y_int, probs)) if len(np.unique(y_int)) > 1 else 0.5
        conf_mat = confusion_matrix(y_int, preds).tolist()

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "loss": round(loss, 4),
            "confusion_matrix": conf_mat,
        }

    @staticmethod
    def _compute_client_distribution_metadata(
        X_train: np.ndarray, y_train: np.ndarray, num_clients: int, strategy: str, alpha: float, seed: int
    ) -> Dict[str, Any]:
        """
        Computes detailed client-by-client sample counts and class balance distributions.
        """
        partitions = partition_dataset(
            X_train=X_train,
            y_train=y_train,
            num_clients=num_clients,
            strategy=strategy,
            alpha=alpha,
            seed=seed,
        )

        client_stats = {}
        for i, p in enumerate(partitions):
            y_c = p["y"]
            c0 = int(np.sum(y_c == 0))
            c1 = int(np.sum(y_c == 1))
            total = len(y_c)
            client_stats[f"client_{i+1}"] = {
                "sample_count": total,
                "class_0_count": c0,
                "class_1_count": c1,
                "class_0_ratio": round(c0 / total, 4) if total > 0 else 0.0,
                "class_1_ratio": round(c1 / total, 4) if total > 0 else 0.0,
            }

        return {
            "strategy": strategy,
            "num_clients": num_clients,
            "dirichlet_alpha": alpha if strategy == "NON_IID" else None,
            "total_samples": int(sum(len(p["y"]) for p in partitions)),
            "client_breakdown": client_stats,
        }

    @classmethod
    def run_benchmark(
        cls,
        db: Session,
        experiment_id: str,
        config: BenchmarkConfig,
        user: Optional[User] = None,
    ) -> Experiment:
        """
        Runs a comprehensive controlled evaluation comparing Centralized, FedAvg (IID),
        FedAvg (Non-IID), and FedProx (Non-IID with multiple μ values) on the same test split.
        """
        # 1. Fetch exact preprocessing artifact and test split
        artifact = PreprocessingService.get_artifact(db, config.artifact_id)
        preprocessor, splits = PreprocessingService.load_preprocessor_and_splits(artifact)

        X_train = splits["X_train"].astype(np.float32)
        y_train = splits["y_train"].astype(np.float32)
        X_test = splits["X_test"].astype(np.float32)
        y_test = splits["y_test"].astype(np.float32)

        # 2. Compute client distribution statistics (seeds match training sub-runs)
        iid_seed = derive_benchmark_run_seed(config.random_state, "FEDAVG", "IID")
        non_iid_seed = derive_benchmark_run_seed(
            config.random_state, "FEDAVG", "NON_IID", dirichlet_alpha=config.dirichlet_alpha
        )
        iid_distribution = cls._compute_client_distribution_metadata(
            X_train, y_train, config.num_clients, "IID", config.dirichlet_alpha, iid_seed
        )
        non_iid_distribution = cls._compute_client_distribution_metadata(
            X_train, y_train, config.num_clients, "NON_IID", config.dirichlet_alpha, non_iid_seed
        )

        derived_seeds: Dict[str, int] = {
            "centralized": derive_benchmark_run_seed(config.random_state, "CENTRALIZED", "BASELINE"),
            "fedavg_iid": iid_seed,
            "fedavg_non_iid": non_iid_seed,
        }

        # Initialize or retrieve Experiment DB record
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            experiment = Experiment(
                id=experiment_id,
                name=config.name or "FL vs Centralized Benchmark",
                status=JobStatus.RUNNING,
                artifact_id=config.artifact_id,
                config=config.model_dump(),
                user_id=user.id if user else None,
            )
            db.add(experiment)
            db.commit()
            db.refresh(experiment)
        else:
            experiment.status = JobStatus.RUNNING
            db.commit()

        model_summaries: List[Dict[str, Any]] = []
        trajectories: Dict[str, Dict[str, List[float]]] = {}
        mu_analysis: List[Dict[str, Any]] = []

        # ==========================================
        # RUN 1: Centralized Baseline
        # ==========================================
        cent_derived_seed = derived_seeds["centralized"]
        cent_cfg = CentralizedTrainingConfig(
            artifact_id=config.artifact_id,
            epochs=config.centralized_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            random_state=cent_derived_seed,
            base_random_state=config.random_state,
        )
        cent_model_rec, cent_run = TrainingService.run_centralized_training(
            db=db, run_id=f"{experiment_id}-cent", config=cent_cfg, user=user
        )
        cent_run.experiment_id = experiment_id
        db.commit()

        cent_pytorch_model = ModelService.load_pytorch_model(cent_model_rec)
        cent_metrics = cls.evaluate_model_on_test_split(cent_pytorch_model, X_test, y_test)

        cent_summary = {
            "model_id": cent_model_rec.id,
            "model_version": cent_model_rec.version,
            "algorithm": "CENTRALIZED",
            "variant": "BASELINE",
            "metrics": cent_metrics,
            "delta_from_baseline": {"delta_accuracy": 0.0, "delta_f1": 0.0, "delta_roc_auc": 0.0, "delta_loss": 0.0},
        }
        model_summaries.append(cent_summary)

        if cent_run.history:
            trajectories["CENTRALIZED"] = {
                "epochs": cent_run.history.get("epoch", []),
                "loss": cent_run.history.get("val_loss", []),
                "validation_accuracy": cent_run.history.get("val_accuracy", []),
            }

        # ==========================================
        # RUN 2: FedAvg IID
        # ==========================================
        fedavg_iid_derived_seed = derived_seeds["fedavg_iid"]
        fedavg_iid_cfg = FedAvgTrainingConfig(
            artifact_id=config.artifact_id,
            num_clients=config.num_clients,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            partitioning="IID",
            random_state=fedavg_iid_derived_seed,
            base_random_state=config.random_state,
        )
        fedavg_iid_model, fedavg_iid_run = FedAvgService.run_fedavg_training(
            db=db, run_id=f"{experiment_id}-fedavg-iid", config=fedavg_iid_cfg, user=user
        )
        fedavg_iid_run.experiment_id = experiment_id
        db.commit()

        fedavg_iid_pytorch = ModelService.load_pytorch_model(fedavg_iid_model)
        fedavg_iid_metrics = cls.evaluate_model_on_test_split(fedavg_iid_pytorch, X_test, y_test)

        fedavg_iid_summary = {
            "model_id": fedavg_iid_model.id,
            "model_version": fedavg_iid_model.version,
            "algorithm": "FEDAVG",
            "variant": "IID",
            "metrics": fedavg_iid_metrics,
            "delta_from_baseline": {
                "delta_accuracy": round(fedavg_iid_metrics["accuracy"] - cent_metrics["accuracy"], 4),
                "delta_f1": round(fedavg_iid_metrics["f1"] - cent_metrics["f1"], 4),
                "delta_roc_auc": round(fedavg_iid_metrics["roc_auc"] - cent_metrics["roc_auc"], 4),
                "delta_loss": round(fedavg_iid_metrics["loss"] - cent_metrics["loss"], 4),
            },
        }
        model_summaries.append(fedavg_iid_summary)

        if fedavg_iid_run.history:
            trajectories["FEDAVG_IID"] = {
                "rounds": fedavg_iid_run.history.get("round", []),
                "loss": fedavg_iid_run.history.get("val_loss", []),
                "validation_accuracy": fedavg_iid_run.history.get("val_accuracy", []),
            }

        # ==========================================
        # RUN 3: FedAvg Non-IID
        # ==========================================
        fedavg_non_iid_derived_seed = derived_seeds["fedavg_non_iid"]
        fedavg_non_iid_cfg = FedAvgTrainingConfig(
            artifact_id=config.artifact_id,
            num_clients=config.num_clients,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
            partitioning="NON_IID",
            dirichlet_alpha=config.dirichlet_alpha,
            random_state=fedavg_non_iid_derived_seed,
            base_random_state=config.random_state,
        )
        fedavg_non_iid_model, fedavg_non_iid_run = FedAvgService.run_fedavg_training(
            db=db, run_id=f"{experiment_id}-fedavg-non-iid", config=fedavg_non_iid_cfg, user=user
        )
        fedavg_non_iid_run.experiment_id = experiment_id
        db.commit()

        fedavg_non_iid_pytorch = ModelService.load_pytorch_model(fedavg_non_iid_model)
        fedavg_non_iid_metrics = cls.evaluate_model_on_test_split(fedavg_non_iid_pytorch, X_test, y_test)

        fedavg_non_iid_summary = {
            "model_id": fedavg_non_iid_model.id,
            "model_version": fedavg_non_iid_model.version,
            "algorithm": "FEDAVG",
            "variant": f"NON_IID_ALPHA_{config.dirichlet_alpha}",
            "metrics": fedavg_non_iid_metrics,
            "delta_from_baseline": {
                "delta_accuracy": round(fedavg_non_iid_metrics["accuracy"] - cent_metrics["accuracy"], 4),
                "delta_f1": round(fedavg_non_iid_metrics["f1"] - cent_metrics["f1"], 4),
                "delta_roc_auc": round(fedavg_non_iid_metrics["roc_auc"] - cent_metrics["roc_auc"], 4),
                "delta_loss": round(fedavg_non_iid_metrics["loss"] - cent_metrics["loss"], 4),
            },
        }
        model_summaries.append(fedavg_non_iid_summary)

        if fedavg_non_iid_run.history:
            trajectories[f"FEDAVG_NON_IID_ALPHA_{config.dirichlet_alpha}"] = {
                "rounds": fedavg_non_iid_run.history.get("round", []),
                "loss": fedavg_non_iid_run.history.get("val_loss", []),
                "validation_accuracy": fedavg_non_iid_run.history.get("val_accuracy", []),
            }

        # ==========================================
        # RUN 4..N: FedProx Non-IID (varying μ)
        # ==========================================
        for mu in config.mu_values:
            prox_derived_seed = derive_benchmark_run_seed(
                config.random_state,
                "FEDPROX",
                "NON_IID",
                dirichlet_alpha=config.dirichlet_alpha,
                mu=mu,
            )
            derived_seeds[f"fedprox_mu_{mu}"] = prox_derived_seed
            prox_cfg = FedProxTrainingConfig(
                artifact_id=config.artifact_id,
                num_clients=config.num_clients,
                rounds=config.rounds,
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                mu=mu,
                hidden_dims=config.hidden_dims,
                dropout_rate=config.dropout_rate,
                partitioning="NON_IID",
                dirichlet_alpha=config.dirichlet_alpha,
                random_state=prox_derived_seed,
                base_random_state=config.random_state,
            )
            prox_model, prox_run = FedProxService.run_fedprox_training(
                db=db, run_id=f"{experiment_id}-fedprox-mu-{mu}", config=prox_cfg, user=user
            )
            prox_run.experiment_id = experiment_id
            db.commit()

            prox_pytorch = ModelService.load_pytorch_model(prox_model)
            prox_metrics = cls.evaluate_model_on_test_split(prox_pytorch, X_test, y_test)

            prox_summary = {
                "model_id": prox_model.id,
                "model_version": prox_model.version,
                "algorithm": "FEDPROX",
                "variant": f"NON_IID_ALPHA_{config.dirichlet_alpha}_MU_{mu}",
                "metrics": prox_metrics,
                "delta_from_baseline": {
                    "delta_accuracy": round(prox_metrics["accuracy"] - cent_metrics["accuracy"], 4),
                    "delta_f1": round(prox_metrics["f1"] - cent_metrics["f1"], 4),
                    "delta_roc_auc": round(prox_metrics["roc_auc"] - cent_metrics["roc_auc"], 4),
                    "delta_loss": round(prox_metrics["loss"] - cent_metrics["loss"], 4),
                },
            }
            model_summaries.append(prox_summary)

            var_name = f"FEDPROX_MU_{mu}"
            if prox_run.history:
                trajectories[var_name] = {
                    "rounds": prox_run.history.get("round", []),
                    "loss": prox_run.history.get("val_loss", []),
                    "validation_accuracy": prox_run.history.get("val_accuracy", []),
                }

            mu_analysis.append(
                {
                    "mu": mu,
                    "accuracy": prox_metrics["accuracy"],
                    "f1": prox_metrics["f1"],
                    "roc_auc": prox_metrics["roc_auc"],
                    "loss": prox_metrics["loss"],
                    "delta_f1_vs_centralized": round(prox_metrics["f1"] - cent_metrics["f1"], 4),
                    "delta_f1_vs_fedavg_non_iid": round(prox_metrics["f1"] - fedavg_non_iid_metrics["f1"], 4),
                }
            )

        # 3. Compile full experiment results
        results_payload = {
            "models": model_summaries,
            "trajectories": trajectories,
            "mu_analysis": mu_analysis,
            "client_distributions": {
                "iid": iid_distribution,
                "non_iid": non_iid_distribution,
            },
            "reproducibility": {
                "base_random_state": config.random_state,
                "derived_seeds": derived_seeds,
                "artifact_id": config.artifact_id,
                "num_clients": config.num_clients,
                "rounds": config.rounds,
                "local_epochs": config.local_epochs,
                "centralized_epochs": config.centralized_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "hidden_dims": config.hidden_dims,
                "dropout_rate": config.dropout_rate,
                "dirichlet_alpha": config.dirichlet_alpha,
                "mu_values": config.mu_values,
            },
            "summary": {
                "baseline_centralized_f1": cent_metrics["f1"],
                "fedavg_iid_f1": fedavg_iid_metrics["f1"],
                "fedavg_non_iid_f1": fedavg_non_iid_metrics["f1"],
                "best_federated_model": max(
                    model_summaries[1:], key=lambda m: m["metrics"]["f1"]
                ),
            },
        }

        experiment.results = results_payload
        experiment.status = JobStatus.SUCCESS
        db.commit()
        db.refresh(experiment)
        return experiment

    @classmethod
    def compare_models(
        cls,
        db: Session,
        request: ModelComparisonRequest,
        user: Optional[User] = None,
    ) -> List[ModelEvaluationSummary]:
        """
        Evaluates and compares arbitrary registered models side-by-side on the same test split.
        """
        if not request.model_ids:
            raise BadRequestException("At least one model ID must be provided for comparison")

        models: List[ModelRegistry] = []
        for mid in request.model_ids:
            m = ModelService.get_model(db, mid)
            if user and user.role != UserRole.ADMIN and m.user_id is not None and m.user_id != user.id:
                raise ForbiddenException(f"You do not have access to model ID {mid}")
            models.append(m)

        # Determine target artifact for evaluation
        target_artifact_id = request.artifact_id or models[0].lineage.get("preprocessing_artifact_id")
        if not target_artifact_id:
            raise BadRequestException("No valid preprocessing artifact could be resolved for model comparison")

        artifact = PreprocessingService.get_artifact(db, target_artifact_id)
        _, splits = PreprocessingService.load_preprocessor_and_splits(artifact)

        X_test = splits["X_test"].astype(np.float32)
        y_test = splits["y_test"].astype(np.float32)
        expected_features = artifact.feature_schema.get("feature_count")

        eval_summaries: List[ModelEvaluationSummary] = []
        baseline_f1: Optional[float] = None

        for m in models:
            m_feat_count = m.lineage.get("feature_schema", {}).get("feature_count")
            if expected_features and m_feat_count and m_feat_count != expected_features:
                raise BadRequestException(
                    f"Model ID {m.id} feature dimensions ({m_feat_count}) do not match target artifact ({expected_features})"
                )

            pytorch_model = ModelService.load_pytorch_model(m)
            metrics_dict = cls.evaluate_model_on_test_split(pytorch_model, X_test, y_test)
            metrics_obj = EvaluationMetrics(**metrics_dict)

            if baseline_f1 is None:
                baseline_f1 = metrics_obj.f1
                deltas = {"delta_f1": 0.0, "delta_accuracy": 0.0}
            else:
                deltas = {
                    "delta_f1": round(metrics_obj.f1 - baseline_f1, 4),
                    "delta_accuracy": round(metrics_obj.accuracy - eval_summaries[0].metrics.accuracy, 4),
                }

            algo = m.lineage.get("hyperparameters", {}).get("algorithm", "UNKNOWN")
            eval_summaries.append(
                ModelEvaluationSummary(
                    model_id=m.id,
                    model_version=m.version,
                    algorithm=algo,
                    variant=f"MODEL_{m.id}",
                    metrics=metrics_obj,
                    delta_from_baseline=deltas,
                )
            )

        return eval_summaries

    @staticmethod
    def get_experiment_by_id(db: Session, experiment_id: str, user: Optional[User] = None) -> Experiment:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            raise NotFoundException(f"Experiment with ID '{experiment_id}' not found")
        if user and user.role != UserRole.ADMIN and exp.user_id is not None and exp.user_id != user.id:
            raise ForbiddenException("You do not have permission to access this experiment")
        return exp

    @staticmethod
    def list_experiments(
        db: Session, skip: int = 0, limit: int = 50, user: Optional[User] = None
    ) -> Tuple[List[Experiment], int]:
        query = db.query(Experiment)
        if user and user.role != UserRole.ADMIN:
            query = query.filter(Experiment.user_id == user.id)
        total = query.count()
        items = query.order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()
        return items, total
