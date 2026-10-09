import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from app.core.reproducibility import build_reproducibility_metadata, seed_everything
from app.ml.federated.fedprox_client import DropoutFedProxFlowerClient
from app.ml.federated.strategy import aggregate_fedavg
from app.ml.models import DropoutMLP
from app.ml.partitioning import partition_dataset
from app.models.domain import FederatedClient, JobStatus, ModelRegistry, TrainingRun, User
from app.schemas.federated import FedProxTrainingConfig
from app.services.model_service import ModelService
from app.services.preprocessing_service import PreprocessingService


class FedProxService:
    @staticmethod
    def run_fedprox_training(
        db: Session,
        run_id: str,
        config: FedProxTrainingConfig,
        user: Optional[User] = None,
    ) -> Tuple[ModelRegistry, TrainingRun]:
        # 1. Fetch exact Phase 2 preprocessing artifact and splits
        artifact = PreprocessingService.get_artifact(db, config.artifact_id)
        preprocessor, splits = PreprocessingService.load_preprocessor_and_splits(artifact)

        X_train_full = splits["X_train"].astype(np.float32)
        y_train_full = splits["y_train"].astype(np.float32)
        X_val = splits["X_val"].astype(np.float32)
        y_val = splits["y_val"].astype(np.float32)
        X_test = splits["X_test"].astype(np.float32)
        y_test = splits["y_test"].astype(np.float32)

        input_dim = int(X_train_full.shape[1])

        seed_everything(config.random_state)

        # 2. Partition dataset across num_clients
        partitions = partition_dataset(
            X_train=X_train_full,
            y_train=y_train_full,
            num_clients=config.num_clients,
            strategy=config.partitioning,
            alpha=config.dirichlet_alpha,
            seed=config.random_state,
        )

        client_sample_counts = {f"client_{i}": int(len(p["y"])) for i, p in enumerate(partitions)}

        # Register or update FederatedClient entities in DB
        for i in range(config.num_clients):
            client_db_id = f"client-node-{i+1}"
            fed_client = db.query(FederatedClient).filter(FederatedClient.id == client_db_id).first()
            if not fed_client:
                fed_client = FederatedClient(
                    id=client_db_id,
                    name=f"Federated Client Node {i+1}",
                    status="TRAINING",
                    user_id=user.id if user else None,
                )
                db.add(fed_client)
            else:
                fed_client.status = "TRAINING"
        db.commit()

        # 3. Instantiate initial global PyTorch model
        global_model = DropoutMLP(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
        )

        initial_weights = [val.cpu().numpy() for val in global_model.state_dict().values()]
        current_global_weights = initial_weights

        # 4. Instantiate FedProx Flower clients with configurable mu parameter
        flower_clients: List[DropoutFedProxFlowerClient] = []
        for i, p in enumerate(partitions):
            fc = DropoutFedProxFlowerClient(
                client_id=f"client_{i}",
                input_dim=input_dim,
                X_train=p["X"],
                y_train=p["y"],
                hidden_dims=config.hidden_dims,
                dropout_rate=config.dropout_rate,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                local_epochs=config.local_epochs,
                mu=config.mu,
                optimizer_name=config.optimizer,
                random_state=config.random_state,
            )
            flower_clients.append(fc)

        history: Dict[str, List[Any]] = {
            "round": [],
            "val_loss": [],
            "val_accuracy": [],
            "prox_loss": [],
        }

        criterion = nn.BCEWithLogitsLoss()

        # 5. Execute FedProx Federated Rounds
        for fl_round in range(1, config.rounds + 1):
            fit_results: List[Tuple[List[np.ndarray], int]] = []
            round_prox_losses = []

            for fc in flower_clients:
                weights_k, num_examples_k, fit_metrics = fc.fit(
                    current_global_weights, {"round": fl_round}
                )
                fit_results.append((weights_k, num_examples_k))
                if "prox_loss" in fit_metrics:
                    round_prox_losses.append(fit_metrics["prox_loss"])

            # Sample-weighted parameter aggregation
            current_global_weights = aggregate_fedavg(fit_results)

            # Evaluate aggregated global model on validation set
            params_dict = zip(global_model.state_dict().keys(), current_global_weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            global_model.load_state_dict(state_dict)

            global_model.eval()
            with torch.no_grad():
                val_logits = global_model(torch.tensor(X_val))
                val_loss = float(criterion(val_logits, torch.tensor(y_val).unsqueeze(1)).item())
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_preds = (val_probs >= 0.5).astype(int)
                val_acc = float(accuracy_score(y_val, val_preds))

            history["round"].append(fl_round)
            history["val_loss"].append(round(val_loss, 4))
            history["val_accuracy"].append(round(val_acc, 4))
            history["prox_loss"].append(round(float(np.mean(round_prox_losses)), 4) if round_prox_losses else 0.0)

        # 6. Final Evaluation on held-out Test Set
        global_model.eval()
        with torch.no_grad():
            test_logits = global_model(torch.tensor(X_test))
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs >= 0.5).astype(int)

        accuracy = float(accuracy_score(y_test, test_preds))
        precision = float(precision_score(y_test, test_preds, zero_division=0))
        recall = float(recall_score(y_test, test_preds, zero_division=0))
        f1 = float(f1_score(y_test, test_preds, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5
        conf_matrix = confusion_matrix(y_test, test_preds).tolist()

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": conf_matrix,
        }

        hyperparameters = config.model_dump()
        hyperparameters["algorithm"] = "FEDPROX"
        hyperparameters["reproducibility"] = build_reproducibility_metadata(
            base_random_state=config.base_random_state or config.random_state,
            derived_seed=config.random_state,
            algorithm="FEDPROX",
            variant=config.partitioning.upper(),
            mu=config.mu,
            num_clients=config.num_clients,
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            dirichlet_alpha=config.dirichlet_alpha if config.partitioning.upper() == "NON_IID" else None,
        )
        hyperparameters["partitioning_details"] = {
            "partitioning": config.partitioning,
            "dirichlet_alpha": config.dirichlet_alpha,
            "num_clients": config.num_clients,
            "client_sample_counts": client_sample_counts,
        }

        # 7. Persist Aggregated FedProx Global Model to Registry
        model_record = ModelService.save_model(
            db=db,
            name=config.name,
            model=global_model,
            artifact=artifact,
            hyperparameters=hyperparameters,
            metrics=metrics,
            user=user,
        )

        # 8. Record TrainingRun entry
        training_run = TrainingRun(
            id=run_id,
            type="FEDPROX",
            status=JobStatus.SUCCESS,
            hyperparameters=hyperparameters,
            metrics=metrics,
            history=history,
            user_id=user.id if user else None,
        )
        db.add(training_run)

        # Reset federated client statuses to IDLE
        for i in range(config.num_clients):
            client_db_id = f"client-node-{i+1}"
            fed_client = db.query(FederatedClient).filter(FederatedClient.id == client_db_id).first()
            if fed_client:
                fed_client.status = "IDLE"

        db.commit()
        db.refresh(training_run)

        return model_record, training_run
