import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from app.core.reproducibility import seed_everything, get_torch_generator
from app.ml.models import DropoutMLP


class CentralizedTrainer:
    """
    Trainer for Centralized PyTorch DropoutMLP model.
    Handles class-weighted loss, epoch training/validation, test evaluation, and metrics calculation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,
        optimizer_name: str = "adam",
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer_name.lower()
        self.random_state = random_state

        seed_everything(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DropoutMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def train_and_evaluate(
        self,
        splits: Dict[str, np.ndarray],
    ) -> Tuple[DropoutMLP, Dict[str, Any], Dict[str, Any]]:
        """
        Trains the PyTorch model on X_train/y_train with weighted loss computed ONLY on y_train.
        Validates during training and evaluates on X_test/y_test after training.

        Returns:
            model: Trained DropoutMLP PyTorch module
            metrics: Test evaluation metrics dictionary
            history: Training/Validation loss and accuracy history per epoch
        """
        X_train_np = splits["X_train"].astype(np.float32)
        y_train_np = splits["y_train"].astype(np.float32)
        X_val_np = splits["X_val"].astype(np.float32)
        y_val_np = splits["y_val"].astype(np.float32)
        X_test_np = splits["X_test"].astype(np.float32)
        y_test_np = splits["y_test"].astype(np.float32)

        # 1. Compute weighted loss strictly based on training set y_train!
        n_pos = np.sum(y_train_np == 1.0)
        n_neg = np.sum(y_train_np == 0.0)
        pos_weight_val = float(n_neg / max(n_pos, 1.0))
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 2. Setup DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train_np), torch.tensor(y_train_np).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(X_val_np), torch.tensor(y_val_np).unsqueeze(1))
        test_dataset = TensorDataset(torch.tensor(X_test_np), torch.tensor(y_test_np).unsqueeze(1))

        train_gen = get_torch_generator(self.random_state)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=train_gen)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 3. Setup Optimizer
        if self.optimizer_name == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        # 4. Training Loop
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Validation step
            self.model.eval()
            val_losses = []
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    logits = self.model(X_b)
                    loss = criterion(logits, y_b)
                    val_losses.append(loss.item())

                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    val_preds.extend(preds.flatten())
                    val_targets.extend(y_b.cpu().numpy().flatten())

            avg_val_loss = float(np.mean(val_losses))
            val_acc = float(accuracy_score(val_targets, val_preds))

            history["epoch"].append(epoch)
            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_accuracy"].append(round(val_acc, 4))

        # 5. Final Evaluation on Test Set
        self.model.eval()
        test_probs = []
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                logits = self.model(X_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                test_probs.extend(probs.flatten())
                test_preds.extend(preds.flatten())
                test_targets.extend(y_b.cpu().numpy().flatten())

        test_targets = np.array(test_targets)
        test_preds = np.array(test_preds)
        test_probs = np.array(test_probs)

        accuracy = float(accuracy_score(test_targets, test_preds))
        precision = float(precision_score(test_targets, test_preds, zero_division=0))
        recall = float(recall_score(test_targets, test_preds, zero_division=0))
        f1 = float(f1_score(test_targets, test_preds, zero_division=0))

        if len(np.unique(test_targets)) > 1:
            roc_auc = float(roc_auc_score(test_targets, test_probs))
        else:
            roc_auc = 0.5

        conf_matrix = confusion_matrix(test_targets, test_preds).tolist()

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": conf_matrix,
            "class_weight_pos": round(pos_weight_val, 4),
        }

        return self.model, metrics, history
