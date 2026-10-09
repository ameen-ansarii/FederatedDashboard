import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List, Tuple
import flwr as fl

from app.core.reproducibility import derive_client_round_seed, get_torch_generator, seed_client_round
from app.ml.models import DropoutMLP


class DropoutFedProxFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient implementation for FedProx local client optimization.
    Implements local client objective: L_k(w) + (mu / 2) ||w - w_global||^2
    with a frozen copy of global parameters preserved per round.
    """

    def __init__(
        self,
        client_id: str,
        input_dim: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        local_epochs: int = 3,
        mu: float = 0.01,
        optimizer_name: str = "adam",
        random_state: int = 42,
    ):
        self.client_id = client_id
        self.input_dim = input_dim
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.mu = float(mu)
        self.optimizer_name = optimizer_name.lower()
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DropoutMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def get_parameters(self, config: Dict[str, Any] = {}) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def compute_proximal_term(self, global_params_frozen: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes FedProx proximal penalty: (mu / 2) * sum(||w - w_global||^2)
        """
        if self.mu <= 0.0:
            return torch.tensor(0.0, device=self.device)

        prox_loss = torch.tensor(0.0, device=self.device)
        for p, p_g in zip(self.model.parameters(), global_params_frozen):
            prox_loss += torch.sum((p - p_g) ** 2)
        return (self.mu / 2.0) * prox_loss

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Executes local model training using FedProx objective:
        total_loss = base_loss + (mu / 2) * ||w - w_global||^2
        """
        # 1. Install global parameters
        self.set_parameters(parameters)

        round_num = int(config.get("round", 1))
        loader_seed = derive_client_round_seed(self.random_state, self.client_id, round_num)
        seed_client_round(self.random_state, self.client_id, round_num)

        # 2. Preserve a frozen copy of the global parameters before local optimization
        global_params_frozen = [p.clone().detach().to(self.device) for p in self.model.parameters()]

        # Compute class weights for local dataset
        n_pos = float(np.sum(self.y_train == 1.0))
        n_neg = float(np.sum(self.y_train == 0.0))
        pos_weight_val = n_neg / max(n_pos, 1.0)
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if self.optimizer_name == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(
            torch.tensor(self.X_train), torch.tensor(self.y_train).unsqueeze(1)
        )
        train_gen = get_torch_generator(loader_seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(self.X_train)),
            shuffle=True,
            generator=train_gen,
        )

        self.model.train()
        train_losses = []
        prox_losses = []

        for _ in range(self.local_epochs):
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()

                logits = self.model(X_b)
                base_loss = criterion(logits, y_b)
                prox_term = self.compute_proximal_term(global_params_frozen)

                total_loss = base_loss + prox_term
                total_loss.backward()
                optimizer.step()

                train_losses.append(total_loss.item())
                prox_losses.append(prox_term.item())

        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_prox = float(np.mean(prox_losses)) if prox_losses else 0.0
        num_examples = len(self.X_train)

        return (
            self.get_parameters(config),
            num_examples,
            {
                "train_loss": round(avg_loss, 4),
                "prox_loss": round(avg_prox, 4),
                "mu": self.mu,
                "client_id": self.client_id,
            },
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        self.set_parameters(parameters)
        self.model.eval()

        criterion = nn.BCEWithLogitsLoss()
        dataset = TensorDataset(
            torch.tensor(self.X_train), torch.tensor(self.y_train).unsqueeze(1)
        )
        loader = DataLoader(
            dataset, batch_size=min(self.batch_size, len(self.X_train)), shuffle=False
        )

        losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == y_b).sum().item()
                total += len(y_b)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        acc = float(correct / max(total, 1))

        return avg_loss, len(self.X_train), {"accuracy": round(acc, 4)}
