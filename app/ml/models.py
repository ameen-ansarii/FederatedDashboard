import torch
import torch.nn as nn
from typing import List


class DropoutMLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron (MLP) for Student Dropout Prediction.
    Outputs a single logit (unscaled score) per sample for binary classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        activation: str = "relu",
    ):
        super(DropoutMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "gelu":
                layers.append(nn.GELU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = h_dim

        # Final output layer -> 1 logit output for binary classification
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unscaled logits of shape (batch_size, 1).
        """
        return self.model(x)
