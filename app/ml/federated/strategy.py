import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def aggregate_fedavg(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Performs sample-weighted FedAvg parameter aggregation:
    theta_global = sum_{k=1}^K (N_k / N_total) * theta_k

    Args:
        results: List of tuples (client_weights, num_examples) from each participating client.

    Returns:
        Aggregated global model weights as a list of NumPy ndarrays.
    """
    if not results:
        raise ValueError("Cannot aggregate empty client results list")

    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples == 0:
        total_examples = 1

    # Initialize zero arrays with shape of first client's weights
    num_layers = len(results[0][0])
    aggregated_weights = [np.zeros_like(w) for w in results[0][0]]

    for client_weights, num_examples in results:
        weight_factor = num_examples / total_examples
        for i in range(num_layers):
            aggregated_weights[i] += client_weights[i] * weight_factor

    return aggregated_weights
