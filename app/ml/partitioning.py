import numpy as np
from typing import Any, Dict, List, Tuple


def partition_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int = 5,
    strategy: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Partitions training features (X_train) and targets (y_train) into num_clients client partitions.

    Supported strategies:
    - "iid": Uniformly shuffles and splits samples evenly across clients.
    - "non_iid": Uses Dirichlet distribution Dir(alpha) per class label to generate label skew across clients.

    Uses a LOCAL RandomState instance to avoid mutating the global np.random RNG.
    """
    rng = np.random.RandomState(seed)
    num_samples = len(y_train)
    client_partitions: List[Dict[str, np.ndarray]] = []

    if num_clients <= 1:
        return [{"X": X_train.copy(), "y": y_train.copy()}]

    strategy = strategy.lower().strip()

    if strategy == "iid":
        indices = np.arange(num_samples)
        rng.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)

        for client_idx in split_indices:
            client_partitions.append({
                "X": X_train[client_idx].copy(),
                "y": y_train[client_idx].copy(),
            })

    elif strategy in ["non_iid", "dirichlet", "non-iid"]:
        unique_classes = np.unique(y_train)
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for c in unique_classes:
            class_indices = np.where(y_train == c)[0]
            rng.shuffle(class_indices)

            # Draw proportions from Dirichlet distribution
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # Normalize to sum to len(class_indices)
            proportions = (proportions * len(class_indices)).astype(int)

            # Fix rounding discrepancy to ensure all samples are assigned
            diff = len(class_indices) - np.sum(proportions)
            for i in range(abs(diff)):
                proportions[i % num_clients] += 1 if diff > 0 else -1

            # Split class indices according to proportions
            current_pos = 0
            for client_id in range(num_clients):
                count = max(0, proportions[client_id])
                assigned = class_indices[current_pos : current_pos + count]
                client_indices[client_id].extend(assigned)
                current_pos += count

            # Assign any residual samples from rounding
            if current_pos < len(class_indices):
                client_indices[0].extend(class_indices[current_pos:])

        for client_id in range(num_clients):
            c_idx = np.array(client_indices[client_id])
            if len(c_idx) == 0:
                # Fallback: assign at least 1 random sample if Dirichlet proportion gave 0
                c_idx = rng.choice(num_samples, size=2, replace=False)
            rng.shuffle(c_idx)

            client_partitions.append({
                "X": X_train[c_idx].copy(),
                "y": y_train[c_idx].copy(),
            })

    else:
        raise ValueError(f"Unsupported partitioning strategy: '{strategy}'. Supported options: 'iid', 'non_iid'")

    return client_partitions

