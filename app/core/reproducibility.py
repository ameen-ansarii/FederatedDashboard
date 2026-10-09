"""
Experiment Reproducibility & Seed Isolation Module.

Provides deterministic seed derivation and RNG isolation for benchmark experiments.
Each sub-run (Centralized, FedAvg IID, FedAvg Non-IID, FedProx μ=X) receives
a unique but deterministic child seed derived from:

    base_seed → algorithm identity → variant parameters → client/round context

This ensures:
    1. Same base_seed + same config → identical results (reproducibility)
    2. Different sub-runs → independent RNG streams (isolation)
    3. Execution order does not affect results (order independence)
"""

import hashlib
import random
import struct
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def derive_seed(base_seed: int, *components: str) -> int:
    """
    Derives a deterministic child seed from a base seed and string identity components.

    Uses SHA-256 to hash the base seed together with ordered identity strings,
    then folds the digest into a 32-bit unsigned integer suitable for all RNG APIs.

    Args:
        base_seed: Root experiment seed (e.g. 42).
        *components: Ordered identity strings that uniquely identify the sub-run.
                     Examples: ("CENTRALIZED",), ("FEDAVG", "IID"), ("FEDPROX", "NON_IID", "mu_0.01")

    Returns:
        Deterministic integer seed in [0, 2^31 - 1].

    Properties:
        - derive_seed(42, "A") == derive_seed(42, "A")           (deterministic)
        - derive_seed(42, "A") != derive_seed(42, "B")           (isolated)
        - derive_seed(42, "A") != derive_seed(123, "A")          (seed-dependent)
    """
    hasher = hashlib.sha256()
    hasher.update(struct.pack(">q", base_seed))  # 8-byte big-endian signed long
    for component in components:
        hasher.update(component.encode("utf-8"))
    digest = hasher.digest()
    # Fold first 4 bytes into a non-negative 31-bit integer
    raw = int.from_bytes(digest[:4], byteorder="big")
    return raw % (2**31)


def seed_everything(seed: int) -> None:
    """
    Seeds all global RNG sources for complete reproducibility of a single sub-run.

    Must be called at the START of every independent training sub-run
    (centralized, each fedavg variant, each fedprox μ value) to fully
    reset global RNG state and prevent cross-run leakage.

    Args:
        seed: Derived child seed for this specific sub-run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_generator(seed: int) -> torch.Generator:
    """
    Creates an isolated PyTorch Generator for DataLoader shuffling.

    Using an explicit Generator prevents DataLoader shuffle operations
    from consuming the global PyTorch RNG, keeping model training
    deterministic regardless of DataLoader iteration patterns.

    Args:
        seed: Seed for this generator instance.

    Returns:
        A seeded torch.Generator for use as DataLoader's generator parameter.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_numpy_rng(seed: int) -> np.random.Generator:
    """
    Returns an isolated NumPy Generator for experiment-specific randomness.

    Prefer this over global np.random to avoid cross-run leakage.
    """
    return np.random.default_rng(seed)


def derive_benchmark_run_seed(
    base_seed: int,
    algorithm: str,
    variant: str,
    *,
    dirichlet_alpha: Optional[float] = None,
    mu: Optional[float] = None,
) -> int:
    """
    Derives an isolated seed for a single benchmark sub-run.

    Identity components:
        algorithm  — e.g. CENTRALIZED, FEDAVG, FEDPROX
        variant    — e.g. BASELINE, IID, NON_IID
        dirichlet_alpha — included for Non-IID runs (when provided)
        mu         — included for FedProx runs (when provided)
    """
    components: List[str] = [algorithm.upper(), variant.upper()]
    if dirichlet_alpha is not None:
        components.append(f"alpha_{dirichlet_alpha}")
    if mu is not None:
        components.append(f"mu_{mu}")
    return derive_seed(base_seed, *components)


def derive_client_round_seed(experiment_seed: int, client_id: str, round_num: int) -> int:
    """
    Derives a deterministic seed for a single client's local training in one round.

    Independent of client execution order because identity is keyed by client_id
    and round number, not iteration position.
    """
    return derive_seed(experiment_seed, "CLIENT", client_id, f"round_{round_num}")


def seed_client_round(experiment_seed: int, client_id: str, round_num: int) -> int:
    """
    Resets global RNG state for one client-round local training step.

    Returns the derived seed used (for metadata / debugging).
    """
    child_seed = derive_client_round_seed(experiment_seed, client_id, round_num)
    seed_everything(child_seed)
    return child_seed


def model_state_checksum(model: torch.nn.Module) -> str:
    """
    Computes a deterministic SHA-256 checksum over model parameter tensors.
    Useful for reproducibility assertions without exposing raw weights.
    """
    hasher = hashlib.sha256()
    for param in model.state_dict().values():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def build_reproducibility_metadata(
    base_random_state: int,
    derived_seed: int,
    algorithm: str,
    variant: str,
    *,
    mu: Optional[float] = None,
    num_clients: Optional[int] = None,
    rounds: Optional[int] = None,
    local_epochs: Optional[int] = None,
    dirichlet_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Builds a reproducibility metadata block for experiment/training records."""
    meta: Dict[str, Any] = {
        "base_random_state": base_random_state,
        "derived_seed": derived_seed,
        "algorithm": algorithm,
        "variant": variant,
    }
    if mu is not None:
        meta["mu"] = mu
    if num_clients is not None:
        meta["num_clients"] = num_clients
    if rounds is not None:
        meta["rounds"] = rounds
    if local_epochs is not None:
        meta["local_epochs"] = local_epochs
    if dirichlet_alpha is not None:
        meta["dirichlet_alpha"] = dirichlet_alpha
    return meta


def configure_deterministic_cuda() -> bool:
    """
    Applies best-effort CUDA deterministic settings when GPU is available.

    Returns True if CUDA is available, False for CPU-only execution.
    Note: some CUDA kernels may remain non-deterministic; bit-exact GPU
    reproducibility is not guaranteed across hardware/driver versions.
    """
    if not torch.cuda.is_available():
        return False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True
