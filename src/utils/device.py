from __future__ import annotations
import torch
from typing import Tuple, Union

def get_device(preferred: str = "auto") -> torch.device:
    """Resolve a torch.device from a preference string."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def move_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move a (spikes, labels) batch to device."""
    spikes, labels = batch
    return spikes.to(device), labels.to(device)
