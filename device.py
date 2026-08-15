"""Global cuda/cpu switch. Set via --device or DEVICE=cuda|cpu."""

import os
import torch


def resolve(name=None):
    name = name or os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if name not in ("cuda", "cpu"):
        raise ValueError(f"device must be cuda or cpu, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    os.environ["DEVICE"] = name
    return torch.device(name)
