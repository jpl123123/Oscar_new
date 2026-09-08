"""Strict startup-only loading of OSCAR's calibrated rotation artifacts."""

import re
from functools import lru_cache


def layer_index(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if match is None:
        raise ValueError(f"Cannot resolve global decoder layer index: {name}")
    return int(match.group(1))


@lru_cache(maxsize=4)
def load_checkpoint(path):
    import torch

    if not path:
        raise ValueError("Qwen3.5 D256 calibrated K/V rotation paths are required")
    source = torch.load(path, map_location="cpu", weights_only=True)
    if torch.is_tensor(source) and source.ndim == 3:
        items = enumerate(source)
    elif isinstance(source, dict):
        items = source.get("layers", source).items()
    else:
        raise ValueError(f"Unsupported rotation checkpoint format: {path}")
    result = {}
    for key, entry in items:
        matrix = entry["rotation"] if isinstance(entry, dict) else entry
        if not torch.is_tensor(matrix):
            raise ValueError(f"Invalid matrix for layer {key} in {path}")
        result[int(key)] = matrix.float().contiguous()
    return result


def get_rotation(path, name, device, dim=256):
    import torch

    index = layer_index(name)
    table = load_checkpoint(path)
    if index not in table:
        raise ValueError(f"Missing rotation for {name} (layer {index}) in {path}")
    matrix = table[index]
    if matrix.shape != (dim, dim):
        raise ValueError(f"Expected D{dim} rotation for {name}; got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"Nonfinite rotation for {name}")
    if not torch.allclose(matrix.T @ matrix, torch.eye(dim), atol=0.015, rtol=0.015):
        raise ValueError(f"Rotation for {name} is not orthogonal")
    return matrix.to(device=device, dtype=torch.bfloat16).contiguous()
