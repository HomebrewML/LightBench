"""LightBench benchmark suite."""

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

import torch

__all__ = ["available", "load", "resolve_dtype"]


def load(name: str):
    """Import a LightBench module by its benchmark name."""
    return import_module(f".{name}", __name__)


_EXCLUDED = {
    "__init__",
    "__main__",
    "utils",
    "run_all_benchmarks",
    "cli",
    "format_results",
    "merge_logs",
    "generate_animations",
    "create_hiker_analogy_diagram",
    "loss_contour",
    "relu_boundaries",
}


def available():
    """Return all benchmark module names bundled with LightBench."""
    pkg_dir = Path(__file__).resolve().parent
    return sorted(p.stem for p in pkg_dir.glob("*.py") if p.stem not in _EXCLUDED)


def resolve_dtype(value):
    """Normalize a dtype specification to a ``torch.dtype`` or ``None``.

    Accepts string names (``"float32"``), ``torch.dtype`` objects, or
    single-element sequences of either.  ``None`` and empty sequences
    return ``None``.
    """
    if value is None:
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = list(value)
        if not value:
            return None
        if len(value) > 1:
            raise ValueError(f"Expected a single dtype, received {len(value)} values.")
        value = value[0]

    if isinstance(value, torch.dtype):
        return value

    if isinstance(value, str):
        attr = getattr(torch, value, None)
        if isinstance(attr, torch.dtype):
            return attr
        raise ValueError(f"Unknown torch dtype '{value}'")

    raise TypeError(f"Unsupported dtype specification: {type(value)!r}")
