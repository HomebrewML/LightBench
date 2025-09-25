"""LightBench benchmark suite."""

from importlib import import_module
from pathlib import Path

__all__ = ["load", "available"]


def load(name: str):
    """Import a LightBench module by its benchmark name."""
    return import_module(f".{name}", __name__)


def available():
    """Return all benchmark module names bundled with LightBench."""
    pkg_dir = Path(__file__).resolve().parent
    return sorted(p.stem for p in pkg_dir.glob("*.py") if p.stem not in {"__init__", "utils", "run_all_benchmarks"})
