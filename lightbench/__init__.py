"""LightBench benchmark suite."""

from importlib import import_module
from pathlib import Path

from .runtime import configure_environment, get_device

configure_environment()

__all__ = ["load", "available", "configure_environment", "get_device"]


def load(name: str):
    """Import a LightBench module by its benchmark name."""
    return import_module(f".{name}", __name__)


def available():
    """Return all benchmark module names bundled with LightBench."""
    pkg_dir = Path(__file__).resolve().parent
    excluded = {"__init__", "utils", "run_all_benchmarks", "cli"}
    return sorted(p.stem for p in pkg_dir.glob("*.py") if p.stem not in excluded)
