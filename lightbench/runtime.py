"""Runtime configuration helpers for LightBench deployments."""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Optional

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - defensive fallback
    raise RuntimeError("LightBench runtime configuration requires PyTorch to be installed.") from exc

_CONFIGURED = False
_DEVICE: Optional[torch.device] = None


def _format_device_identifier(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    if device.index is None:
        return "cuda"
    return f"cuda:{device.index}"


def _resolve_requested_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None:
        env_device = os.environ.get("LIGHTBENCH_DEVICE")
        if env_device:
            device = env_device
        elif torch.cuda.is_available():  # pragma: no branch - environment dependent
            device = "cuda:0"
        else:
            device = "cpu"

    if isinstance(device, torch.device):
        return device

    try:
        return torch.device(device)
    except (TypeError, RuntimeError) as exc:  # pragma: no cover - invalid configuration guard
        raise ValueError(f"Unsupported device specification: {device!r}") from exc


def configure_environment(device: Optional[str | torch.device] = None) -> torch.device:
    """Normalize CUDA/CPU configuration for LightBench runs.

    The function is idempotent and can be invoked multiple times safely. It
    ensures that only the requested accelerator is visible, pins execution to
    GPU 0 by default (suitable for single-H100 deployments), and enables Hopper
    performance knobs when CUDA is available.
    """

    global _CONFIGURED, _DEVICE

    if _CONFIGURED and _DEVICE is not None:
        return _DEVICE

    target = _resolve_requested_device(device)

    os.environ.setdefault("LIGHTBENCH_DEVICE", _format_device_identifier(target))

    if target.type == "cuda":
        # Ensure deterministic GPU ordering and restrict visibility to the
        # requested device unless the user already overrode it.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            index = 0 if target.index is None else target.index
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

    try:
        if target.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(target)

            # Hopper (H100) benefits from TF32, BF16, and flash SDP kernels.
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception as exc:  # pragma: no cover - logging only
        warnings.warn(f"Failed to configure CUDA environment: {exc}")

    _CONFIGURED = True
    _DEVICE = target
    return target


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Return the configured torch.device, triggering configuration if needed."""

    return configure_environment()


def get_device_string() -> str:
    """Return a canonical device string (e.g. 'cuda:0' or 'cpu')."""

    return _format_device_identifier(get_device())


__all__ = [
    "configure_environment",
    "get_device",
    "get_device_string",
]
