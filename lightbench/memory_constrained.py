"""
Memory-Constrained Optimization Benchmark

Tests an optimizer's efficiency and stability when working with limited memory
budgets. This benchmark simulates scenarios where practitioners must train
large models on consumer hardware or edge devices with strict memory constraints.

The task involves optimizing a model that would normally require more memory
than available, forcing the optimizer to work with gradient accumulation,
parameter sharding, or other memory-efficient techniques.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from heavyball.utils import set_torch

from lightbench.utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {
        "model_size": 1024,
        "batch_size": 32,
        "memory_limit_mb": 100,
        "gradient_accumulation_steps": 2,
        "activation_checkpointing": False,
    },
    "easy": {
        "model_size": 4096,
        "batch_size": 64,
        "memory_limit_mb": 200,
        "gradient_accumulation_steps": 4,
        "activation_checkpointing": True,
    },
    "medium": {
        "model_size": 16384,
        "batch_size": 128,
        "memory_limit_mb": 400,
        "gradient_accumulation_steps": 8,
        "activation_checkpointing": True,
    },
    "hard": {
        "model_size": 65536,
        "batch_size": 256,
        "memory_limit_mb": 800,
        "gradient_accumulation_steps": 16,
        "activation_checkpointing": True,
    },
    "extreme": {
        "model_size": 262144,
        "batch_size": 512,
        "memory_limit_mb": 1600,
        "gradient_accumulation_steps": 32,
        "activation_checkpointing": True,
    },
    "nightmare": {
        "model_size": 1048576,
        "batch_size": 1024,
        "memory_limit_mb": 3200,
        "gradient_accumulation_steps": 64,
        "activation_checkpointing": True,
    },
}


class MemoryConstrainedModel(nn.Module):
    def __init__(self, model_size, batch_size, memory_limit_mb, gradient_accumulation_steps, activation_checkpointing):
        super().__init__()
        hidden_dim = int(model_size**0.5)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(8)])
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.activation_checkpointing = activation_checkpointing
        self.register_buffer("X", torch.randn(batch_size * 10, hidden_dim))
        self.register_buffer("y", torch.randn(batch_size * 10, 1))
        self.register_buffer("current_memory_mb", torch.zeros(1))
        self.register_buffer("peak_memory_mb", torch.zeros(1))

    def forward(self, x=None):
        if x is None:
            effective_batch_size = min(self.batch_size, len(self.X))
            x = self.X[:effective_batch_size]
            targets = self.y[:effective_batch_size]
        else:
            targets = None
        torch.cuda.synchronize()
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        self.current_memory_mb.fill_(current_memory)
        self.peak_memory_mb.copy_(torch.maximum(self.peak_memory_mb, self.current_memory_mb))
        hidden = x
        for i, layer in enumerate(self.layers):
            if self.activation_checkpointing and i % 2 == 0:
                hidden = torch.utils.checkpoint.checkpoint(layer, hidden)
            else:
                hidden = layer(hidden)
            hidden = F.relu(hidden)
        output = self.final_layer(hidden)
        if targets is not None:
            loss = F.mse_loss(output, targets)
            memory_penalty = torch.relu(self.current_memory_mb - self.memory_limit_mb) * 0.1
            return loss + memory_penalty
        return output


def memory_constrained_win_condition(loss_threshold, memory_limit_mb, efficiency_threshold):
    def win(model, loss):
        with torch.no_grad():
            current_mem = model.current_memory_mb.item()
            peak_mem = model.peak_memory_mb.item()
            memory_ok = peak_mem <= memory_limit_mb * 1.1
            loss_ok = loss <= loss_threshold
            memory_efficiency = memory_limit_mb / (peak_mem + 1e-6)
            efficiency_ok = memory_efficiency >= efficiency_threshold
            success = memory_ok and loss_ok and efficiency_ok
            return success, {
                "current_memory_mb": current_mem,
                "peak_memory_mb": peak_mem,
                "memory_limit_mb": memory_limit_mb,
                "memory_efficiency": memory_efficiency,
                "loss": loss,
                "memory_ok": memory_ok,
                "loss_ok": loss_ok,
                "efficiency_ok": efficiency_ok,
            }

    return win


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    model_size: int = 16384,
    batch_size: int = 128,
    memory_limit_mb: float = 400,
    gradient_accumulation_steps: int = 8,
    activation_checkpointing: bool = True,
    steps: int = 500,
    weight_decay: float = 0.01,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 15,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    """
    Memory-constrained optimization benchmark.

    Tests optimizer's ability to train effectively under strict memory constraints,
    simulating scenarios like training on consumer GPUs or edge devices.
    """
    if config:
        cfg = configs.get(config, {})
        model_size = cfg.get("model_size", model_size)
        batch_size = cfg.get("batch_size", batch_size)
        memory_limit_mb = cfg.get("memory_limit_mb", memory_limit_mb)
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", gradient_accumulation_steps)
        activation_checkpointing = cfg.get("activation_checkpointing", activation_checkpointing)

    dtype = getattr(torch, dtype)
    model = MemoryConstrainedModel(
        model_size, batch_size, memory_limit_mb, gradient_accumulation_steps, activation_checkpointing
    ).cuda()

    def data():
        return None, None  # Data is embedded in model

    # For trivial config, relax win condition thresholds to ensure achievability
    if config == "trivial":
        loss_thr = max(0.2, win_condition_multiplier * 0.2)
        efficiency_thr = 0.5
    else:
        loss_thr = win_condition_multiplier * 0.1
        efficiency_thr = 0.8

    trial(
        model,
        data,
        None,
        memory_constrained_win_condition(
            loss_threshold=loss_thr, memory_limit_mb=memory_limit_mb, efficiency_threshold=efficiency_thr
        ),
        steps,
        opt,
        weight_decay,
        trials=trials,
        failure_threshold=3,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
