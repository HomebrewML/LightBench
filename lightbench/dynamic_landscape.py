"""
Tests optimizer's ability to track moving targets.

This benchmark simulates a dynamic loss landscape where the optimal parameters
continuously shift over time. This tests the optimizer's ability to:
1. Track moving targets
2. Adapt to non-stationary objectives
3. Handle continuous parameter updates
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"frequency": 1000},
    "easy": {"frequency": 100},
    "medium": {"frequency": 20},
    "hard": {"frequency": 10},
    "extreme": {"frequency": 6},
    "nightmare": {"frequency": 4},
}


class ShiftingSphere(nn.Module):
    def __init__(self, dim, frequency):
        super().__init__()
        self.param = nn.Parameter(torch.randn(dim))
        self.phase = 0
        self.frequency = 1 / frequency * 1.1  # so that we don't repeat numbers

    def forward(self):
        self.phase += self.frequency
        target = torch.linspace(0, 2 * math.pi, len(self.param), device=self.param.device, dtype=self.param.dtype)
        target = torch.sin(target + self.phase)
        return (self.param - target).square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    dim: int = 16384,
    steps: int = 500,
    weight_decay: float = 0,
    opt: str = typer.Option("adamw", help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 3,
    config: Optional[str] = None,
):
    """Run dynamic landscape benchmark with specified parameters."""
    frequency = configs.get(config, {}).get("frequency", 0.1)

    model = ShiftingSphere(dim, frequency)

    trial(
        model,
        None,
        None,
        loss_win_condition(0.01 * win_condition_multiplier),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
