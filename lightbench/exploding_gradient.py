"""
Tests optimizer's ability to handle exploding gradients.

This benchmark creates a scenario where gradients can grow exponentially,
testing the optimizer's:
1. Gradient clipping/scaling mechanisms
2. Numerical stability
3. Ability to make progress despite extreme gradient values
"""

from typing import Optional

import torch
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"scale": 1},
    "easy": {"scale": 2},
    "medium": {"scale": 4},
    "hard": {"scale": 8},
    "extreme": {"scale": 12},
    "nightmare": {"scale": 16},
}


class ExplodingGradient(nn.Module):
    def __init__(self, scale, dim):
        super().__init__()
        self.param = nn.Parameter(torch.randn(dim))
        self.scale = scale  # Controls how quickly gradients grow

    def forward(self):
        # Creates exponentially growing gradients
        # Gradient will be scale * exp(|param|) * sign(param)
        return torch.exp(self.scale * torch.abs(self.param.double())).mean()


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    dim: int = 512,
    steps: int = 500,
    weight_decay: float = 0,
    opt: str = typer.Option("adamw", help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 3,
    config: Optional[str] = None,
):
    scale = configs.get(config, {}).get("scale", 2)
    """Run exploding gradient benchmark with specified parameters."""
    dtype = getattr(torch, dtype)

    model = ExplodingGradient(dim, scale)

    trial(
        model,
        None,
        None,
        param_norm_win_condition(0.01 * win_condition_multiplier, 0),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
