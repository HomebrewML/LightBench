from typing import Optional

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"powers": 4},
    "easy": {"powers": 8},
    "medium": {"powers": 16},
    "hard": {"powers": 32},
    "extreme": {"powers": 128},
    "nightmare": {"powers": 512},
}


class Model(nn.Module):
    def __init__(self, size, powers, target):
        super().__init__()
        self.target = target
        self.param = nn.Parameter(torch.rand(powers, size) * 2)
        self.register_buffer("scale", torch.arange(powers).float().add(1))

    def forward(self):
        x = self.param - self.target
        x = x ** self.scale.view(-1, 1)
        return x.square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    size: int = 64,
    powers: int = 8,
    steps: int = 10,
    target: float = 1.0,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    config: Optional[str] = None,
):
    powers = configs.get(config, {}).get("powers", powers)
    model = Model(size, powers, target).cuda().double()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-8),
        steps,
        opt,
        weight_decay,
        failure_threshold=3,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
