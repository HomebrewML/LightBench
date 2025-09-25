from typing import Optional

import torch
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"spike": 4},
    "easy": {"spike": 16},
    "medium": {"spike": 512},
    "hard": {"spike": 8192},
    "extreme": {"spike": 2**15},
    "nightmare": {"spike": 2**17},
}


class Model(nn.Module):
    def __init__(self, spike: float, size=128):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("scale", torch.arange(1, 1 + size).float() / (1 + size))
        self.spike = spike

    def forward(self):
        spikes = torch.rand_like(self.scale) < self.scale
        out = self.param.square()
        return torch.where(spikes, out * self.spike, out).mean()


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    size: int = 1024,
    batch: int = 256,
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 10,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    kwargs = configs[config or "trivial"]
    model = Model(**kwargs).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-7, 0),
        steps,
        opt,
        weight_decay=weight_decay,
        failure_threshold=2,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
