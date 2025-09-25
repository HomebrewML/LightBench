from typing import Optional

import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"weight": 0.004},
    "easy": {"weight": 0.02},
    "medium": {"weight": 0.1},
    "hard": {"weight": 0.5},
    "extreme": {"weight": 1},
    "nightmare": {"weight": 2},
}


class Model(nn.Module):
    def __init__(self, weight: float, size=1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("t", torch.zeros(1))
        self.weight = weight

    def forward(self):
        """Tests effective use of momentum for oscillating landscapes."""
        self.t += 0.1
        x = self.param
        return (x.square() + self.weight * torch.sin(10 * x) * torch.cos(self.t)).mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    weight: float = 0.1,
    config: Optional[str] = None,
):
    weight = configs.get(config, {}).get("weight", weight)
    dtype = getattr(torch, dtype)
    model = Model(weight).cuda().double()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-6),
        steps,
        opt,
        weight_decay,
        failure_threshold=3,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
