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
    "trivial": {"range": 1},
    "easy": {"range": 2},
    "medium": {"range": 3},
    "hard": {"range": 4},
    "extreme": {"range": 5},
    "nightmare": {"range": 6},
}


class Model(nn.Module):
    def __init__(self, size, value_range):
        super().__init__()
        self.scale = nn.Buffer(torch.logspace(-value_range, value_range, size))
        self.param = nn.Parameter(torch.randn(size))

    def forward(self):
        p2 = self.param**2
        loss = (p2 * self.scale).mean()
        return p2.mean().detach() + loss - loss.detach()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    size: int = 512,
    config: Optional[str] = None,
):
    value_range = configs.get(config, {}).get("range", 3)

    model = Model(size, value_range).cuda().double()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-3),
        steps,
        opt,
        weight_decay,
        failure_threshold=3,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
