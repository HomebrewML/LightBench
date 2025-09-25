from typing import Optional

import torch
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"bins": 2**6},
    "easy": {"size": 2**5},
    "medium": {"size": 2**4},
    "hard": {"size": 2**3},
    "extreme": {"size": 2**2},
    "nightmare": {"size": 2**1},
}


class Model(nn.Module):
    def __init__(self, bins: int, size=128):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.bins = bins / 2  # because +/-

    def forward(self):
        absmax = self.param.abs().max().detach()
        param = self.param / absmax * self.bins
        param = param.round() / self.bins * absmax
        return param.square().mean()


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
