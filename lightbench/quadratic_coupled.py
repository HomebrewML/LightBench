from typing import Optional

import torch
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"groups": 1},
    "easy": {"groups": 2},
    "medium": {"groups": 4},
    "hard": {"groups": 8},
    "extreme": {"groups": 16},
    "nightmare": {"groups": 32},
}


class Model(nn.Module):
    def __init__(self, groups: int, size: int = 128):
        super().__init__()
        self.param = nn.Parameter(torch.randn((size, groups)))

    def forward(self):
        return self.param.sum(-1).square().mean()


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
