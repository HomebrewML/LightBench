from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "groups", "lo": 1, "hi": 32, "scale": "log", "dtype": "int"}


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
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 10,
    win_condition_multiplier: float = 1.0,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    model = Model(kwargs["groups"]).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-7, 0),
        steps,
        opt,
        weight_decay=weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
