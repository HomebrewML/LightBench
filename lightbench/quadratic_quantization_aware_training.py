from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "bins", "lo": 64, "hi": 2, "scale": "log", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, bins: int, size=128):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.bins = bins / 2  # because +/-

    def forward(self):
        absmax = self.param.abs().max().detach()
        param = self.param / absmax * self.bins
        param = param + (param.round() - param).detach()
        param = param / self.bins * absmax
        return param.square().mean()


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
    model = Model(kwargs["bins"]).cuda()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-6),
        steps,
        opt,
        weight_decay=weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
