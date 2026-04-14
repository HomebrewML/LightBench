from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "order", "lo": 2, "hi": 16, "scale": "linear", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, order, size=256):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size) * 0.5)
        self.register_buffer("targets", torch.randn(size - order + 1) * 0.1)
        self.order = order

    def forward(self):
        windows = self.param.unfold(0, self.order, 1)
        return (windows.prod(-1) - self.targets).square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    dtype = resolve_dtype(dtype)
    model = Model(kwargs["order"]).cuda()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-3),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
