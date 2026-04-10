from typing import Optional

import torch
import torch.nn.functional as F
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "size", "lo": 4, "hi": 2**17, "scale": "log", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("scale", F.normalize(torch.arange(1, 1 + size).float(), dim=0, p=1))

    def forward(self):
        return self.param.square().max()


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
    model = Model(kwargs["size"]).cuda()

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
