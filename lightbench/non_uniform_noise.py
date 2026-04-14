from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "noise_range", "lo": 1, "hi": 6, "scale": "linear"}


class Model(nn.Module):
    def __init__(self, noise_range, size=4096):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("noise_scales", torch.logspace(-noise_range / 2, noise_range / 2, size) * 0.01)

    def forward(self):
        noise = torch.randn_like(self.param) * self.noise_scales
        return (self.param + noise).square().mean()


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
    model = Model(kwargs["noise_range"]).cuda()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-2),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
