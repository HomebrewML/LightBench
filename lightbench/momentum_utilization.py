from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
difficulty_spec = {"param": "weight", "lo": 0.1, "hi": 1000, "scale": "log"}


class Model(nn.Module):
    def __init__(self, weight: float, size=1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("t", torch.zeros(1))
        self.weight = weight

    def forward(self):
        self.t += 1
        base = self.param.square().mean()
        adversarial = self.weight * torch.sin(self.t * 0.1) * base
        return base + adversarial - adversarial.detach()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    weight: float = 0.1,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    weight = kwargs["weight"]
    dtype = resolve_dtype(dtype)
    model = Model(weight).cuda().double()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-6),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
