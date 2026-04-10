from collections import deque
from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "max_delay", "lo": 2, "hi": 256, "scale": "log", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, max_delay=16, param_size=256):
        super().__init__()
        self.param = nn.Parameter(torch.randn(param_size))
        self.history = deque(maxlen=max_delay)
        self.target = nn.Buffer(torch.randn_like(self.param))

    def forward(self):
        self.history.append(self.param.detach().clone())
        if self.history:
            hist_loss = sum(torch.norm(p - self.target) for p in self.history)
        else:
            hist_loss = 0
        return hist_loss + torch.norm(self.param - self.target)


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    model = Model(kwargs["max_delay"]).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-4, -model.target),
        steps * 2,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )  # Double steps, more attempts


if __name__ == "__main__":
    app()
