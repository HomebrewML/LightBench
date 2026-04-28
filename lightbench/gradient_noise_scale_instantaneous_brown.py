from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "offset", "lo": 32, "hi": 1, "scale": "log", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, offset, size=4096):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.offset = offset

    def forward(self):
        noise_scale = 1.0 / self.offset
        noise = torch.randn_like(self.param).cumsum(0) / self.param.numel() ** 0.5 * noise_scale
        return (self.param + noise).square().mean()


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
    offset = kwargs["offset"]
    model = Model(offset).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-2, 0),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
