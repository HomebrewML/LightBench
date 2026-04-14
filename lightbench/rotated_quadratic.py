from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.utils import difficulty_to_kwargs, param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "condition", "lo": 1, "hi": 6, "scale": "linear", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, condition, size=256):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        Q, _ = torch.linalg.qr(torch.randn(size, size))
        D = torch.logspace(-condition / 2, condition / 2, size)
        self.register_buffer("A", D.sqrt().unsqueeze(1) * Q)

    def forward(self):
        rotated = self.A @ self.param
        return rotated.square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 500,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 10,
    win_condition_multiplier: float = 1.0,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    dtype = resolve_dtype(dtype)
    model = Model(kwargs["condition"]).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-3, 0),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
