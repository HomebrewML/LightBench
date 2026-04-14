from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "depth", "lo": 2, "hi": 16, "scale": "log", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, depth, width=32, batch=16):
        super().__init__()
        self.layers = nn.ParameterList(
            [nn.Parameter(torch.eye(width) + torch.randn(width, width) * 0.01) for _ in range(depth)]
        )
        Q, _ = torch.linalg.qr(torch.randn(width, width))
        self.register_buffer("target", Q)
        self.width = width
        self.batch = batch

    def forward(self):
        x = torch.randn(self.batch, self.width, device=self.target.device, dtype=self.target.dtype)
        out = x
        for w in self.layers:
            out = out @ w
        return (out - x @ self.target).square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 10,
    win_condition_multiplier: float = 1.0,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    dtype = resolve_dtype(dtype)
    model = Model(kwargs["depth"]).cuda()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-4),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
