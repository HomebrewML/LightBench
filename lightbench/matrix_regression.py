from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "condition", "lo": 1, "hi": 4, "scale": "linear", "dtype": "int"}


def _make_ill_conditioned(n, condition):
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    S = torch.logspace(-condition / 2, condition / 2, n)
    return Q * S.unsqueeze(0)


class Model(nn.Module):
    def __init__(self, condition, size=64):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size, size))
        A = _make_ill_conditioned(size, condition)
        B = _make_ill_conditioned(size, condition)
        target = torch.randn(size, size) / size**0.5
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", A @ target @ B)

    def forward(self):
        return (self.A @ self.param @ self.B - self.C).square().mean()


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
    model = Model(kwargs["condition"]).cuda()

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
