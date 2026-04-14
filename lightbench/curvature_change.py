import math
from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "condition", "lo": 10, "hi": 1000000, "scale": "log"}


class Model(nn.Module):
    def __init__(self, condition, size=256, flip_period=50):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("step", torch.zeros(1))
        Q, _ = torch.linalg.qr(torch.randn(size, size))
        half = math.log10(condition) / 2
        D_a = torch.logspace(-half, half, size)
        D_b = D_a.flip(0)
        self.register_buffer("A_a", D_a.sqrt().unsqueeze(1) * Q)
        self.register_buffer("A_b", D_b.sqrt().unsqueeze(1) * Q)
        self.flip_period = flip_period
        self.noise_scale = 0.01 * math.sqrt(condition)

    def forward(self):
        self.step += 1
        phase = ((self.step.long() // self.flip_period) % 2).item()
        A = self.A_a if phase == 0 else self.A_b
        rotated = A @ self.param
        noise = torch.randn_like(rotated) * self.noise_scale
        return (rotated + noise).square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 200,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 100,
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
        loss_win_condition(win_condition_multiplier * 1e-2),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
