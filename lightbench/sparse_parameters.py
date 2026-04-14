from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
difficulty_spec = {"param": "sparsity", "lo": 0.5, "hi": 2**-14, "scale": "log"}


class Model(nn.Module):
    def __init__(self, size=2**16, sparsity=2**-6):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.sparsity = sparsity
        self.noise_scale = 1.0 / (sparsity * size) ** 0.5
        mask = torch.zeros_like(self.param)
        while mask.sum().item() < 1:
            mask = torch.rand_like(self.param) < self.sparsity
        self.param.data.mul_(mask)
        if size * sparsity < 1:
            print(f"enforcing sparsity = {1 / size}")

    def forward(self):
        noise = torch.randn_like(self.param) * self.noise_scale
        return (self.param + noise).square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    sparsity: float = 2**-6,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    sparsity = kwargs["sparsity"]
    model = Model(sparsity=sparsity).cuda().double()

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
