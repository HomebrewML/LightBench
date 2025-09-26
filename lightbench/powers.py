import pathlib
from typing import List, Optional

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import Plotter, disabled_win_condition, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"powers": 4},
    "easy": {"powers": 8},
    "medium": {"powers": 16},
    "hard": {"powers": 32},
    "extreme": {"powers": 128},
    "nightmare": {"powers": 512},
}


class Model(nn.Module):
    def __init__(self, size, powers, target):
        super().__init__()
        self.target = target
        self.param = nn.Parameter(torch.rand(powers, size) * 2)
        self.register_buffer("scale", torch.arange(powers).float().add(1))

    def forward(self):
        x = self.param - self.target
        x = x.double() ** self.scale.view(-1, 1)
        return x.square().mean().to(self.param.dtype)


@app.command()
def main(
    dtype: List[str] = typer.Option(["float64"], help="Data type to use"),
    size: int = 64,
    powers: int = 8,
    steps: int = 10,
    target: float = 1.0,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(["ForeachSOAP"], help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    config: Optional[str] = None,
    show_image: bool = False,
    ema_beta: float = 0.9,
):
    dtype = [getattr(torch, d) for d in dtype]

    power_count = configs.get(config, {}).get("powers", powers)

    model = Model(size, power_count, target)
    if show_image:
        for path in pathlib.Path(".").glob("powers.png"):
            path.unlink()

        model.param.data = model.param.data[:2, :1]
        model.scale.data = torch.stack([model.scale.data[0], model.scale.data[-1]])
        model = Plotter(
            model, x_limits=(-2 + target, 2 + target), y_limits=(-2 + target, 2 + target), should_normalize=True
        )
        win_condition = disabled_win_condition
    else:
        win_condition = loss_win_condition(win_condition_multiplier * 1e-8)

    model.double()

    model = trial(
        model,
        None,
        None,
        win_condition,
        steps,
        opt,
        weight_decay,
        failure_threshold=3,
        trials=trials,
        dtype=dtype,
        return_best=show_image,
        ema_beta=ema_beta,
    )

    if not show_image:
        return

    title = ", ".join(opt)
    model.plot(title=title if len(opt) > 1 else None, save_path="powers.png")


if __name__ == "__main__":
    app()
