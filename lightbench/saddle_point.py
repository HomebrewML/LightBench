import pathlib
import random
from typing import Optional

import matplotlib.colors
import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn
from utils import Plotter

from lightbench.utils import loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"power": 1},
    "easy": {"power": 2},
    "medium": {"power": 4},
    "hard": {"power": 8},
    "extreme": {"power": 16},
    "nightmare": {"power": 32},
}


def objective(*xs, power):
    """Classic saddle point objective - tests ability to escape saddle points."""
    return sum(x**power for x in xs)


class Model(nn.Module):
    def __init__(self, power, offset):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.2, 1.9]).float())
        self.offset = offset
        self.power = 2 * power + 1

    def forward(self):
        return objective(*self.param, power=self.power) + self.offset


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    show_image: bool = False,
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    dtype = getattr(torch, dtype)
    coords = configs.get(config, {}).get("power", 1)

    # Clean up old plots
    for path in pathlib.Path(".").glob("saddle_point.png"):
        path.unlink()

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    rng = random.Random(0x1239121)
    rng.shuffle(colors)

    offset = win_condition_multiplier * 10

    if show_image:
        model = Plotter(
            lambda *x: objective(*x).add(offset).log(),
            coords=coords,
            xlim=(-2, 2),
            ylim=(-2, 2),
            normalize=8,
            after_step=torch.exp,
        )
    else:
        model = Model(coords, offset)
    model.double()

    trial(
        model,
        None,
        None,
        loss_win_condition(0.1),
        steps,
        opt,
        weight_decay,
        failure_threshold=3,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
