from typing import Optional

import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import SkipConfig, param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()


def objective(x):
    """Tests optimizer robustness to non-smooth landscapes with discontinuous gradients."""
    return torch.where(x < 0, x**2, 2 * x).mean()  # Discontinuous gradient at x=0


class Model(nn.Module):
    def __init__(self, size=1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))

    def forward(self):
        return objective(self.param)


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    if config is not None and config != "trivial":
        raise SkipConfig("'config' must be 'trivial'.")
    model = Model().cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-4, 0),
        steps,
        opt,
        weight_decay,
        failure_threshold=3,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
