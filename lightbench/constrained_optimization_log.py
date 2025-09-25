from typing import Optional

import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"penalty": 1e1},
    "easy": {"penalty": 1e2},
    "medium": {"penalty": 1e4},
    "hard": {"size": 1e6},
    "extreme": {"penalty": 1e8},
    "nightmare": {"penalty": 1e10},
}

# Objective: Minimize (x-2)^2 subject to x <= 1
# Implemented using a penalty: (x-2)^2 + penalty * max(0, x - 1)
TARGET_X = 1.0
TOLERANCE = 1e-3


def objective(x, penalty):
    """Objective function with a penalty for violating the constraint x <= 1."""
    return (x - 2.0) ** 2 + penalty * torch.log(TARGET_X - x)


class Model(nn.Module):
    def __init__(self, penalty):
        super().__init__()
        # Using a tensor with requires_grad=True directly as the parameter
        self.param = nn.Parameter(torch.zeros((16,)))
        self.penalty = penalty

    def forward(self):
        return objective(self.param, self.penalty).mean()


def win_condition(model, loss):
    with torch.no_grad():
        success = ((model.param - TARGET_X).abs() < TOLERANCE).all().item()
        return success, {}


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    steps: int = 200,
    # Increased steps slightly
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 50,  # Reduced trials slightly for faster testing
    win_condition_multiplier: float = 1.0,  # Not used directly, but kept for consistency
    config: Optional[str] = None,
):
    penalty = configs.get(config, {}).get("penalty", 1e6)
    model = Model(penalty)

    trial(
        model,
        None,
        None,
        win_condition,
        steps,
        opt,
        weight_decay,
        failure_threshold=3,
        trials=trials,
        group=32,  # Smaller group size might be better for simple problems
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
