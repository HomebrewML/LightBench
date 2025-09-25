from typing import Optional

import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"frequency": 1000},
    "easy": {"frequency": 100},
    "medium": {"frequency": 10},
    "hard": {"frequency": 7},
    "extreme": {"frequency": 4},
    "nightmare": {"frequency": 2},
}


class Model(nn.Module):
    def __init__(self, frequency, size=1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer("step", torch.zeros(1))
        self.frequency = 1 / frequency * 1.1  # to avoid repeating

    def forward(self):
        """Test optimizer's robustness to adversarial gradient patterns."""
        self.step += 1
        # Create an oscillating adversarial component
        direction = torch.sin(self.step * torch.pi * self.frequency)
        # Main objective plus adversarial component
        return self.param.square().mean() + direction * self.param.mean()


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
    frequency = configs.get(config, {}).get("frequency", 10)
    model = Model(frequency).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-3, 0),
        steps,
        opt,
        weight_decay,
        failure_threshold=7,
        trials=trials,
        dtype=dtype,
    )  # More attempts for adversarial case


if __name__ == "__main__":
    app()
