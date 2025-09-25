from typing import Optional

import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"sparsity": 0.5},
    "easy": {"sparsity": 2**-3},
    "medium": {"sparsity": 2**-6},
    "hard": {"sparsity": 2**-8},
    "extreme": {"sparsity": 2**-11},
    "nightmare": {"sparsity": 2**-14},
}


class Model(nn.Module):
    def __init__(self, size=2**16, sparsity=2**-6):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.sparsity = sparsity
        self.register_buffer("prev_mask", torch.zeros_like(self.param))

    def forward(self):
        """Test optimizer's ability to handle sparse gradients."""
        # Generate new random mask each time, but keep some consistency
        new_mask = (torch.rand_like(self.param) < self.sparsity).float()
        mask = (new_mask + self.prev_mask) > 0  # Union of current and previous mask
        self.prev_mask.copy_(new_mask)

        return (self.param * mask.float()).square().mean()


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    sparsity: float = 2**-6,
    config: Optional[str] = None,
):
    sparsity = configs.get(config, {}).get("sparsity", sparsity)
    model = Model(sparsity=sparsity).cuda().double()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-4),
        steps,
        opt,
        weight_decay,
        failure_threshold=5,
        trials=trials,
        dtype=dtype,
    )  # More failure attempts allowed


if __name__ == "__main__":
    app()
