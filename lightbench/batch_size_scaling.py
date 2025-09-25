import math
import random
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
    "trivial": {"max_batch": 65536},
    "easy": {"max_batch": 8192},
    "medium": {"max_batch": 1024},
    "hard": {"max_batch": 128},
    "extreme": {"max_batch": 16},
    "nightmare": {"max_batch": 2},
}


class Model(nn.Module):
    def __init__(self, max_batch: int, size=1024):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.max_batch = max_batch
        self.rng = random.Random(0x1238192)

    def forward(self):
        """Test optimizer's ability to handle different batch sizes and noise scales."""
        generator = torch.Generator(device=self.param.device).manual_seed(self.rng.randint(0, 2**31))
        noise = torch.randn(self.param.shape, generator=generator, device=self.param.device)
        scale = self.param.norm() / (noise.norm() + 1e-6)
        batch_scale = self.max_batch ** (self.rng.random() / 2)  # sqrt of random uniform between 1 and max_batch
        noise *= scale.detach() / math.sqrt(batch_scale)
        return (self.param + noise).square().mean()


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
    max_batch = configs.get(config, {}).get("max_batch", 256)
    model = Model(max_batch).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-8, 0),
        steps,
        opt,
        weight_decay,
        failure_threshold=5,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
