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
    "trivial": {"offset": 32},
    "easy": {"offset": 16},
    "medium": {"offset": 8},
    "hard": {"offset": 4},
    "extreme": {"offset": 2},
    "nightmare": {"offset": 1},
}


class Model(nn.Module):
    def __init__(self, offset, size=4096):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.offset = offset

    def forward(self):
        noise_scale = 1.0 / self.offset
        noise = torch.randn_like(self.param) * noise_scale
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
    offset = configs.get(config, {}).get("offset", 4)
    model = Model(offset).cuda()

    trial(
        model,
        None,
        None,
        param_norm_win_condition(win_condition_multiplier * 1e-3, 0),
        steps,
        opt,
        weight_decay,
        failure_threshold=5,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
