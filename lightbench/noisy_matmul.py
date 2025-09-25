from typing import Optional

import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn
from torch.nn import functional as F

from lightbench.utils import param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"depth": 1},
    "easy": {"depth": 2},
    "medium": {"depth": 8},
    "hard": {"depth": 12},
    "extreme": {"depth": 16},
    "nightmare": {"depth": 20},
}


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.randn((size,)))
        self.offset = nn.Buffer(torch.randn_like(self.param))

    def forward(self, inp):
        y = None
        y0 = self.param.view(1, -1).expand(inp.size(0), -1) + self.offset  # offset, so weight decay doesnt help
        for i in inp.unbind(1):
            y = torch.einsum("bi,bik->bk", y0, i)
            y0 = F.leaky_relu(y, 0.1)
        return y


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    size: int = 64,
    depth: int = 4,
    batch: int = 128,
    steps: int = 10,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    config: Optional[str] = None,
):
    depth = configs.get(config, {}).get("depth", depth)
    dtype = getattr(torch, dtype)
    model = Model(size).cuda()

    def data():
        inp = torch.randn((batch, depth, size, size), device="cuda", dtype=dtype) / size**0.5
        return inp, torch.zeros((batch, size), device="cuda", dtype=dtype)

    trial(
        model,
        data,
        F.mse_loss,
        param_norm_win_condition(1e-7 * win_condition_multiplier, model.offset),
        steps,
        opt,
        weight_decay,
        failure_threshold=depth * 2,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
