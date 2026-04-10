from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn
from torch.nn import functional as F

from lightbench import resolve_dtype
from lightbench.qrnn import QRNNLayer
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "length", "lo": 2, "hi": 14, "scale": "linear", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, size, length):
        super().__init__()
        self.embed0 = nn.Embedding(2, size)
        self.embed1 = nn.Embedding(2, size)
        self.qrnn = QRNNLayer(size, size)
        self.proj = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, 1))
        self.length = length

    def forward(self, inp):
        i0, i1 = inp.chunk(2, 1)
        x = torch.cat([self.embed0(i0), self.embed1(i1)], dim=1)
        return self.proj(self.qrnn(x)[:, self.length :])


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    length: int = 14,
    size: int = 16,
    batch: int = 256,
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    win_condition_multiplier: float = 1,
    trials: int = 10,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    length = kwargs["length"]
    dtype = resolve_dtype(dtype)
    torch.manual_seed(0x1239121)
    model = Model(size, length).cuda()

    def data():
        inp = torch.randn((batch, 2 * length, 1), device="cuda", dtype=dtype)
        inp = inp > 0
        i0, i1 = inp.chunk(2, 1)
        xored = torch.logical_xor(i0, i1)
        return inp.long().squeeze(-1), xored.to(dtype)

    trial(
        model,
        data,
        F.binary_cross_entropy_with_logits,
        loss_win_condition(win_condition_multiplier * 1e-2),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
