from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench import resolve_dtype
from lightbench.qrnn import QRNNLayer
from lightbench.utils import difficulty_to_kwargs, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "length", "lo": 4, "hi": 96, "scale": "linear", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embed = nn.Embedding(4, size)
        self.qrnn = QRNNLayer(size, size)
        self.proj = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, 1))

    def forward(self, inp):
        x = self.embed(inp.squeeze(-1).long())
        x = x[0] + x[1]
        return self.proj(self.qrnn(x)[:, -1])


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    length: int = 64,
    size: int = 64,
    batch: int = 256,
    steps: int = 10,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    length = kwargs["length"]
    dtype = resolve_dtype(dtype)
    model = Model(size).cuda()

    def data():
        inp = torch.randn((batch, length, 1), device="cuda", dtype=dtype)
        inp = inp > 0
        zeros = torch.zeros_like(inp)
        rows = torch.arange(batch, device=zeros.device)
        first = torch.randint(0, length, (batch,), device=zeros.device)
        second = (first + torch.randint(1, length, (batch,), device=zeros.device)) % length
        zeros[rows, first, 0] = 1
        zeros[rows, second, 0] = 1
        target = (inp * zeros).sum(1) % 2
        return torch.stack((inp, zeros + 2), 0).to(dtype), target.to(dtype)

    trial(
        model,
        data,
        torch.nn.functional.binary_cross_entropy_with_logits,
        loss_win_condition(win_condition_multiplier * 1e-2),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
