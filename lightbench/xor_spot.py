"""
Inspired by https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py
This version is strongly simplified but follows the same basic idea:
1) Generate random sequence
2) Mark two spots
3) Train a model to predict the xor of the two spots
This does NOT elicit memory in the RNN, but it does force it to learn a pointwise forget mechanism.
"""

from typing import List, Optional

import torch
import torch.backends.opt_einsum
import torch.nn as nn
import typer
from heavyball.utils import set_torch

from lightbench.utils import loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()
configs = {
    "trivial": {"length": 4},
    "easy": {"length": 8},
    "medium": {"length": 16},
    "hard": {"length": 32},
    "extreme": {"length": 64},
    "nightmare": {"length": 96},
}


class Model(nn.Module):
    def __init__(self, size, depth):
        super().__init__()
        self.embed = nn.Embedding(4, size)
        self.enc = nn.LSTM(size, size, depth, batch_first=False)
        self.enc.flatten_parameters()
        self.proj = nn.Sequential(
            nn.LayerNorm(size),  #
            nn.Linear(size, 1),
        )

    def forward(self, inp):
        inp = self.embed(inp.squeeze(-1).long())
        inp = inp[0] + inp[1]
        out, _ = torch.compiler.disable()(self.enc)(inp.transpose(0, 1))
        return self.proj(out[-1, :])


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    length: int = 64,
    size: int = 64,
    depth: int = 1,
    batch: int = 256,
    steps: int = 10,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(
        ["ForeachSOAP", "PaLMForeachSOAP", "PrecondScheduleForeachSOAP"], help="Optimizers to use"
    ),
    win_condition_multiplier: float = 1.0,
    trials: int = 10,
    config: Optional[str] = None,
):
    length = configs.get(config, {}).get("length", length)
    dtype = getattr(torch, dtype)

    model = Model(size, depth).cuda()

    def data():
        inp = torch.randn((batch, length, 1), device="cuda", dtype=dtype)
        inp = inp > 0
        zeros = torch.zeros_like(inp)
        zeros[:, torch.randint(0, length, (batch,), device="cuda")] = 1
        zeros[:, torch.randint(0, length, (batch,), device="cuda")] = 1
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
        failure_threshold=10,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
