from collections import deque
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
    "trivial": {"max_delay": 2},
    "easy": {"max_delay": 4},
    "medium": {"max_delay": 16},
    "hard": {"max_delay": 64},
    "extreme": {"max_delay": 128},
    "nightmare": {"max_delay": 256},
}


class Model(nn.Module):
    def __init__(self, max_delay=16, param_size=256):
        super().__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(param_size)) for _ in range(max_delay)])
        # Different update frequencies for each parameter
        self.delays = [i for i in range(max_delay)]
        self.step = 0
        self.grad_queues = [deque(maxlen=i + 1) for i in self.delays]

    def forward(self):
        """Test optimizer's ability to handle delayed gradients."""
        total_loss = 0
        self.step += 1

        for param, delay, queue in zip(self.params, self.delays, self.grad_queues):
            # Current loss for this parameter
            loss = param.square().mean()

            # Store the gradient in the queue
            queue.append(loss)

            # Only add to total loss when we have enough history
            if len(queue) == queue.maxlen and self.step % (delay + 1) == 0:
                total_loss = total_loss + queue[0]  # Use oldest gradient

        return total_loss / len(self.params)


@app.command()
def main(
    dtype: str = typer.Option("float64", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    max_delay = configs.get(config, {}).get("max_delay", 4)
    dtype = getattr(torch, dtype)
    model = Model(max_delay).cuda().double()

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-4),
        steps * 2,
        opt,
        weight_decay,
        failure_threshold=5,
        trials=trials,
        dtype=dtype,
    )  # Double steps, more attempts


if __name__ == "__main__":
    app()
