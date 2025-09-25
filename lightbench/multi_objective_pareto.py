"""
Multi-Objective Pareto Optimization Benchmark

Tests an optimizer's ability to navigate Pareto-optimal trade-offs between
conflicting objectives: model accuracy and computational efficiency. This
benchmark simulates real-world scenarios where practitioners must balance
performance with resource constraints.

The task optimizes a neural network that must simultaneously:
1. Minimize prediction error (accuracy objective)
2. Minimize computational cost (efficiency objective)

Success is measured by the optimizer's ability to find solutions on or near
the Pareto frontier, representing optimal trade-offs between these objectives.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from heavyball.utils import set_torch

from lightbench.utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {
        "input_dim": 10,
        "hidden_dim": 8,
        "output_dim": 2,
        "n_samples": 100,
        "efficiency_weight": 0.1,
        "target_efficiency": 0.5,
    },
    "easy": {
        "input_dim": 32,
        "hidden_dim": 16,
        "output_dim": 4,
        "n_samples": 500,
        "efficiency_weight": 0.3,
        "target_efficiency": 0.3,
    },
    "medium": {
        "input_dim": 64,
        "hidden_dim": 32,
        "output_dim": 8,
        "n_samples": 1000,
        "efficiency_weight": 0.5,
        "target_efficiency": 0.2,
    },
    "hard": {
        "input_dim": 128,
        "hidden_dim": 64,
        "output_dim": 16,
        "n_samples": 2000,
        "efficiency_weight": 0.7,
        "target_efficiency": 0.15,
    },
    "extreme": {
        "input_dim": 256,
        "hidden_dim": 128,
        "output_dim": 32,
        "n_samples": 5000,
        "efficiency_weight": 0.8,
        "target_efficiency": 0.1,
    },
    "nightmare": {
        "input_dim": 512,
        "hidden_dim": 256,
        "output_dim": 64,
        "n_samples": 10000,
        "efficiency_weight": 0.9,
        "target_efficiency": 0.05,
    },
}


class MultiObjectiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.efficiency_param = nn.Parameter(torch.ones(1))
        self.register_buffer("X", torch.randn(n_samples, input_dim))
        self.register_buffer("y", torch.randint(0, output_dim, (n_samples,)))

    def forward(self):
        logits = self.layer2(F.relu(self.layer1(self.X)))
        accuracy_loss = F.cross_entropy(logits, self.y)
        weight_penalty = sum(p.square().sum() for p in self.parameters() if p is not self.efficiency_param)
        efficiency_loss = weight_penalty / torch.sigmoid(self.efficiency_param)
        return accuracy_loss, efficiency_loss


def pareto_win_condition(accuracy_threshold, efficiency_threshold, weight):
    def win(model, loss):
        with torch.no_grad():
            accuracy_loss, efficiency_loss = model()
            combined_loss = weight * accuracy_loss + (1 - weight) * efficiency_loss
            accuracy_ok = accuracy_loss <= accuracy_threshold
            efficiency_ok = efficiency_loss <= efficiency_threshold
            return (accuracy_ok and efficiency_ok).item(), {
                "accuracy_loss": accuracy_loss.item(),
                "efficiency_loss": efficiency_loss.item(),
                "combined_loss": combined_loss.item(),
                "is_pareto_feasible": (accuracy_ok and efficiency_ok).item(),
            }

    return win


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    input_dim: int = 64,
    hidden_dim: int = 32,
    output_dim: int = 8,
    n_samples: int = 1000,
    efficiency_weight: float = 0.5,
    target_efficiency: float = 0.2,
    steps: int = 1000,
    weight_decay: float = 0.01,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 20,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    """
    Multi-objective optimization benchmark balancing accuracy and efficiency.

    This benchmark tests the optimizer's ability to find Pareto-optimal solutions
    that balance model accuracy with computational efficiency constraints.
    """
    if config:
        cfg = configs.get(config, {})
        input_dim = cfg.get("input_dim", input_dim)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        output_dim = cfg.get("output_dim", output_dim)
        n_samples = cfg.get("n_samples", n_samples)
        efficiency_weight = cfg.get("efficiency_weight", efficiency_weight)
        target_efficiency = cfg.get("target_efficiency", target_efficiency)

    dtype = getattr(torch, dtype)
    model = MultiObjectiveModel(input_dim, hidden_dim, output_dim, n_samples).cuda()

    def loss_fn(outputs, target):
        accuracy_loss, efficiency_loss = outputs
        return efficiency_weight * accuracy_loss + (1 - efficiency_weight) * efficiency_loss

    # For trivial config, relax win condition thresholds to ensure achievability
    if config == "trivial":
        accuracy_thr = max(0.05, win_condition_multiplier * 0.2)
        efficiency_thr = max(0.1, win_condition_multiplier * target_efficiency * 2)
    else:
        accuracy_thr = win_condition_multiplier * 0.1
        efficiency_thr = win_condition_multiplier * target_efficiency

    trial(
        model,
        None,
        loss_fn,
        pareto_win_condition(
            accuracy_threshold=accuracy_thr,
            efficiency_threshold=efficiency_thr,
            weight=efficiency_weight,
        ),
        steps,
        opt,
        weight_decay,
        trials=trials,
        failure_threshold=3,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
