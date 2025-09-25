"""
Class Imbalance Rare Event Detection Benchmark

Tests an optimizer's ability to learn from severely imbalanced datasets where
rare positive events are critical to detect. This benchmark simulates real-world
scenarios like fraud detection, medical diagnosis, or anomaly detection where
the minority class is both rare and important.

The task uses a synthetic classification problem with configurable class
imbalance ratios. Success is measured by the optimizer's ability to achieve
good performance on the minority class despite the severe imbalance.
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
    "trivial": {"n_samples": 1000, "input_dim": 20, "hidden_dim": 16, "imbalance_ratio": 0.2, "noise_level": 0.1},
    "easy": {"n_samples": 2000, "input_dim": 32, "hidden_dim": 24, "imbalance_ratio": 0.1, "noise_level": 0.2},
    "medium": {"n_samples": 5000, "input_dim": 48, "hidden_dim": 32, "imbalance_ratio": 0.05, "noise_level": 0.3},
    "hard": {"n_samples": 10000, "input_dim": 64, "hidden_dim": 48, "imbalance_ratio": 0.02, "noise_level": 0.4},
    "extreme": {"n_samples": 20000, "input_dim": 96, "hidden_dim": 64, "imbalance_ratio": 0.01, "noise_level": 0.5},
    "nightmare": {"n_samples": 50000, "input_dim": 128, "hidden_dim": 96, "imbalance_ratio": 0.005, "noise_level": 0.6},
}


class ImbalancedClassifier(nn.Module):
    def __init__(self, n_samples, input_dim, hidden_dim, imbalance_ratio, noise_level):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        n_positive = int(n_samples * imbalance_ratio)
        n_negative = n_samples - n_positive
        positive_X = torch.ones(n_positive, input_dim) + noise_level * torch.randn(n_positive, input_dim)
        positive_y = torch.ones(n_positive, dtype=torch.long)
        negative_X = -torch.ones(n_negative, input_dim) + noise_level * torch.randn(n_negative, input_dim)
        negative_y = torch.zeros(n_negative, dtype=torch.long)
        X = torch.cat([positive_X, negative_X], dim=0)
        y = torch.cat([positive_y, negative_y], dim=0)
        perm = torch.randperm(n_samples)
        self.register_buffer("X", X[perm])
        self.register_buffer("y", y[perm])
        self.register_buffer("pos_weight", torch.tensor(n_negative / n_positive))

    def forward(self):
        logits = self.classifier(self.X)
        alpha = 0.25
        gamma = 2.0
        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, self.y, reduction="none")
        p_t = probs.gather(1, self.y.unsqueeze(1)).squeeze(1)
        focal_weight = alpha * (1 - p_t) ** gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


def imbalanced_win_condition(f1_threshold, recall_threshold):
    def win(model, loss):
        with torch.no_grad():
            logits = model.classifier(model.X)
            predictions = torch.argmax(logits, dim=1)
            true_positives = ((predictions == 1) & (model.y == 1)).sum().float()
            false_positives = ((predictions == 1) & (model.y == 0)).sum().float()
            false_negatives = ((predictions == 0) & (model.y == 1)).sum().float()
            precision = true_positives / (true_positives + false_positives + 1e-7)
            recall = true_positives / (true_positives + false_negatives + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            success = (f1 >= f1_threshold) and (recall >= recall_threshold)
            return success.item(), {
                "f1_score": f1.item(),
                "precision": precision.item(),
                "recall": recall.item(),
                "true_positives": true_positives.item(),
                "false_positives": false_positives.item(),
                "false_negatives": false_negatives.item(),
            }

    return win


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    n_samples: int = 5000,
    input_dim: int = 48,
    hidden_dim: int = 32,
    imbalance_ratio: float = 0.05,
    noise_level: float = 0.3,
    steps: int = 2000,
    weight_decay: float = 0.01,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    trials: int = 25,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    """
    Class imbalance rare event detection benchmark.

    Tests optimizer's ability to learn from severely imbalanced datasets
    where detecting rare positive events is critical.
    """
    if config:
        cfg = configs.get(config, {})
        n_samples = cfg.get("n_samples", n_samples)
        input_dim = cfg.get("input_dim", input_dim)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        imbalance_ratio = cfg.get("imbalance_ratio", imbalance_ratio)
        noise_level = cfg.get("noise_level", noise_level)

    model = ImbalancedClassifier(n_samples, input_dim, hidden_dim, imbalance_ratio, noise_level).cuda()

    base_f1 = 0.7
    base_recall = 0.8
    f1_threshold = min(1.0, win_condition_multiplier * base_f1 * imbalance_ratio * 10)
    recall_threshold = min(1.0, win_condition_multiplier * base_recall * imbalance_ratio * 8)

    trial(
        model,
        None,
        None,
        imbalanced_win_condition(f1_threshold, recall_threshold),
        steps,
        opt,
        weight_decay,
        trials=trials,
        failure_threshold=4,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
