"""
Transfer Domain Shift Robustness Benchmark

Tests an optimizer's ability to adapt a pre-trained model to a new domain
with distribution shift. This benchmark simulates real-world transfer learning
scenarios where models trained on one domain must adapt to related but different
domains, such as adapting a model trained on natural images to medical images.

The task involves fine-tuning a pre-trained feature extractor on a target domain
that has systematic differences from the source domain, testing the optimizer's
ability to navigate transfer learning challenges.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {
    "params": {
        "feature_dim": {"lo": 32, "hi": 1024, "scale": "log", "dtype": "int"},
        "n_classes": {"lo": 4, "hi": 128, "scale": "log", "dtype": "int"},
        "n_samples": {"lo": 500, "hi": 16000, "scale": "log", "dtype": "int"},
        "domain_shift_strength": {"lo": 0.1, "hi": 0.9, "scale": "linear"},
        "pretrain_noise": {"lo": 0.05, "hi": 0.5, "scale": "linear"},
        "freeze_layers": {"lo": 0, "hi": 5, "scale": "linear", "dtype": "int"},
    }
}


class TransferLearningModel(nn.Module):
    def __init__(self, feature_dim, n_classes, n_samples, domain_shift_strength, pretrain_noise, freeze_layers):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim // 8),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim // 8, n_classes)
        for i, layer in enumerate(self.feature_extractor):
            if isinstance(layer, nn.Linear) and i // 2 < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        self.register_buffer("source_X", torch.randn(n_samples, feature_dim))
        self.register_buffer("source_y", torch.randint(0, n_classes, (n_samples,)))
        shift_matrix = torch.randn(feature_dim, feature_dim) * domain_shift_strength
        target_X = self.source_X @ (torch.eye(feature_dim) + shift_matrix)
        target_X += pretrain_noise * torch.randn_like(target_X)
        label_shift = torch.randint(-n_classes // 4, n_classes // 4, (n_samples,))
        target_y = torch.clamp(self.source_y + label_shift, 0, n_classes - 1)
        self.register_buffer("target_X", target_X)
        self.register_buffer("target_y", target_y)
        self._initialize_pretrained_weights()

    def _initialize_pretrained_weights(self):
        with torch.no_grad():
            for _ in range(10):
                features = self.feature_extractor(self.source_X)
                features = F.normalize(features, dim=1)

    def forward(self, use_target_domain=True):
        if use_target_domain:
            X, y = self.target_X, self.target_y
        else:
            X, y = self.source_X, self.source_y
        features = self.feature_extractor(X)
        logits = self.classifier(features)
        transfer_loss = F.cross_entropy(logits, y)
        source_features = self.feature_extractor(self.source_X)
        target_features = self.feature_extractor(self.target_X)
        domain_discrepancy = (source_features.mean(0) - target_features.mean(0)).square().sum()
        total_loss = transfer_loss + 0.1 * domain_discrepancy
        return total_loss


def transfer_win_condition(accuracy_threshold, domain_gap_threshold):
    def win(model, loss):
        with torch.no_grad():
            target_features = model.feature_extractor(model.target_X)
            target_logits = model.classifier(target_features)
            target_predictions = torch.argmax(target_logits, dim=1)
            target_accuracy = (target_predictions == model.target_y).float().mean()
            source_features = model.feature_extractor(model.source_X)
            source_logits = model.classifier(source_features)
            source_predictions = torch.argmax(source_logits, dim=1)
            source_accuracy = (source_predictions == model.source_y).float().mean()
            domain_gap = abs(source_accuracy - target_accuracy)
            feature_mmd = (source_features.mean(0) - target_features.mean(0)).square().sum()
            accuracy_ok = target_accuracy >= accuracy_threshold
            domain_gap_ok = domain_gap <= domain_gap_threshold
            success = accuracy_ok and domain_gap_ok
            return success.item(), {
                "target_accuracy": target_accuracy.item(),
                "source_accuracy": source_accuracy.item(),
                "domain_gap": domain_gap.item(),
                "feature_mmd": feature_mmd.item(),
                "accuracy_ok": accuracy_ok.item(),
                "domain_gap_ok": domain_gap_ok.item(),
            }

    return win


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    feature_dim: int = 128,
    n_classes: int = 16,
    n_samples: int = 2000,
    domain_shift_strength: float = 0.5,
    pretrain_noise: float = 0.2,
    freeze_layers: int = 2,
    steps: int = 1500,
    weight_decay: float = 0.005,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    trials: int = 20,
    win_condition_multiplier: float = 1.0,
    config: Optional[float] = None,
):
    """
    Transfer learning domain shift robustness benchmark.

    Tests optimizer's ability to adapt pre-trained models to new domains
    with distribution shift, simulating real-world transfer learning scenarios.
    """
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    feature_dim = kwargs["feature_dim"]
    n_classes = kwargs["n_classes"]
    n_samples = kwargs["n_samples"]
    domain_shift_strength = kwargs["domain_shift_strength"]
    pretrain_noise = kwargs["pretrain_noise"]
    freeze_layers = kwargs["freeze_layers"]

    model = TransferLearningModel(
        feature_dim, n_classes, n_samples, domain_shift_strength, pretrain_noise, freeze_layers
    ).cuda()

    base_accuracy = 0.7
    base_domain_gap = 0.2
    accuracy_threshold = win_condition_multiplier * base_accuracy * (1 - domain_shift_strength * 0.3)
    domain_gap_threshold = win_condition_multiplier * base_domain_gap * (1 + domain_shift_strength)

    trial(
        model,
        None,
        None,
        transfer_win_condition(accuracy_threshold, domain_gap_threshold),
        steps,
        opt,
        weight_decay,
        trials=trials,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
