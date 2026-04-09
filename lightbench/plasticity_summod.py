from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import disabled_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {"trivial": {"classes": 2, "items": 8, "add_class_every": 128},  # 256
    "easy": {"classes": 8, "items": 8, "add_class_every": 64},  # 512
    "medium": {"classes": 32, "items": 8, "add_class_every": 32},  # 1024
    "hard": {"classes": 128, "items": 8, "add_class_every": 16},  # 2048
    "extreme": {"classes": 512, "items": 8, "add_class_every": 8},  # 4096
    "nightmare": {"classes": 2048, "items": 8, "add_class_every": 4},  # 8192
}


class Model(nn.Module):
    def __init__(self, classes: int, items: int, add_class_every: int, features: int = 4):
        super().__init__()
        self.model = nn.Sequential(nn.Embedding(classes, features), nn.Flatten(), nn.Linear(items * features, classes),
            nn.ReLU(), nn.Linear(classes, classes), )
        self.step = 0
        self.add_class_every = add_class_every
        self.classes = classes
        self.items = items
        self.weight = nn.Buffer(torch.randn((items,)))

    def forward(self):
        data = torch.randint(0, self.classes, (16, self.items), device=self.model[0].weight.device)
        y = self.model(data)
        target = data.sum(-1) % self.classes
        loss = torch.nn.functional.cross_entropy(y, target, reduction='none')
        mask = target >= min(self.step / self.add_class_every + 2, self.classes)
        return (loss * mask).sum() / mask.sum().clamp(min=1)


def loss_win_condition(target):
    def win(model: Model, loss: float):
        if model.step <= (model.classes - 2) * model.add_class_every:
            return False, {}
        return loss <= target, {}

    return win


@app.command()
def main(dtype: str = typer.Option("float32", help="Data type to use"), steps: int = 100, weight_decay: float = 0,
        opt: str = typer.Option("SOAP", help="Optimizers to use"), show_image: bool = False, trials: int = 100,
        win_condition_multiplier: float = 1.0, config: Optional[str] = None, ):
    config = configs[config or "trivial"]
    assert steps > config["add_class_every"] * (config["classes"] - 2)
    model = Model(**config)

    win_condition = disabled_win_condition if show_image else loss_win_condition(win_condition_multiplier * 1e-8)

    trial(model, None, None, win_condition, steps, opt, weight_decay, trials=trials, return_best=False, dtype=dtype, )


if __name__ == "__main__":
    app()
