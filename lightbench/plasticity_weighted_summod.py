from typing import Optional

import torch
import torch.backends.opt_einsum
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

configs = {
    "trivial": {"classes": 16, "items": 8, "add_class_every": 2**6},
    "easy": {"classes": 16, "items": 8, "add_class_every": 2**8},
    "medium": {"classes": 16, "items": 8, "add_class_every": 2**10},
    "hard": {"classes": 16, "items": 8, "add_class_every": 2**12},
    "extreme": {"classes": 16, "items": 8, "add_class_every": 2**14},
    "nightmare": {"classes": 16, "items": 8, "add_class_every": 2**16},
}


class Model(nn.Module):
    def __init__(self, classes: int, items: int, add_class_every: int, features: int = 4, hidden: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(classes, features),
            nn.Flatten(),
            nn.Linear(items * features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes + 1),
        )
        self.step = 0
        self.add_class_every = add_class_every
        self.classes = classes
        self.items = items
        self.weight = nn.Buffer(torch.randn((items,)))

    def forward(self):
        data = torch.randint(0, self.classes, (16, self.items), device=self.model[0].weight.device)
        y = self.model(data)
        target = (data.float() @ self.weight.float()).round().long() % self.classes
        target = torch.where(target >= min(self.step / self.add_class_every + 2, self.classes), 0, target + 1)
        self.step += 1
        return torch.nn.functional.cross_entropy(y, target)


def loss_win_condition(target):
    def win(model: Model, loss: float):
        if model.step <= (model.classes - 2) * model.add_class_every:
            return False, {}
        return loss <= target, {}

    return win


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("ForeachSOAP", help="Optimizers to use"),
    show_image: bool = False,
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[str] = None,
):
    config = configs[config or "trivial"]
    assert steps > config["add_class_every"] * (config["classes"] - 2)
    model = Model(**config)

    trial(
        model,
        None,
        None,
        loss_win_condition(win_condition_multiplier * 1e-8 * (not show_image)),
        steps,
        opt,
        weight_decay,
        trials=trials,
        return_best=False,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
