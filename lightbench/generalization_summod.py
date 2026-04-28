from typing import Optional

import torch
import typer
from heavyball.utils import set_torch
from torch import nn

from lightbench.utils import difficulty_to_kwargs, disabled_win_condition, loss_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

difficulty_spec = {"param": "items", "lo": 2, "hi": 12, "scale": "linear", "dtype": "int"}


class Model(nn.Module):
    def __init__(self, classes: int, items: int, features: int = 4, hidden: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(classes, features),
            nn.Flatten(),
            nn.Linear(items * features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )
        self.classes = classes
        self.items = items

    def forward(self):
        data = torch.randint(0, self.classes, (16, self.items), device=self.model[0].weight.device)
        y = self.model(data)
        target = data.sum(-1) % self.classes
        return torch.nn.functional.cross_entropy(y, target)


@app.command()
def main(
    dtype: str = typer.Option("float32", help="Data type to use"),
    steps: int = 100,
    weight_decay: float = 0,
    opt: str = typer.Option("SOAP", help="Optimizers to use"),
    show_image: bool = False,
    trials: int = 100,
    win_condition_multiplier: float = 1.0,
    config: Optional[float] = None,
):
    kwargs = difficulty_to_kwargs(difficulty_spec, config if config is not None else 0.0)
    model = Model(classes=64, items=kwargs["items"])

    win_condition = disabled_win_condition if show_image else loss_win_condition(win_condition_multiplier * 1e-2)

    trial(
        model,
        None,
        None,
        win_condition,
        steps,
        opt,
        weight_decay,
        trials=trials,
        return_best=False,
        dtype=dtype,
    )


if __name__ == "__main__":
    app()
