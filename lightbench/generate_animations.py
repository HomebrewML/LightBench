import importlib
import math
import pathlib
from copy import deepcopy
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from utils import get_optim as get_optimizer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    benchmark_name: str = typer.Option(..., help="Name of the benchmark to run (e.g., 'saddle_point')."),
    optimizer_names: List[str] = typer.Option(
        ..., "--optimizer-name", help="Name of an optimizer to include in the comparison."
    ),
    output_file: str = typer.Option(..., help="Path to save the generated GIF."),
    steps: int = 100,
):
    """
    Generates an animated GIF of optimizer paths on a given benchmark.
    """
    try:
        benchmark_module = importlib.import_module(f"benchmark.{benchmark_name}")
    except ImportError:
        print(f"Error: Benchmark '{benchmark_name}' not found.")
        raise typer.Exit(1)

    if benchmark_name == "saddle_point":
        model = benchmark_module.Model(1, 0)

        def objective_fn(*xs):
            return benchmark_module.objective(*xs, power=model.power)

        x_limits, y_limits = (-2, 2), (-2, 2)
    elif benchmark_name == "rosenbrock":
        coords = (-7, 4)
        model = benchmark_module.Model(coords)
        objective_fn = benchmark_module.objective
        x_limits, y_limits = (-10, 10), (-10, 10)
    elif benchmark_name == "beale":
        coords = (-7, -4)
        model = benchmark_module.Model(coords)
        objective_fn = benchmark_module.objective
        x_limits, y_limits = (-8, 2), (-8, 2)
    else:
        print(f"Error: Benchmark '{benchmark_name}' is not supported for animation yet.")
        raise typer.Exit(1)

    trajectories = []
    models = []
    for optimizer_name in optimizer_names:
        print(f"Tuning LR for {optimizer_name}...")
        best_lr = None
        min_loss = float("inf")
        tuning_steps = math.ceil(steps / 3)
        lr_candidates = np.logspace(-8, 0, 50)

        for test_lr in lr_candidates:
            temp_model = deepcopy(model)
            temp_optimizer = get_optimizer(optimizer_name, temp_model.parameters(), lr=test_lr)

            def _closure():
                loss = temp_model()
                loss.backward()

            for _ in range(tuning_steps):
                temp_optimizer.zero_grad()
                temp_optimizer.step(_closure)

            final_loss = temp_model().item()
            if final_loss < min_loss:
                min_loss = final_loss
                best_lr = test_lr

        print(f"  > Best LR for {optimizer_name}: {best_lr:.5f}")

        m = deepcopy(model)
        models.append(m)
        optimizer = get_optimizer(optimizer_name, m.parameters(), lr=best_lr)

        trajectory = [m.param.detach().clone()]

        def _closure():
            loss = m()
            loss.backward()

        for _ in range(steps):
            optimizer.zero_grad()
            optimizer.step(_closure)
            trajectory.append(m.param.detach().clone())

        trajectories.append(list(torch.stack(trajectory).cpu().numpy()))
        print(f"  > Final position for {optimizer_name}: {trajectories[-1][-1]}")

    target_trajectory = None
    if benchmark_name == "dynamic_targets":
        target_trajectory = models[0].get_target_trajectory()

    paths_xy = []
    for traj in trajectories:
        path_array = np.array(traj)
        paths_xy.append((path_array[:, 0], path_array[:, 1]))

    target_path_xy = None
    if target_trajectory:
        target_array = np.array(target_trajectory)
        target_path_xy = (target_array[:, 0], target_array[:, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{benchmark_name.replace('_', ' ').title()} with {', '.join(optimizer_names)}")

    if objective_fn:
        x = torch.linspace(x_limits[0], x_limits[1], 100)
        y = torch.linspace(y_limits[0], y_limits[1], 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = objective_fn(X, Y)
        Z -= Z.min()
        Z = Z.log()
        ax.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=50, cmap="viridis")
        ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=50, colors="k", linewidths=1)

    lines = [ax.plot([], [], "x-", label=name)[0] for name in optimizer_names]
    ax.legend()

    target_dot = None
    if benchmark_name == "dynamic_targets":
        (target_dot,) = ax.plot([], [], "ro", label="Target")
        ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        if target_dot:
            target_dot.set_data([], [])
            return tuple(lines) + (target_dot,)
        return tuple(lines)

    frames = min(100, steps + 1)

    def animate(i):
        i = math.ceil(i / frames * steps)
        for j, line in enumerate(lines):
            x_data, y_data = paths_xy[j]
            line.set_data(x_data[: i + 1], y_data[: i + 1])

        if target_path_xy:
            target_x, target_y = target_path_xy
            if i > 0:
                target_dot.set_data([target_x[i - 1]], [target_y[i - 1]])
            else:
                target_dot.set_data([], [])
            return tuple(lines) + (target_dot,)

        return tuple(lines)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=100, blit=True)

    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer="ffmpeg", fps=10)
    print(f"Animation saved to {output_file}")


if __name__ == "__main__":
    app()
