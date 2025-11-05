"""Command-line interface for the LightBench benchmark suite."""

from __future__ import annotations

import inspect
from typing import Callable

import typer

from . import available, load
from .run_all_benchmarks import main as run_all_benchmarks_main

app = typer.Typer(pretty_exceptions_enable=False, rich_markup_mode="markdown", help="LightBench benchmark suite")


@app.callback(invoke_without_command=True)
def _root_callback(ctx: typer.Context) -> None:
    """LightBench CLI entrypoint. Use a subcommand to run a benchmark."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _register_command(name: str, callback: Callable) -> None:
    """Register a benchmark callback under the provided command name."""

    # Typer uses function metadata for CLI generation; keep the original signature and docs.

    def wrapper(*args, **kwargs):
        return callback(*args, **kwargs)

    wrapper.__doc__ = callback.__doc__
    wrapper.__name__ = f"{callback.__name__}__{name}"
    wrapper.__signature__ = inspect.signature(callback)
    app.command(name=name)(wrapper)


# Register the orchestration command with a friendly name and a compatibility alias.
_register_command("run-all", run_all_benchmarks_main)
app.command(name="run_all_benchmarks", hidden=True)(run_all_benchmarks_main)


# Dynamically expose each benchmark module as a subcommand.
for module_name in available():
    module = load(module_name)
    command = getattr(module, "main", None)
    if command is None:
        continue
    _register_command(module_name, command)
