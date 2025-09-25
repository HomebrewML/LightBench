# LightBench

[![PyPI - Version](https://img.shields.io/pypi/v/lightbench.svg)](https://pypi.org/project/lightbench)

LightBench is the standalone benchmark suite for HeavyBall optimizers. It curates dozens of stiff, stochastic, and
resource-intensive optimization workloads so you can benchmark new PyTorch optimizers with the same rigor used inside
HeavyBall.

## Highlights

- 50+ ready-to-run tasks spanning convex, non-convex, multi-objective, and data-imbalanced regimes.
- Typer-powered CLI apps with consistent flags for optimizers, schedules, logging, and hardware placement.
- Reproducible harness that streams metrics to stdout and captures Markdown reports and CSV artifacts by default.
- Ships with helper utilities and visualizations, from the original HeavyBall benchmark module.

## Installation

LightBench targets Python 3.9+ and PyTorch 2.1 through 2.x. The suite currently depends on HeavyBall 2.0 or newer for
optimizer implementations.

### Stable release (PyPI)

```bash
pip install lightbench
```

The wheel publishes the `lightbench` console script and allows `python -m lightbench` entry points in any
environment.

### From source (preview channel)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install exposes the same console script while tracking the main branch.

## Quick Start

Run the full benchmark harness after installation to reproduce the internal HeavyBall dashboards:

```bash
lightbench --opt ForeachSOAP --opt AdamW --steps 100000 --difficulties trivial medium
```

Every CLI entry supports `--help`. For the global runner:

```bash
lightbench --help
```

By default LightBench streams progress to stdout and stores aggregate results in `benchmark_results.md` in the working directory.

## Running Individual Benchmarks

Each benchmark is a Python module under the `lightbench` namespace and can be executed directly:

```bash
python -m lightbench.beale --opt ForeachSOAP
python -m lightbench.class_imbalance_rare --opt Adam
python -m lightbench.xor_sequence --help
```

All modules expose consistent flags for optimizers (`--opt`), device placement (`--device cuda:0`), seeds, and
termination criteria.

## Benchmark Families

LightBench organizes tasks into thematic groups to help you target problem classes quickly:

- **Deterministic classics**: Rosenbrock, Beale, Himmelblau, and other analytic surfaces for rapid optimizer sanity checks.
- **Stochastic data pipelines**: Class imbalance, noisy regression, and streaming reinforcement learning probes stress
  variance handling and adaptive scheduling.
- **Multi-objective and constrained**: Pareto, frontier, and penalty-based suites validate trade-off handling.
- **High-memory stressors**: Sparse attention, sequence XOR, and large-batch language workloads surface allocator and
  checkpointing regressions.
- **AutoML integrations**: Seamless Optuna/OptunaHub hooks enable hyperparameter sweeps and collaborative experiment
  sharing.

Use the Python API to enumerate available tasks and metadata:

```python
import lightbench

print(lightbench.available())
```

## Outputs & Artifacts

- Markdown summaries: `benchmark_results.md`
- Tabular metrics: `*/metrics.csv`
- Optional plots and GIFs from helper scripts in `lightbench/tools`

Pass `--report-dir <path>` to direct outputs elsewhere, or `--no-write` to rely solely on stdout.

## Integrating With HeavyBall

LightBench imports optimizers, schedulers, and utilities from `heavyball`. Install HeavyBall from source or PyPI before
running benchmarks against unreleased optimizer changes. When developing both packages locally, keep them in the same
environment to ensure Typer entry points locate matching versions.

Historical HeavyBall benchmark docs now live in `HeavyBall/docs/benchmark.md` and link back to this project for
deployment steps, dataset descriptions, and canonical results.

## Repository Layout

- `lightbench/`: Benchmarks, Typer CLIs, utilities, and helper scripts.
- `lightbench/data/`: Bundled CSVs, Markdown reports, and supporting assets distributed with the wheel.
- `pyproject.toml`: Project metadata, dependencies, and console-script definitions.
- `README.md`: You are here.

## Contributing & Support

- File issues or feature requests in the HeavyBall repository while LightBench is in preview.
- Use discussions to propose new benchmark families or share reproducibility reports.
- Run `pip install -e .[dev]` for plotting extras when developing notebooks or visualizations.

## License

LightBench is released under the BSD 3-Clause License, matching the HeavyBall optimizer library.
