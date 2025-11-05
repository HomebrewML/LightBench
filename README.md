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

## H100 Deployment

LightBench now auto-configures its runtime for single-GPU Hopper (H100) nodes and pins execution to GPU 0 by default. The configuration layer enables TF32/BF16 kernels, flash attention, and sets `torch.cuda.set_device` based on `LIGHTBENCH_DEVICE` (defaults to `cuda:0`). Override the device string as needed when running multiple workers on the same host.

### Bare-Metal Setup

- Install NVIDIA drivers ≥ 535 with CUDA 12.4 runtime libraries.
- Create/activate your Python environment and install the Hopper wheel of PyTorch:

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  export LIGHTBENCH_DEVICE=cuda:0
  pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
  pip install -e .
  ```

- Run the benchmark suite; all jobs will target GPU 0:

  ```bash
  lightbench run-all --opt ForeachSOAP --steps 200000 --difficulties trivial medium hard
  ```

### Container Workflow

- Build the tuned CUDA 12.4 image (ships with the correct PyTorch wheels and defaults to GPU 0):

  ```bash
  docker build -t lightbench:h100 .
  ```

- Launch the worker or CLI, overriding `LIGHTBENCH_DEVICE` if you need a different GPU mapping:

  ```bash
  docker run --gpus device=0 -it \
    -e LIGHTBENCH_DEVICE=cuda:0 \
    lightbench:h100 \
    lightbench --opt ForeachSOAP --steps 50000 --difficulties trivial
  ```

When orchestrating through the web worker (`lightbench-worker`), the same environment variables ensure each worker honours the GPU assignment without requiring per-process edits.

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

## Web Leaderboard & Submission Portal

The `lightbench.web` package exposes a FastAPI application that powers the compute-aware leaderboard and submission
flow described in `goal.md`.

### Local development

```bash
uvicorn lightbench.web.app:create_app --factory --reload
```

By default the service persists state to `./state/lightbench_submissions.db`. Override with
`LIGHTBENCH_DB_PATH=/path/to/db.sqlite`. Leaderboard snapshots are stored alongside at `leaderboard.json`.

### Vast.ai deployment

The same FastAPI control plane now drives a federated queue suitable for Vast.ai 4090 instances. The workflow uses a
single **control server** (no GPU requirements) plus any number of **worker nodes** with GPUs.

1. Build and publish the Docker image:

   ```bash
   docker build -t <registry>/lightbench:latest .
   docker push <registry>/lightbench:latest
   ```

2. **Control plane (CPU host).** Run the container with a mounted volume for the SQLite database and leaderboard state:

   ```bash
   docker run -d \
     -p 8000:8000 \
     -v $PWD/control-state:/state \
     -e LIGHTBENCH_DB_PATH=/state/lightbench.db \
     -e LIGHTBENCH_LEADERBOARD_STATE=/state/leaderboard.json \
     -e LIGHTBENCH_WORKER_TOKEN=<shared-secret> \
     <registry>/lightbench:latest \
     uvicorn lightbench.web.app:create_app --factory --host 0.0.0.0 --port 8000
   ```

   The built-in API serves:

   - `/submit` for browser submissions.
   - `/api/submissions` for REST automation.
   - `/api/worker/*` endpoints used by GPU workers.

3. **Worker nodes (Vast 4090 instances).** Launch the same image with GPU access and point it at the control plane:

   ```bash
   docker run --gpus all -it \
     -e PYTHONPATH=/opt/lightbench \
     <registry>/lightbench:latest \
     lightbench-worker \
       --api-url https://<control-plane-host>:8000 \
       --token <shared-secret> \
       --workspace /workspace/cache
   ```

   Each worker process fetches jobs from the global queue, evaluates the optimizer on the attached GPU, and streams
   metrics back to the control plane. Add more Vast nodes to scale throughput; all workers share the same queue via the
   REST API.

4. Optional: run the worker in `--once` mode for spot jobs or set `--poll-interval` / `--lease-seconds` to match your
   instance lifetimes.

The `/api/jobs/{id}` endpoint remains available when the optional in-process queue is enabled (used mainly for local
development).

### Vast automation helpers

Two Typer CLIs ship with the package once installed:

- `lightbench-vast offers` – quick filter for rentable 4090 offers (`--max-price`, `--min-vram`, `--location`).
- `lightbench-vast launch-worker` – spins up a GPU worker and auto-runs `lightbench-worker` against your control plane:

  ```bash
  lightbench-vast launch-worker 123456 \
    --control-url https://lightbench.example.com \
    --image <registry>/lightbench:latest \
    --disk-gb 80 \
    --worker-token $LIGHTBENCH_WORKER_TOKEN
  ```

- `lightbench-vast instances` / `lightbench-vast destroy <id>` – basic lifecycle management.

All commands look for `VAST_API_KEY` (or `.env`) for authentication.

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
- `lightbench/web/`: FastAPI app, storage layer, item bank, and Vast-compatible queue endpoints for the hosted leaderboard.
- `lightbench/data/`: Bundled CSVs, Markdown reports, and supporting assets distributed with the wheel.
- `pyproject.toml`: Project metadata, dependencies, and console-script definitions.
- `README.md`: You are here.

## Contributing & Support

- File issues or feature requests in the HeavyBall repository while LightBench is in preview.
- Use discussions to propose new benchmark families or share reproducibility reports.
- Run `pip install -e .[dev]` for plotting extras when developing notebooks or visualizations.

## License

LightBench is released under the BSD 3-Clause License, matching the HeavyBall optimizer library.
