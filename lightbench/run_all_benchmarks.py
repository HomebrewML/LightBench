import inspect
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
import traceback
from dataclasses import dataclass
from queue import Empty

import numpy as np
import typer

from lightbench.utils import SkipConfig, parse_opt_name, _HPARAM_NAMES, _DTYPE_PREREQ, opt_label

app = typer.Typer()

_DTYPE_PREFIX = {"fp32": "fp32-", "bf16+ecc": "bf16+ecc-", "bf16": "bf16-", "fp16+ecc": "fp16+ecc-", "fp16": "fp16-"}
_WATCHDOG_SLACK = 30
_GATE_MIN_PASS = 0.01
_GATE_MASS = 0.05
_POSTERIOR_Q = 0.05
_POSTERIOR_EPS = 1e-3


def _normalize_dtype(value):
    mode = str(value).strip().lower()
    if mode not in _DTYPE_PREFIX:
        raise ValueError(f"Unknown dtype '{value}'. Allowed: {', '.join(_DTYPE_PREFIX)}")
    return mode


def _ordered_dtypes(dtypes):
    return tuple(d for d in _DTYPE_PREFIX if d in dtypes)


def _validate_dtype_deps(dtypes):
    selected = set(dtypes)
    missing = [(c, p) for c, p in _DTYPE_PREREQ.items() if c in selected and p not in selected]
    if missing:
        raise ValueError(f"Missing dtype dependencies: {', '.join(f'{c} requires {p}' for c, p in missing)}")


def _task_key(name, difficulty, opt, dtype):
    return name, float(difficulty), opt, dtype


def _result_key(result):
    try:
        difficulty = float(result["difficulty"])
        if not math.isfinite(difficulty):
            return None
        return _task_key(str(result["benchmark"]), difficulty, str(result["opt"]), _normalize_dtype(result["dtype"]))
    except (KeyError, TypeError, ValueError):
        return None


def _matches_shape(result, *, steps, trials, timeout):
    try:
        return int(result["steps"]) == steps and int(result["trials"]) == trials and int(result["timeout"]) == timeout
    except (KeyError, TypeError, ValueError):
        return False


def _pass_state(result):
    if bool(result.get("success")):
        return True
    if bool(result.get("timed_out")):
        return False
    error = str(result.get("error") or "").strip()
    return False if not error or error.startswith(("Skipped", "Gated")) else None


def _fail(
    benchmark,
    difficulty,
    opt,
    dtype,
    steps,
    trials,
    timeout,
    error,
    *,
    timed_out=False,
    search_step=None,
    search_lo=None,
    search_hi=None,
):
    return {
        "benchmark": benchmark,
        "difficulty": float(difficulty),
        "opt": opt,
        "dtype": dtype,
        "success": False,
        "runtime": 0.0,
        "attempts": 0,
        "timed_out": timed_out,
        "loss": None,
        "error": error,
        "steps": int(steps),
        "trials": int(trials),
        "timeout": int(timeout),
        "winning": None,
        "search_step": search_step,
        "search_lo": search_lo,
        "search_hi": search_hi,
    }


def _winning_to_hint(winning):
    if not winning:
        return None
    try:
        vals = tuple(winning[k] for k in _HPARAM_NAMES)
        return vals if all(v is not None for v in vals) else None
    except (KeyError, TypeError):
        return None


def _best_hint(hints, *, below):
    if not hints:
        return None
    candidates = [d for d in hints if d <= below]
    if not candidates:
        return None
    return _winning_to_hint(hints[max(candidates)])


@dataclass
class _SearchState:
    grid: np.ndarray
    weights: np.ndarray
    seen: set[int]
    hints: dict[float, dict]
    evals_done: int
    attempts_done: int
    min_evals: int
    max_attempts: int
    precision: float
    slope: float

    @classmethod
    def from_results(
        cls,
        *,
        precision,
        min_evals,
        max_evals,
        prior=None,
        parent_results=None,
        prior_results=None,
        attempts_done=0,
    ):
        grid = _difficulty_grid(precision)
        weights = _posterior_weights(grid, prior)
        hints = {}
        seen = set()
        evals_done = 0
        attempts_done = int(attempts_done)
        slope = prior[1] if prior is not None else 8.0

        if parent_results:
            for d, passed, winning in parent_results:
                if passed and winning:
                    hints[d] = winning

        if prior_results:
            for d, passed, winning in prior_results:
                if passed is None:
                    continue
                evals_done += 1
                seen.add(_grid_index(grid, d))
                if passed and winning:
                    hints[d] = winning
                weights = _posterior_update(grid, weights, d, passed, slope)

        attempts_done = max(attempts_done, evals_done)
        return cls(
            grid=grid,
            weights=weights,
            seen=seen,
            hints=hints,
            evals_done=evals_done,
            attempts_done=attempts_done,
            min_evals=min_evals,
            max_attempts=max_evals,
            precision=precision,
            slope=slope,
        )

    def interval(self):
        return _posterior_interval(self.grid, self.weights, _POSTERIOR_Q)

    def done(self):
        if self.attempts_done >= self.max_attempts:
            return True
        if self.evals_done < self.min_evals:
            return False
        lo, hi = self.interval()
        return (hi - lo) < self.precision

    def propose(self):
        return _posterior_median(self.grid, self.weights, self.seen)

    def update(self, difficulty, state, winning, error):
        if error.startswith(("Skipped", "Gated")):
            state = None
        self.seen.add(_grid_index(self.grid, difficulty))
        if state is not None:
            self.weights = _posterior_update(self.grid, self.weights, difficulty, state, self.slope)
            if state and winning:
                self.hints[difficulty] = winning
            self.evals_done += 1
        self.attempts_done += 1

    def mass_above(self, threshold):
        return float(self.weights[self.grid >= threshold].sum())


def _difficulty_grid(precision):
    n = max(2, int(math.ceil(1.0 / precision)) + 1)
    return np.linspace(0.0, 1.0, n)


def _posterior_weights(grid, prior):
    if prior is None:
        weights = np.ones_like(grid)
        return weights / weights.sum()
    dstar, slope, theta_se, precision = prior
    scale = 1.0 / max(slope, 1e-3)
    if np.isfinite(theta_se):
        scale = math.hypot(scale, theta_se / max(slope, 1e-3))
    scale = max(scale, precision * 2)
    scale = min(scale, 0.5)
    weights = np.exp(-0.5 * ((grid - dstar) / scale) ** 2)
    weights += 1e-12
    return weights / weights.sum()


def _posterior_update(grid, weights, difficulty, passed, slope):
    logits = slope * (grid - difficulty)
    probs = np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )
    likelihood = probs if passed else (1.0 - probs)
    likelihood = likelihood * (1.0 - _POSTERIOR_EPS) + _POSTERIOR_EPS
    weights = weights * likelihood
    weights += 1e-12
    return weights / weights.sum()


def _posterior_interval(grid, weights, q):
    cdf = np.cumsum(weights)
    lo_idx = np.searchsorted(cdf, q, side="left")
    hi_idx = np.searchsorted(cdf, 1 - q, side="left")
    lo_idx = min(max(lo_idx, 0), len(grid) - 1)
    hi_idx = min(max(hi_idx, 0), len(grid) - 1)
    return float(grid[lo_idx]), float(grid[hi_idx])


def _posterior_median(grid, weights, seen):
    base = weights
    weights = weights.copy()
    if seen:
        seen_idx = np.array(sorted(seen), dtype=int)
        seen_idx = seen_idx[(seen_idx >= 0) & (seen_idx < len(weights))]
        if seen_idx.size:
            weights[seen_idx] = 0.0
            total = weights.sum()
            if total > 0:
                weights /= total
            else:
                weights = base
    cdf = np.cumsum(weights)
    idx = np.searchsorted(cdf, 0.5, side="left")
    idx = min(max(idx, 0), len(grid) - 1)
    return float(grid[idx])


def _posterior_entropy(weights):
    w = weights[weights > 0]
    if w.size == 0:
        return 0.0
    return float(-np.sum(w * np.log(w)))


def _grid_index(grid, difficulty):
    if len(grid) <= 1:
        return 0
    idx = int(round(float(difficulty) * (len(grid) - 1)))
    return min(max(idx, 0), len(grid) - 1)


def _payload_priority(
    *,
    opt,
    bench,
    dtypes,
    prior_results,
    attempts_by_dtype,
    gated_by_dtype,
    irt_model,
    precision,
    min_evals,
    max_evals,
):
    score = 0.0
    active = 0
    for dt in dtypes:
        if gated_by_dtype.get(dt):
            continue
        state = _SearchState.from_results(
            precision=precision,
            min_evals=min_evals,
            max_evals=max_evals,
            prior=_get_prior(irt_model, opt, bench, dt, precision),
            prior_results=prior_results.get(dt, []),
            attempts_done=attempts_by_dtype.get(dt, 0),
        )
        if state.done():
            continue
        active += 1
        score += _posterior_entropy(state.weights)
    if active == 0:
        return 0.0
    return score / active


def _build_irt_prior(path, *, max_iter, min_rows, steps=None, trials=None, timeout=None):
    if not path or not os.path.exists(path):
        return None
    try:
        from lightbench.irt_ranking import fit, load_results
    except Exception:
        return None
    data = load_results(path, steps=steps, trials=trials, timeout=timeout)
    if data is None or len(data.outcome) < min_rows:
        return None
    return fit(data, max_iter=max_iter)


def _get_prior(irt_model, opt, bench, dtype, precision):
    if irt_model is None:
        return None
    return irt_model.prior_for(opt_label(opt), bench, dtype, precision)


# --- Subprocess ---


def _run_one(gpu, name, opt, dtype, diff, steps, trials, timeout, hint):
    import lightbench
    import torch
    from lightbench import utils

    diff = float(diff)
    torch.cuda.set_device(gpu)
    utils._BENCHMARK_TIMEOUT_SECONDS = timeout
    utils._trial_hint = hint

    mod = lightbench.load(name)
    arguments = {
        "steps": steps,
        "weight_decay": 0,
        "opt": f"{_DTYPE_PREFIX[dtype]}{opt}",
        "trials": trials,
        "win_condition_multiplier": 1.0,
        "config": diff,
        "dtype": "float32",
    }
    sig = inspect.signature(mod.main)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        arguments = {k: v for k, v in arguments.items() if k in sig.parameters}

    mod.main(**arguments)

    r = utils._trial_result
    loss = r["best_loss"]
    winning = r["winning"]
    if winning:
        winning = {k: (float(v) if math.isfinite(v) else None) for k, v in winning.items()}
    return {
        "benchmark": name,
        "difficulty": diff,
        "opt": opt,
        "dtype": dtype,
        "success": bool(r["did_win"]),
        "runtime": float(r["elapsed"]),
        "loss": float(loss) if loss is not None and math.isfinite(loss) else None,
        "attempts": int(r["attempts"]),
        "timed_out": bool(r["timed_out"]),
        "error": "",
        "steps": int(steps),
        "trials": int(trials),
        "timeout": int(timeout),
        "winning": winning,
    }


def _subprocess_loop(task_q, result_q, gpu, warmup_event):
    import torch
    from lightbench import utils

    torch.cuda.set_device(gpu)
    sys.stdout = open(os.devnull, "w")
    utils._warmup_event = warmup_event

    while True:
        try:
            kwargs = task_q.get()
        except (EOFError, OSError):
            return
        if kwargs is None:
            return
        try:
            result_q.put(("ok", _run_one(gpu=gpu, **kwargs), ""))
        except SkipConfig as e:
            result_q.put(("skip", None, str(e)))
        except Exception:
            result_q.put(("error", None, traceback.format_exc()))


class _SubprocessRunner:
    def __init__(self, gpu):
        self.gpu = gpu
        self._proc = self._tq = self._rq = None

    def _start(self):
        ctx = multiprocessing.get_context("spawn")
        self._tq = ctx.Queue()
        self._rq = ctx.Queue()
        self._warmup_event = ctx.Event()
        self._proc = ctx.Process(target=_subprocess_loop, args=(self._tq, self._rq, self.gpu, self._warmup_event), daemon=False)
        self._proc.start()

    def _kill(self):
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2)
            self._proc = self._tq = self._rq = None

    def run(self, watchdog_timeout, **kwargs):
        if self._proc is None or not self._proc.is_alive():
            self._start()
        self._warmup_event.clear()
        self._tq.put(kwargs)
        poll = 5.0
        elapsed = 0.0
        deadline = float(watchdog_timeout + _WATCHDOG_SLACK) if watchdog_timeout > 0 else float("inf")
        warmup_done = False
        while True:
            try:
                msg = self._rq.get(timeout=poll)
            except Empty:
                if not self._proc.is_alive():
                    self._kill()
                    return "error", None, "Subprocess died"
                if not warmup_done and self._warmup_event.is_set():
                    warmup_done = True
                    elapsed = 0.0
                if warmup_done:
                    elapsed += poll
                    if elapsed >= deadline:
                        self._kill()
                        return "timeout", None, f"Hard timeout after {watchdog_timeout}s"
                continue
            if msg[0] == "error":
                self._kill()
            return msg

    def close(self):
        if self._proc is not None and self._proc.is_alive():
            try:
                self._tq.put(None)
                self._proc.join(timeout=5)
            except Exception:
                pass
        self._kill()


# --- Worker ---


def adaptive_search(
    runner,
    name,
    opt,
    dtype,
    steps,
    trials,
    timeout,
    min_evals,
    max_evals,
    precision,
    parent_results=None,
    prior_results=None,
    irt_model=None,
    attempts_done=0,
):
    results = []
    state = _SearchState.from_results(
        precision=precision,
        min_evals=min_evals,
        max_evals=max_evals,
        prior=_get_prior(irt_model, opt, name, dtype, precision),
        parent_results=parent_results,
        prior_results=prior_results,
        attempts_done=attempts_done,
    )

    while not state.done():
        lo, hi = state.interval()
        mid = state.propose()
        hint = _best_hint(state.hints, below=mid)

        print(f"{'[hint] ' if hint else ''}Start: {name}-{mid:.4f} | {opt} | {dtype}", flush=True)
        task_kwargs = dict(
            name=name,
            opt=opt,
            dtype=dtype,
            diff=mid,
            steps=steps,
            trials=trials,
            timeout=timeout,
            hint=hint,
        )
        kind, result, err = runner.run(timeout, **task_kwargs)

        if kind == "skip":
            result = _fail(name, mid, opt, dtype, steps, trials, timeout, f"Skipped: {err}")
        elif kind == "timeout":
            result = _fail(name, mid, opt, dtype, steps, trials, timeout, err, timed_out=True)
        elif kind != "ok":
            result = _fail(name, mid, opt, dtype, steps, trials, timeout, err or "Unknown error")

        result["search_step"] = state.attempts_done
        result["search_lo"] = lo
        result["search_hi"] = hi
        results.append(result)

        state.update(
            mid,
            _pass_state(result),
            result.get("winning"),
            str(result.get("error") or "").strip(),
        )

    return results


def worker(task_queue, result_queue, worker_index, timeout, min_evals, max_evals, precision, irt_model):
    import torch

    gpu = worker_index % torch.cuda.device_count()
    runner = _SubprocessRunner(gpu)

    try:
        while True:
            payload = task_queue.get()
            if payload is None:
                return

            name, opt, steps, trials, dtypes, prior_results, prior_attempts, gated_by_dtype = payload
            accumulated = {dt: list(results) for dt, results in prior_results.items()}
            attempts = dict(prior_attempts)

            for dt in dtypes:
                if gated_by_dtype.get(dt):
                    continue
                parent = _DTYPE_PREREQ.get(dt)
                prior = accumulated.get(dt, [])
                if parent and parent in accumulated:
                    parent_state = _SearchState.from_results(
                        precision=precision,
                        min_evals=min_evals,
                        max_evals=max_evals,
                        prior_results=accumulated[parent],
                        attempts_done=attempts.get(parent, 0),
                    )
                    if parent_state.evals_done >= min_evals:
                        if parent_state.mass_above(_GATE_MIN_PASS) < _GATE_MASS:
                            result_queue.put(
                                _fail(
                                    name,
                                    0.0,
                                    opt,
                                    dt,
                                    steps,
                                    trials,
                                    timeout,
                                    f"Gated: {parent} failed",
                                    search_step=0,
                                    search_lo=0.0,
                                    search_hi=1.0,
                                )
                            )
                            continue

                search_state = _SearchState.from_results(
                    precision=precision,
                    min_evals=min_evals,
                    max_evals=max_evals,
                    prior=_get_prior(irt_model, opt, name, dt, precision),
                    parent_results=accumulated.get(parent, []) if parent else None,
                    prior_results=prior,
                    attempts_done=attempts.get(dt, 0),
                )
                if search_state.done():
                    continue

                search_results = adaptive_search(
                    runner,
                    name,
                    opt,
                    dt,
                    steps,
                    trials,
                    timeout,
                    min_evals,
                    max_evals,
                    precision,
                    parent_results=accumulated.get(parent, []) if parent else None,
                    prior_results=prior,
                    irt_model=irt_model,
                    attempts_done=attempts.get(dt, 0),
                )
                new_results = []
                for r in search_results:
                    error = str(r.get("error") or "").strip()
                    if error.startswith(("Skipped", "Gated")):
                        continue
                    state = _pass_state(r)
                    if state is None:
                        continue
                    new_results.append((r["difficulty"], state, r.get("winning")))
                if prior or new_results:
                    accumulated[dt] = prior + new_results

                for r in search_results:
                    result_queue.put(r)
    except Exception:
        traceback.print_exc()
    finally:
        runner.close()
        result_queue.put(None)


def _load_resume(path, *, steps, trials, timeout):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _result_key(row)
            if key is None or not _matches_shape(row, steps=steps, trials=trials, timeout=timeout):
                continue
            if _pass_state(row) is None:
                continue
            out[key] = row
    return out


# --- CLI ---


@app.command()
def main(
    opt: list[str] = typer.Option([], help="Optimizers"),
    steps: int = 100_000,
    timeout: int = 3600,
    trials: int = 1000,
    dtype: list[str] = typer.Option(["fp32"], help="Storage dtype(s)"),
    parallelism: int = typer.Option(8, help="Max worker processes"),
    caution: bool = False,
    mars: bool = False,
    unscaled_caution: bool = False,
    min_evals: int = typer.Option(4, help="Minimum adaptive evaluations"),
    max_evals: int = typer.Option(8, help="Maximum adaptive evaluations"),
    precision: float = typer.Option(1 / 128, help="Stop when uncertainty is below this"),
    irt_prior: str | None = typer.Option(None, help="Results log to seed IRT priors"),
    irt_max_iter: int = typer.Option(200, help="Max IRT optimizer iterations"),
    irt_min_rows: int = typer.Option(200, help="Minimum rows to fit IRT priors"),
    no_irt: bool = typer.Option(False, help="Disable IRT priors"),
    sampling: str = typer.Option("full", help="Sampling policy: full, entropy, random"),
    max_payloads: int | None = typer.Option(None, help="Limit number of payloads"),
    include_benchmark: list[str] = typer.Option([], help="Only these benchmarks"),
    exclude_benchmark: list[str] = typer.Option([], help="Skip these benchmarks"),
    progress_log: str = "benchmark_results.jsonl",
    resume: bool = False,
):
    import heavyball
    import torch
    from lightbench import available

    if parallelism < 1:
        raise ValueError("--parallelism must be >= 1")
    if min_evals < 1:
        raise ValueError("--min-evals must be >= 1")
    if max_evals < min_evals:
        raise ValueError("--max-evals must be >= --min-evals")
    if precision <= 0:
        raise ValueError("--precision must be > 0")
    if sampling not in {"full", "entropy", "random"}:
        raise ValueError("--sampling must be one of: full, entropy, random")
    if max_payloads is not None and max_payloads < 1:
        raise ValueError("--max-payloads must be >= 1")

    all_benchmarks = available()
    bench_set = set(all_benchmarks)
    for label, s in [("include", set(include_benchmark)), ("exclude", set(exclude_benchmark))]:
        unknown = sorted(s - bench_set)
        if unknown:
            raise ValueError(f"Unknown benchmarks in --{label}-benchmark: {unknown}")

    benchmarks = [
        b
        for b in all_benchmarks
        if (not include_benchmark or b in set(include_benchmark)) and b not in set(exclude_benchmark)
    ]
    if not benchmarks:
        raise ValueError("No benchmarks found")

    dtypes = list(dict.fromkeys(_normalize_dtype(v) for v in dtype))
    _validate_dtype_deps(dtypes)

    if mars:
        opt = [f"mars-{o}" for o in opt]
    if caution:
        opt = [f"cautious-{o}" for o in opt]
    if unscaled_caution:
        opt = [f"unscaled_cautious-{o}" for o in opt]
    if not opt:
        raise ValueError("No optimizers specified")

    for o in opt:
        name, _ = parse_opt_name(o)
        if not callable(getattr(heavyball, name, None)):
            raise ValueError(f"Unknown optimizer '{o}' (resolved: '{name}')")

    log_dir = os.path.dirname(progress_log)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    prior_log = irt_prior or (progress_log if os.path.exists(progress_log) else None)
    irt_model = (
        None
        if no_irt
        else _build_irt_prior(
            prior_log,
            max_iter=irt_max_iter,
            min_rows=irt_min_rows,
            steps=steps,
            trials=trials,
            timeout=timeout,
        )
    )

    results_by_key = _load_resume(progress_log, steps=steps, trials=trials, timeout=timeout) if resume else {}
    if not resume:
        open(progress_log, "w").close()

    scope = (set(benchmarks), set(opt), set(dtypes))
    results_by_key = {
        k: v for k, v in results_by_key.items() if k[0] in scope[0] and k[2] in scope[1] and k[3] in scope[2]
    }

    grouped = {}
    for (bench, diff, opt_name, dtype_name), row in results_by_key.items():
        grouped.setdefault((bench, opt_name), {}).setdefault(dtype_name, []).append(row)

    ordered_dtypes = _ordered_dtypes(dtypes)
    payloads = []
    for o, name in itertools.product(opt, benchmarks):
        prior_results = {}
        attempts_by_dtype = {}
        gated_by_dtype = {}
        for dt in ordered_dtypes:
            rows = grouped.get((name, o), {}).get(dt, [])
            attempts_by_dtype[dt] = len(rows)
            gated_by_dtype[dt] = any(str(row.get("error") or "").strip().startswith("Gated") for row in rows)
            tuples = []
            for row in rows:
                error = str(row.get("error") or "").strip()
                if error.startswith(("Skipped", "Gated")):
                    continue
                state = _pass_state(row)
                if state is None:
                    continue
                tuples.append((float(row["difficulty"]), state, row.get("winning")))
            if tuples:
                prior_results[dt] = tuples

        needs_run = False
        for dt in ordered_dtypes:
            if gated_by_dtype.get(dt):
                continue
            state = _SearchState.from_results(
                precision=precision,
                min_evals=min_evals,
                max_evals=max_evals,
                prior=_get_prior(irt_model, o, name, dt, precision),
                prior_results=prior_results.get(dt, []),
                attempts_done=attempts_by_dtype.get(dt, 0),
            )
            if not state.done():
                needs_run = True
                break
        if needs_run:
            payloads.append((
                name,
                o,
                steps,
                trials,
                ordered_dtypes,
                prior_results,
                attempts_by_dtype,
                gated_by_dtype,
            ))

    if sampling == "entropy":
        payloads.sort(
            key=lambda p: _payload_priority(
                opt=p[1],
                bench=p[0],
                dtypes=p[4],
                prior_results=p[5],
                attempts_by_dtype=p[6],
                gated_by_dtype=p[7],
                irt_model=irt_model,
                precision=precision,
                min_evals=min_evals,
                max_evals=max_evals,
            ),
            reverse=True,
        )
    elif sampling == "random":
        random.shuffle(payloads)

    if max_payloads is not None:
        payloads = payloads[:max_payloads]

    total = len(payloads) * max_evals * len(ordered_dtypes)
    if not total:
        raise ValueError("No tasks to run")
    if not payloads:
        print("No pending tasks.")
        return

    cuda_count = torch.cuda.device_count()
    if cuda_count < 1:
        raise RuntimeError("No CUDA devices")
    n_workers = min(parallelism, cuda_count, len(payloads))

    if sampling == "full":
        random.shuffle(payloads)
    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for p in payloads:
        task_queue.put(p)
    for _ in range(n_workers):
        task_queue.put(None)

    procs = []
    for idx in range(n_workers):
        p = ctx.Process(
            target=worker,
            args=(task_queue, result_queue, idx, timeout, min_evals, max_evals, precision, irt_model),
            daemon=False,
        )
        p.start()
        procs.append(p)

    print(f"Queued {len(payloads)} payloads ({n_workers} workers)")

    completed = len(results_by_key)
    done_workers = 0
    interrupted = False
    log_f = open(progress_log, "a", buffering=1)

    try:
        while done_workers < n_workers:
            try:
                result = result_queue.get(timeout=5)
            except Empty:
                exited = sum(not p.is_alive() for p in procs)
                if exited > done_workers:
                    lost = exited - done_workers
                    print(f"Warning: {lost} worker(s) died without reporting")
                    done_workers = exited
                continue
            if result is None:
                done_workers += 1
                continue
            key = _result_key(result)
            if key is None:
                continue
            if key not in results_by_key:
                completed += 1
            results_by_key[key] = result
            try:
                log_f.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                log_f.flush()
            except (TypeError, ValueError) as e:
                print(f"Warning: failed to serialize result for {result.get('benchmark')}: {e}")
            tag = "\u2713" if result["success"] else "\u2717"
            difficulty = result.get("difficulty")
            diff_label = f"{difficulty:.4f}" if isinstance(difficulty, (float, int)) else str(difficulty)
            print(f"[{completed}/{total}] {result['benchmark']}-{diff_label} - {result['opt']}: {tag}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving progress...")
    finally:
        log_f.close()
        task_queue.cancel_join_thread()
        result_queue.cancel_join_thread()
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=1)
        print(f"Results written to {progress_log}")


if __name__ == "__main__":
    app()
