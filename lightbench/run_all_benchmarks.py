import inspect
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
import traceback
from queue import Empty

import typer

from lightbench.utils import SkipConfig, parse_opt_name, _HPARAM_NAMES

app = typer.Typer()

_DIFFICULTIES = ("trivial", "easy", "medium", "hard", "extreme", "nightmare")
_DTYPE_PREFIX = {"fp32": "fp32-", "bf16+ecc": "bf16+ecc-", "bf16": "bf16-", "fp16+ecc": "fp16+ecc-", "fp16": "fp16-"}
_DTYPE_PREREQ = {"bf16+ecc": "fp32", "bf16": "bf16+ecc", "fp16+ecc": "fp32", "fp16": "fp16+ecc"}
_WATCHDOG_SLACK = 30


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
    return name, difficulty, opt, dtype


def _result_key(result):
    try:
        return _task_key(str(result["benchmark"]), str(result["difficulty"]),
                         str(result["opt"]), _normalize_dtype(result["dtype"]))
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


def _fail(benchmark, difficulty, opt, dtype, steps, trials, timeout, error, *, timed_out=False):
    return {
        "benchmark": benchmark, "difficulty": difficulty,
        "opt": opt, "dtype": dtype, "success": False, "runtime": 0.0, "attempts": 0,
        "timed_out": timed_out, "loss": None, "error": error,
        "steps": int(steps), "trials": int(trials), "timeout": int(timeout),
        "winning": None,
    }


def _winning_to_hint(winning):
    if not winning:
        return None
    try:
        vals = tuple(winning[k] for k in _HPARAM_NAMES)
        return vals if all(v is not None for v in vals) else None
    except (KeyError, TypeError):
        return None


# --- Subprocess ---


def _run_one(gpu, name, opt, dtype, diff, steps, trials, timeout, hint):
    import lightbench
    import torch
    from lightbench import utils

    torch.cuda.set_device(gpu)
    utils._BENCHMARK_TIMEOUT_SECONDS = timeout
    utils._trial_hint = hint

    mod = lightbench.load(name)
    args = {
        "steps": steps, "weight_decay": 0, "opt": f"{_DTYPE_PREFIX[dtype]}{opt}",
        "trials": trials, "win_condition_multiplier": 1.0, "config": diff, "dtype": "float32",
    }
    sig = inspect.signature(mod.main)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        args = {k: v for k, v in args.items() if k in sig.parameters}

    mod.main(**args)

    r = utils._trial_result
    loss = r["best_loss"]
    winning = r["winning"]
    if winning:
        winning = {k: (float(v) if math.isfinite(v) else None) for k, v in winning.items()}
    return {
        "benchmark": name, "difficulty": diff,
        "opt": opt, "dtype": dtype,
        "success": bool(r["did_win"]), "runtime": float(r["elapsed"]),
        "loss": float(loss) if loss is not None and math.isfinite(loss) else None,
        "attempts": int(r["attempts"]),
        "timed_out": bool(r["timed_out"]), "error": "",
        "steps": int(steps), "trials": int(trials), "timeout": int(timeout),
        "winning": winning,
    }


def _subprocess_loop(task_q, result_q, gpu):
    import torch
    torch.cuda.set_device(gpu)
    sys.stdout = open(os.devnull, "w")

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
        self._proc = ctx.Process(target=_subprocess_loop, args=(self._tq, self._rq, self.gpu), daemon=False)
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
        self._tq.put(kwargs)
        poll = min(5.0, float(watchdog_timeout + _WATCHDOG_SLACK)) if watchdog_timeout > 0 else 5.0
        elapsed = 0.0
        deadline = float(watchdog_timeout + _WATCHDOG_SLACK) if watchdog_timeout > 0 else float("inf")
        while elapsed < deadline:
            try:
                msg = self._rq.get(timeout=poll)
            except Empty:
                elapsed += poll
                if not self._proc.is_alive():
                    self._kill()
                    return "error", None, "Subprocess died"
                continue
            if msg[0] == "error":
                self._kill()
            return msg
        self._kill()
        return "timeout", None, f"Hard timeout after {watchdog_timeout}s"

    def close(self):
        if self._proc is not None and self._proc.is_alive():
            try:
                self._tq.put(None)
                self._proc.join(timeout=5)
            except Exception:
                pass
        self._kill()


# --- Worker ---


def worker(task_queue, result_queue, worker_index, timeout):
    import torch
    gpu = worker_index % torch.cuda.device_count()
    runner = _SubprocessRunner(gpu)

    try:
        while True:
            payload = task_queue.get()
            if payload is None:
                return

            name, opt, steps, trials, difficulties, dtypes, pending_pairs, prior_pass, prior_winning = payload
            pending = set(pending_pairs)
            passed = dict(prior_pass)
            chain_alive = {dt: True for dt in dtypes}
            chain_reason = {}
            configs = dict(prior_winning)

            max_diff = len(difficulties)

            for diff_idx, difficulty in enumerate(difficulties):
                if diff_idx >= max_diff:
                    for dt in dtypes:
                        pair = (dt, difficulty)
                        passed[pair] = False
                        if pair in pending:
                            result_queue.put(_fail(name, difficulty, opt, dt, steps, trials, timeout,
                                                   f"Skipped: {name} unsupported at {difficulty}"))
                    continue

                for dt in dtypes:
                    pair = (dt, difficulty)

                    if not chain_alive[dt]:
                        passed[pair] = False
                        if pair in pending:
                            result_queue.put(_fail(name, difficulty, opt, dt, steps, trials, timeout,
                                                   chain_reason.get(dt, f"Gated: {dt} chain")))
                        continue

                    parent = _DTYPE_PREREQ.get(dt)
                    if parent and passed.get((parent, difficulty)) is False:
                        chain_alive[dt] = False
                        chain_reason[dt] = f"Gated: {parent}-{difficulty} failed"
                        passed[pair] = False
                        if pair in pending:
                            result_queue.put(_fail(name, difficulty, opt, dt, steps, trials, timeout, chain_reason[dt]))
                        continue

                    if pair not in pending:
                        if passed.get(pair) is False:
                            chain_alive[dt] = False
                            chain_reason[dt] = f"Gated: {dt}-{difficulty} failed"
                        continue

                    hint = _winning_to_hint(configs.get((parent, difficulty))) if parent else None
                    if hint is None and diff_idx > 0:
                        hint = _winning_to_hint(configs.get((dt, difficulties[diff_idx - 1])))

                    print(f"{'[hint] ' if hint else ''}Start: {name}-{difficulty} | {opt} | {dt}", flush=True)
                    task_kwargs = dict(
                        name=name, opt=opt, dtype=dt, diff=difficulty,
                        steps=steps, trials=trials, timeout=timeout, hint=hint,
                    )
                    kind, result, err = runner.run(timeout, **task_kwargs)

                    if kind == "skip":
                        result = _fail(name, difficulty, opt, dt, steps, trials, timeout, f"Skipped: {err}")
                        max_diff = min(max_diff, diff_idx)
                    elif kind == "timeout":
                        result = _fail(name, difficulty, opt, dt, steps, trials, timeout, err, timed_out=True)
                    elif kind != "ok":
                        result = _fail(name, difficulty, opt, dt, steps, trials, timeout, err or "Unknown error")

                    result_queue.put(result)
                    state = _pass_state(result)
                    passed[pair] = state
                    if state is True and result.get("winning"):
                        configs[pair] = result["winning"]
                    if state is False:
                        chain_alive[dt] = False
                        chain_reason[dt] = str(result.get("error") or f"Gated: {dt}-{difficulty} failed")
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
    difficulties: list[str] = typer.Option([], help="Subset of difficulties"),
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

    all_benchmarks = available()
    bench_set = set(all_benchmarks)
    for label, s in [("include", set(include_benchmark)), ("exclude", set(exclude_benchmark))]:
        unknown = sorted(s - bench_set)
        if unknown:
            raise ValueError(f"Unknown benchmarks in --{label}-benchmark: {unknown}")

    benchmarks = [b for b in all_benchmarks
                  if (not include_benchmark or b in set(include_benchmark)) and b not in set(exclude_benchmark)]
    if not benchmarks:
        raise ValueError("No benchmarks found")

    dtypes = list(dict.fromkeys(_normalize_dtype(v) for v in dtype))
    _validate_dtype_deps(dtypes)

    diffs = list(dict.fromkeys(difficulties or _DIFFICULTIES))
    unknown_diffs = sorted(set(diffs) - set(_DIFFICULTIES))
    if unknown_diffs:
        raise ValueError(f"Unknown difficulties: {unknown_diffs}")
    ordered_diffs = tuple(d for d in _DIFFICULTIES if d in diffs)

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

    results_by_key = _load_resume(progress_log, steps=steps, trials=trials, timeout=timeout) if resume else {}
    if not resume:
        open(progress_log, "w").close()

    scope = (set(benchmarks), set(ordered_diffs), set(opt), set(dtypes))
    results_by_key = {k: v for k, v in results_by_key.items()
                      if k[0] in scope[0] and k[1] in scope[1] and k[2] in scope[2] and k[3] in scope[3]}

    ordered_dtypes = _ordered_dtypes(dtypes)
    payloads = []
    queued = 0
    for o, name in itertools.product(opt, benchmarks):
        pending_pairs, prior_pass, prior_winning = [], {}, {}
        for dt, diff in itertools.product(ordered_dtypes, ordered_diffs):
            key = _task_key(name, diff, o, dt)
            prior = results_by_key.get(key)
            prior_pass[(dt, diff)] = None if prior is None else _pass_state(prior)
            if prior is None:
                pending_pairs.append((dt, diff))
                queued += 1
            elif prior.get("winning"):
                prior_winning[(dt, diff)] = prior["winning"]
        if pending_pairs:
            payloads.append((name, o, steps, trials, ordered_diffs, ordered_dtypes,
                             tuple(pending_pairs), prior_pass, prior_winning))

    total = len(benchmarks) * len(opt) * len(ordered_dtypes) * len(ordered_diffs)
    if not total:
        raise ValueError("No tasks to run")
    if not queued:
        print("No pending tasks.")
        return

    cuda_count = torch.cuda.device_count()
    if cuda_count < 1:
        raise RuntimeError("No CUDA devices")
    n_workers = min(parallelism, cuda_count, len(payloads))

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
        p = ctx.Process(target=worker, args=(task_queue, result_queue, idx, timeout), daemon=False)
        p.start()
        procs.append(p)

    print(f"Queued {queued} runs across {len(payloads)} payloads ({n_workers} workers)")

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
            tag = '\u2713' if result['success'] else '\u2717'
            print(f"[{completed}/{total}] {result['benchmark']}-{result['difficulty']} - {result['opt']}: {tag}")
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
