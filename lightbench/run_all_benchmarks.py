import inspect
import itertools
import multiprocessing
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from queue import Empty

import numpy as np
import typer

from lightbench.utils import SkipConfig

app = typer.Typer()


def last_match(pattern, text):
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return float(matches[-1])


def parse_config(text):
    match = re.search(r"Attempt: \d+ \| (.*?) \| Best Loss:", text)
    return match.group(1).strip() if match else ""


def run_benchmark(name, opt, steps, dtype, trials, seed, difficulty):
    import io

    import lightbench
    import torch

    stdout = sys.stdout
    sys.stdout = io.StringIO()
    start_time = time.time()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        module = lightbench.load(name)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        arguments = {
            "dtype": dtype,
            "steps": steps,
            "weight_decay": 0,
            "opt": opt,
            "trials": trials,
            "win_condition_multiplier": 1.0,
            "config": difficulty,
        }
        sig = inspect.signature(module.main)
        if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            arguments = {k: v for k, v in arguments.items() if k in sig.parameters}
        module.main(**arguments)
    except SkipConfig:
        raise
    except Exception:
        output = sys.stdout.getvalue()
        error = traceback.format_exc()
    else:
        output = sys.stdout.getvalue()
        error = ""
    finally:
        sys.stdout = stdout

    success = "Win: Yes" in output
    runtime = last_match(r"Took: ([0-9.]+)", output)
    loss = last_match(r"Best Loss: ([0-9.e\-+]+)", output)
    attempts = int(last_match(r"Attempt: ([0-9]+)", output) or trials)
    total_runtime = time.time() - start_time
    config_str = parse_config(output)

    return {
        "name": f"{name}-{difficulty}",
        "opt": opt,
        "success": success,
        "runtime": float(runtime or total_runtime),
        "loss": float(loss) if loss else float("inf"),
        "attempts": attempts,
        "error": error if error else "",
        "seed": seed,
        "config_str": config_str,
    }


def opt_to_config(opt):
    caution = "No"
    mars = "No"
    if opt.startswith("cautious-"):
        opt = opt[len("cautious-") :]
        caution = "Yes"
    if opt.startswith("unscaled_cautious-"):
        opt = opt[len("unscaled_cautious-") :]
        caution = "Unscaled"
    if opt.startswith("mars-"):
        opt = opt[len("mars-") :]
        mars = "Yes"
    return opt, caution, mars


def write_progress(results, opt, output):
    with open(output, "w") as f:
        f.write(f"# Benchmark Results\nGenerated: {datetime.now()}\nLast updated: {datetime.now()}\n\n")
        f.write("## Summary (In Progress)\n\n")
        f.write("| Optimizer | Caution | Mars | Success | Runtime | Average Attempts |\n")
        f.write("|-----------|---|---|---------|----------|------|\n")

        for o in opt:
            opt_results = [r for r in results if r["opt"] == o]
            if not opt_results:
                continue
            success = sum(r["success"] for r in opt_results)
            runtime = np.mean([r["runtime"] for r in opt_results if r["success"]]) if success else 0
            attempts = np.mean([r["attempts"] for r in opt_results if r["success"]]) if success else 0

            o, caution, mars = opt_to_config(o)
            f.write(f"| {o} | {caution} | {mars} | {success}/{len(opt_results)} | {runtime:.2f}s | {attempts:.1f} |\n")

        f.write("\n## Details\n\n")
        # Add Winning Config column
        f.write(
            "| Benchmark | Optimizer | Cautious | Mars | Success | Runtime | Loss | Attempts | Seed | Winning Config |\n"
        )
        f.write("|-----------|-----------|---------|---|---|----------|------|---|---|----------------|\n")

        for r in sorted(results, key=lambda x: (x["name"], x["opt"])):
            mark = "✓" if r["success"] else "✗"
            runtime = f"{r['runtime']:.2f}s"
            loss = f"{r['loss']:.2e}"
            attempts = f"{r['attempts']:d}"
            seed = f"{r['seed']}"
            config_str = f"`{r['config_str']}`" if r.get("config_str", False) else "N/A"

            opt, caution, mars = opt_to_config(r["opt"])
            f.write(
                f"| {r['name']} | {opt} | {caution} | {mars} | {mark} | {runtime} | {loss} | {attempts} | {seed} | {config_str} |\n"
            )

        if any(not r["success"] for r in results):
            f.write("\n## Errors\n\n")
            for r in sorted(results, key=lambda x: (x["name"], x["opt"])):
                if not r["success"] and r["error"] and r["error"] != "None" and r["error"] != "":
                    f.write(f"\n### {r['name']} - {r['opt']}\n```\n{r['error']}\n```\n")


_difficulty_order = ["trivial", "easy", "medium", "hard", "extreme", "nightmare"]


def worker(task_queue, result_queue, worker_index, difficulties: list, timeout: int):
    import torch

    torch.cuda.set_device(worker_index % torch.cuda.device_count())
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["HEAVYBALL_BENCHMARK_TIMEOUT"] = str(round(timeout))

    dummy = torch.zeros(1, device="cuda")
    del dummy
    torch.cuda.empty_cache()

    difficulties = [d for d in _difficulty_order if d in difficulties]

    try:
        while True:
            try:
                name, o, steps, dtype, trials, seed = task_queue.get(timeout=0.1)
            except Empty:
                return
            name, *diff = name.split("===")
            inner_difficulties = diff if diff else difficulties.copy()
            for _ in range(len(inner_difficulties)):
                d = inner_difficulties.pop(0)
                try:
                    result = run_benchmark(name, o, steps, dtype, trials, seed, d)
                    exc = ""
                except Exception as e:
                    result = isinstance(e, SkipConfig)
                    exc = str(e)

                if result is True:  # SkipConfig
                    break
                if result is not False and result is not None:
                    result_queue.put(result)
                    if result["success"]:
                        continue
                else:
                    inner_difficulties.insert(0, d)

                for d_ in inner_difficulties:
                    result = {
                        "name": f"{name}-{d_}",
                        "opt": o,
                        "success": False,
                        "runtime": timeout if "timed out" in exc else 0,
                        "attempts": 0,
                        "loss": float("inf"),
                        "error": str(exc),
                        "seed": seed,
                        "config_str": "N/A",
                    }
                    result_queue.put(result)
                break
    except Exception:
        traceback.print_exc()
    finally:
        result_queue.put(None)


@app.command()
def main(
    opt: list[str] = typer.Option([], help="Optimizers"),
    steps: int = 100_000,
    timeout: int = 3600,
    output: str = "benchmark_results.md",
    trials: int = 1000,
    dtype: str = "float32",
    parallelism: int = typer.Option(8, help="Number of parallel worker processes"),
    caution: bool = False,
    mars: bool = False,
    unscaled_caution: bool = False,
    seeds: int = 4,
    full_parallel: bool = False,
    difficulties: list[str] = typer.Option([], help=f"{_difficulty_order} or any combination of these"),
):
    multiprocessing.set_start_method("spawn", force=True)  # spawn appears to be safer with CUDA MPS

    from lightbench import available

    benchmarks = available()

    if mars:
        opt = ["mars-" + o for o in opt]
    if caution:
        opt = ["cautious-" + o for o in opt]
    if unscaled_caution:
        opt = ["unscaled_cautious-" + o for o in opt]

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    total_tasks = 0
    for name, o, i in itertools.product(benchmarks, opt, range(seeds)):
        if full_parallel:
            for d in difficulties:
                task_queue.put((f"{name}==={d}", o, steps, dtype, trials, i))
        else:
            task_queue.put((name, o, steps, dtype, trials, i))
        total_tasks += len(difficulties)

    if not total_tasks:
        raise ValueError("No benchmarks found")

    processes = []
    workers = min(parallelism, total_tasks)
    for idx in range(workers):
        p = multiprocessing.Process(
            target=worker, args=(task_queue, result_queue, idx, difficulties, timeout), daemon=False
        )
        p.start()
        processes.append(p)
        time.sleep(3)  # we can't start too many processes very quickly - otherwise there's errors with the cuda context

    # Collect results
    results = []
    completed = 0

    try:
        while workers > 0:
            prev_completed = completed
            while True:  # clear entire backlog and continue
                try:
                    result = result_queue.get(timeout=1)
                except Empty:
                    break
                if result is None:
                    workers -= 1
                    continue
                results.append(result)
                completed += 1
                print(
                    f"Progress: [{completed}/{total_tasks}] {result['name']} - {result['opt']}: "
                    f"{'✓' if result['success'] else '✗'}"
                )
            if prev_completed != completed:  # >= 1 task finished
                write_progress(results, opt, output)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    finally:
        # Clean up
        for p in processes:
            p.terminate()

        # Save final results
        if results:
            write_progress(results, opt, output)


if __name__ == "__main__":
    app()
