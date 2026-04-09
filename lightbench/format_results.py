import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import typer
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from lightbench.utils import parse_opt_name

_OPT_RENAMES = {
    "CachedPSGDKron": "PSGDKron(cached=True)",
    "NewtonPSGDLRA": "PSGDLRA(hessian_approx=True, hvp_interval=10000000)",
}


def split_benchmark_name(name):
    if "-" not in name:
        return name, None
    return name.rsplit("-", 1)


def normalize_optimizer_name(name):
    if name.startswith("Foreach") and len(name) > len("Foreach"):
        name = name[len("Foreach"):]
    return _OPT_RENAMES.get(name, name)


def _opt_flags(opt_str):
    name, kwargs = parse_opt_name(opt_str)
    caution = "Unscaled" if kwargs.get("_unscale_caution") else ("Yes" if kwargs.get("caution") else "No")
    mars = "Yes" if kwargs.get("mars") else "No"
    return normalize_optimizer_name(name), caution, mars


def compose_optimizer(storage, caution, mars, optimizer):
    parts = []
    if storage:
        parts.append(storage)
    if caution == "Yes":
        parts.append("cautious")
    elif caution and caution != "No":
        parts.append(caution)
    if mars == "Yes":
        parts.append("mars")
    elif mars and mars != "No":
        parts.append(mars)
    if optimizer:
        parts.append(optimizer)
    return "-".join(parts)


def read_benchmark_results(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            name, caution, mars = _opt_flags(r["opt"])
            storage = r.get("dtype", "fp32")
            loss = r.get("loss")
            data.append({
                "benchmark": f"{r['benchmark']}-{r['difficulty']}",
                "optimizer": compose_optimizer(storage, caution, mars, name),
                "success": bool(r.get("success")),
                "runtime": float(r.get("runtime", 0)),
                "loss": float(loss) if loss is not None else float("nan"),
                "attempts": int(r.get("attempts", 0)),
            })
    return pd.DataFrame(data)


def create_success_matrix(df):
    if df.empty:
        return pd.DataFrame()
    benchmark_rates = df.groupby(["benchmark", "optimizer"], sort=True)["success"].mean().unstack()
    benchmark_rates["_base_task"] = [split_benchmark_name(name)[0] for name in benchmark_rates.index]
    success_total_matrix = benchmark_rates.groupby("_base_task", sort=True).mean()
    return success_total_matrix.sort_index()


def normalize_matrix_by_row_max(matrix):
    max_in_row = matrix.max(axis=1, skipna=True).fillna(1)
    max_in_row[max_in_row == 0] = 1
    return matrix.div(max_in_row, axis=0) * 100


def create_visual_matrix_normalized(success_total_matrix):
    if success_total_matrix.empty:
        return None

    def rank_nonzero(row):
        values = row.to_numpy(dtype=float)
        ranked = np.full_like(values, np.nan, dtype=float)
        valid = np.isfinite(values)
        nz = valid & (values > 0)
        ranked[valid & ~nz] = 0.0
        if nz.any():
            ranked[nz] = scipy.stats.rankdata(values[nz], method="dense")
        return pd.Series(ranked, index=row.index)

    ranked_matrix = success_total_matrix.apply(rank_nonzero, axis=1)
    normalized_matrix = normalize_matrix_by_row_max(ranked_matrix)
    optimizer_means = normalized_matrix.mean(axis=0, skipna=True)
    task_means = normalized_matrix.mean(axis=1, skipna=True)
    overall_mean = optimizer_means.mean(skipna=True)
    plot_matrix: pd.DataFrame = normalized_matrix.copy()
    plot_matrix.loc["Avg. Optimizer"] = optimizer_means

    # weight of 0.5, as "jack of all trades, master of none" is better than "perfect at xor but awful at delay"
    optimizer_score = (normalized_matrix**0.5).mean(axis=0, skipna=True)
    optimizer_indices = np.argsort(-optimizer_score.fillna(-np.inf).to_numpy())
    plot_matrix = plot_matrix.iloc[:, optimizer_indices]

    full_task_means = pd.concat([task_means, pd.Series([overall_mean], index=["Avg. Optimizer"])])
    plot_matrix["Avg. Task"] = full_task_means
    plot_tasks = plot_matrix.index
    plot_optimizers = plot_matrix.columns

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(
        figsize=(max(14, len(plot_optimizers) * 0.8), max(10, len(plot_tasks) * 0.6)), facecolor="white"
    )
    cmap = plt.cm.Blues
    norm = Normalize(vmin=0, vmax=100)
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    for i, task in enumerate(plot_tasks):
        for j, optimizer in enumerate(plot_optimizers):
            value = plot_matrix.loc[task, optimizer]
            original_count = np.nan
            if task != "Avg. Optimizer" and optimizer != "Avg. Task":
                if task in success_total_matrix.index and optimizer in success_total_matrix.columns:
                    original_count = success_total_matrix.loc[task, optimizer]
            is_summary = task == "Avg. Optimizer" or optimizer == "Avg. Task"
            if pd.isna(value):
                face_color = (0.93, 0.93, 0.93, 0.7)
            elif not is_summary and np.isfinite(original_count) and original_count == 0:
                face_color = (0.85, 0.85, 0.85, 0.7)
            else:
                face_color = mapper.to_rgba(value)
            edge_color = "#666666" if is_summary else "#AAAAAA"
            edge_width = 1.5 if is_summary else 0.5
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face_color, edgecolor=edge_color, linewidth=edge_width)
            ax.add_patch(rect)

    ax.axhline(len(plot_tasks) - 1.5, color="#333333", linewidth=2, linestyle="-")
    ax.axvline(len(plot_optimizers) - 1.5, color="#333333", linewidth=2, linestyle="-")
    ax.set_xticks(np.arange(len(plot_optimizers)))
    ax.set_yticks(np.arange(len(plot_tasks)))
    ax.set_xticklabels(plot_optimizers, rotation=45, ha="right", fontsize=14, fontweight="bold", color="#333333")
    ax.set_yticklabels(plot_tasks, fontsize=14, fontweight="bold", color="#333333")
    ax.set_xlabel("Optimizer", fontsize=16, fontweight="bold", labelpad=15, color="#333333")
    ax.set_ylabel("Task", fontsize=16, fontweight="bold", labelpad=15, color="#333333")
    ax.set_title(
        "Success Rate by Task and Optimizer\n(100% = Best Performance for Each Task)",
        fontsize=18,
        fontweight="bold",
        pad=20,
        color="#333333",
    )

    cbar = fig.colorbar(mapper, ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label("Success Rate (%)", rotation=270, labelpad=20, fontsize=14, fontweight="bold", color="#333333")
    cbar.ax.tick_params(labelsize=12, color="#333333", labelcolor="#333333")

    ax.set_facecolor("#F5F5F5")
    fig.patch.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.5)

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    return fig


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(file: str = typer.Argument("benchmark_results.jsonl")):
    df = read_benchmark_results(file)
    if df.empty:
        print("No data loaded.")
        return
    success_total_matrix = create_success_matrix(df)
    if success_total_matrix.empty:
        print("No successful runs found.")
        return
    fig = create_visual_matrix_normalized(success_total_matrix)
    if fig is None:
        print("Plot generation failed.")
        return
    plt.savefig("benchmark_matrix.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    print("Saved heatmap to: benchmark_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    app()
