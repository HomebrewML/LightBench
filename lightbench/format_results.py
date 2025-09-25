import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import typer
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


def parse_loss(loss_str):
    loss_str = loss_str.strip()
    if not loss_str or loss_str.lower() == "nan":
        return float("nan")
    if loss_str.lower() == "inf":
        return float("inf")
    try:
        return (
            float(loss_str)
            if "e" not in loss_str
            else float(loss_str.split("e")[0]) * 10 ** float(loss_str.split("e")[1])
        )
    except ValueError:
        return float("nan")


def process_str(x, truthy):
    if x == "No":
        return ""
    if x == "Yes":
        return f"{truthy}-"
    return f"{x}-"


def read_benchmark_results(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    details = re.search(r"## Details\n\n(.*?)(?=\n##|\n\Z)", content, re.DOTALL | re.IGNORECASE)
    if not details:
        raise ValueError("Details section not found.")
    table = details.group(1).strip()
    lines = re.search(r"\|:?-+:?(.*?)\|\n(.*)", table, re.DOTALL).group(2).strip().split("\n")
    data = []
    for i, line in enumerate(lines):
        if not line.strip() or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 8:
            continue
        try:
            caution = process_str(parts[2], "cautious")
            mars = process_str(parts[3], "mars")
            optimizer = f"{caution}{mars}{parts[1]}"
            optimizer = optimizer.replace("Foreach", "").replace("Cached", "").strip()
            data.append({
                "benchmark": parts[0],
                "optimizer": optimizer,
                "success": parts[4] == "✓",
                "runtime": float(parts[5].replace("s", "")) if parts[5] else float("nan"),
                "loss": parse_loss(parts[6]) if parts[6] else float("nan"),
                "attempts": int(parts[7]) if parts[7].isdigit() else 0,
            })
        except (IndexError, ValueError):
            continue
    return pd.DataFrame(data)


def create_success_matrix(df):
    if df.empty:
        return pd.DataFrame()
    benchmarks = sorted(df["benchmark"].unique())
    optimizers = sorted(df["optimizer"].unique())
    success_matrix = pd.DataFrame(0, index=benchmarks, columns=optimizers, dtype=int)
    for _, row in df.iterrows():
        if row["success"] and row["benchmark"] in success_matrix.index and row["optimizer"] in success_matrix.columns:
            success_matrix.loc[row["benchmark"], row["optimizer"]] = 1
    base_tasks = sorted(set(b.split("-")[0] for b in benchmarks))
    success_total_matrix = pd.DataFrame(0, index=base_tasks, columns=optimizers, dtype=int)
    for benchmark in success_matrix.index:
        base_task = benchmark.split("-")[0]
        if base_task in success_total_matrix.index:
            success_total_matrix.loc[base_task] += success_matrix.loc[benchmark]
    return success_total_matrix


def normalize_matrix_by_row_max(matrix):
    max_in_row = matrix.max(axis=1)
    max_in_row[max_in_row == 0] = 1
    return matrix.div(max_in_row, axis=0) * 100


def create_visual_matrix_normalized(success_total_matrix):
    if success_total_matrix.empty:
        return None
    tasks_to_keep = success_total_matrix.sum(axis=1) > 0
    filtered_matrix = success_total_matrix[tasks_to_keep].copy()
    if filtered_matrix.empty:
        return None
    filtered_matrix[:] = scipy.stats.rankdata(filtered_matrix, axis=1, method="dense")
    normalized_matrix = normalize_matrix_by_row_max(filtered_matrix)
    optimizer_means = normalized_matrix.mean(axis=0)
    task_means = normalized_matrix.mean(axis=1)
    overall_mean = optimizer_means.mean()
    plot_matrix: pd.DataFrame = normalized_matrix.copy()
    plot_matrix.loc["Avg. Optimizer"] = optimizer_means

    # weight of 0.5, as "jack of all trades, master of none" is better than "perfect at xor but awful at delay"
    optimizer_score = (normalized_matrix**0.5).mean(axis=0)
    optimizer_indices = np.argsort(-optimizer_score.to_numpy())
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
            original_count = 0
            if task != "Avg. Optimizer" and optimizer != "Avg. Task":
                if task in success_total_matrix.index and optimizer in success_total_matrix.columns:
                    original_count = success_total_matrix.loc[task, optimizer]
            is_summary = task == "Avg. Optimizer" or optimizer == "Avg. Task"
            face_color = (0.85, 0.85, 0.85, 0.7) if not is_summary and original_count == 0 else mapper.to_rgba(value)
            edge_color = "#666666" if is_summary else "#AAAAAA"
            edge_width = 1.5 if is_summary else 0.5
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face_color, edgecolor=edge_color, linewidth=edge_width)
            ax.add_patch(rect)
            # brightness = sum(face_color[:3]) * 0.333
            # text_color = 'white' if brightness < 0.6 else 'black'
            # ax.text(j, i, f"{value:.0f}", ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')

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
def main(file: str = typer.Argument("benchmark_results.md")):
    df = read_benchmark_results(file)
    if df.empty:
        print("No data loaded from benchmark file.")
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
