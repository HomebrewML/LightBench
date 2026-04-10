import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from lightbench.irt_ranking import fit, load_results

app = typer.Typer(pretty_exceptions_enable=False)


def create_capability_matrix(model, dtype_name: str | None = None) -> tuple[pd.DataFrame, str]:
    if not model.dtype_names:
        return pd.DataFrame(), ""
    if dtype_name is None:
        dtype_name = "fp32" if "fp32" in model.dtype_names else model.dtype_names[0]
    if dtype_name not in model.dtype_names:
        dtype_name = model.dtype_names[0]
    dtype_idx = model.dtype_names.index(dtype_name)

    data = np.zeros((len(model.bench_names), len(model.opt_names)), dtype=float)
    for i, _bench in enumerate(model.bench_names):
        for j, _opt in enumerate(model.opt_names):
            data[i, j] = model.capability_score(j, i, dtype_idx)
    matrix = pd.DataFrame(data, index=model.bench_names, columns=model.opt_names)
    return matrix, dtype_name


def create_visual_matrix(matrix: pd.DataFrame, dtype_name: str) -> plt.Figure | None:
    if matrix.empty:
        return None
    col_order = matrix.mean(axis=0, skipna=True).sort_values(ascending=False).index
    row_order = matrix.mean(axis=1, skipna=True).sort_values(ascending=False).index
    matrix = matrix.loc[row_order, col_order]
    values = matrix.to_numpy(dtype=float)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    fig, ax = plt.subplots(
        figsize=(max(12, len(col_order) * 0.6), max(8, len(row_order) * 0.4)),
        facecolor="white",
    )
    cmap = plt.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(col_order)))
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(row_order, fontsize=10)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Benchmark")
    ax.set_title(f"Capability Score (d* at P=0.5, dtype={dtype_name})")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label("Difficulty threshold")
    plt.tight_layout()
    return fig


@app.command()
def main(
    file: str = typer.Argument("benchmark_results.jsonl"),
    dtype: str | None = typer.Option(None, help="Dtype to score (defaults to fp32 if available)"),
):
    data = load_results(file)
    if data is None:
        print("No data loaded.")
        return
    model = fit(data)
    matrix, dtype_name = create_capability_matrix(model, dtype)
    fig = create_visual_matrix(matrix, dtype_name)
    if fig is None:
        print("Plot generation failed.")
        return
    plt.savefig("benchmark_matrix.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    print("Saved heatmap to: benchmark_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    app()
