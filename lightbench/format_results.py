import os

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from lightbench.irt_ranking import fit, load_multi_results, load_results

app = typer.Typer(pretty_exceptions_enable=False)


def create_capability_matrix(model) -> pd.DataFrame:
    data = np.zeros((len(model.bench_names), len(model.challenger_names)), dtype=float)
    for i in range(len(model.bench_names)):
        for j in range(len(model.challenger_names)):
            data[i, j] = model.capability_score(j, i)
    single_dtype = len({dt for _, dt in model.challenger_names}) == 1
    if single_dtype:
        col_labels = [opt for opt, _dt in model.challenger_names]
    else:
        col_labels = [f"{opt} [{dt}]" for opt, dt in model.challenger_names]
    return pd.DataFrame(data, index=model.bench_names, columns=col_labels)


def create_visual_matrix(matrix: pd.DataFrame) -> plt.Figure | None:
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
    ax.set_xlabel("Challenger")
    ax.set_ylabel("Benchmark")
    ax.set_title("Capability Score (d* at P=0.5)")

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label("Difficulty threshold")
    plt.tight_layout()
    return fig


@app.command()
def heatmap(
    file: str = typer.Argument("benchmark_results.jsonl"),
):
    """Frequentist 2PL capability heatmap (single file)."""
    data = load_results(file)
    if data is None:
        print("No data loaded.")
        return
    model = fit(data)
    matrix = create_capability_matrix(model)
    fig = create_visual_matrix(matrix)
    if fig is None:
        print("Plot generation failed.")
        return
    plt.savefig("benchmark_matrix.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    print("Saved heatmap to: benchmark_matrix.png")
    plt.close(fig)


DEGENERATE_OPTS = {"OrthoLaProp", "LaPropOrtho", "SplitOpt",
                   "SOAPBase", "PSGDBase", "SAMWrapper"}


def _drop_optimizers(data, drop: set[str]):
    from lightbench.irt_ranking import ResultData
    keep = np.array([opt not in drop for opt, _ in data.challenger_names])
    if keep.all():
        return data
    new_names = [n for n, k in zip(data.challenger_names, keep) if k]
    old_to_new = {old: new for new, old in
                  enumerate(i for i, k in enumerate(keep) if k)}
    mask = np.isin(data.challenger_idx, list(old_to_new))
    new_idx = np.array([old_to_new[i] for i in data.challenger_idx[mask]],
                       dtype=data.challenger_idx.dtype)
    return ResultData(
        challenger_names=new_names,
        bench_names=data.bench_names,
        challenger_idx=new_idx,
        bench_idx=data.bench_idx[mask],
        difficulty=data.difficulty[mask],
        outcome=data.outcome[mask],
    )


@app.command()
def bayes(
    files: list[str] = typer.Argument(..., help="One or more benchmark_results.jsonl files"),
    K: int = typer.Option(1, help="Number of latent factors (CV-selected: 1)"),
    warmup: int = 4000,
    samples: int = 4000,
    chains: int = 16,
    target_accept: float = 0.95,
    seed: int = 0,
    keep_degenerate: bool = typer.Option(False, help="Include degenerate optimizers in the fit"),
    out_dir: str = "bayes_out",
):
    """Hierarchical Bayesian MIRT optimizer ranking — one shareable order.

    Primary outputs (the ranking): rating.csv, rating.png.
    Auxiliary (per-cell predictions, factor loadings/scores, raw posterior):
    figs/p_success_heatmap.png, figs/capability_heatmap.png, figs/*.png,
    posterior.npz.
    """
    from lightbench.bayes_mirt import fit_bayes
    from lightbench.bayes_rating import rating_table
    from lightbench.bayes_viz import (
        plot_capability_heatmap, plot_factor_loadings, plot_opt_factor_scores,
        plot_rating,
    )

    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
    data = load_multi_results(files) if len(files) > 1 else load_results(files[0])
    if data is None:
        typer.echo("No data loaded.")
        raise typer.Exit(1)
    if not keep_degenerate:
        data = _drop_optimizers(data, DEGENERATE_OPTS)
    typer.echo(f"opts={len(data.challenger_names)} benches={len(data.bench_names)} obs={len(data.outcome)}")

    post = fit_bayes(data, K=K, num_warmup=warmup, num_samples=samples,
                     num_chains=chains, target_accept=target_accept,
                     seed=seed, progress_bar=False)
    typer.echo(f"diagnostics: {post.diagnostics}")

    table = rating_table(post)
    table.to_csv(os.path.join(out_dir, "rating.csv"), index=False)
    pretty = table.copy()
    pretty["Theta"] = pretty["Theta"].map(lambda v: f"{v:+.3f}")
    typer.echo(pretty.to_string(index=False))

    fig = plot_rating(post)
    fig.savefig(os.path.join(out_dir, "rating.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.savez_compressed(
        os.path.join(out_dir, "posterior.npz"),
        K=np.array(post.K), theta=post.theta, phi=post.phi,
        gamma=post.gamma, b=post.b, c=post.c,
        log_alpha=post.log_alpha if post.log_alpha is not None else np.zeros(0),
        opts=np.array([f"{o}\t{d}" for o, d in post.challenger_names]),
        benches=np.array(post.bench_names),
    )

    for name, fig in [
        ("p_success_heatmap.png", plot_capability_heatmap(post, value="p_success", difficulty=0.5)),
        ("capability_heatmap.png", plot_capability_heatmap(post, value="capability")),
        ("factor_loadings.png", plot_factor_loadings(post)),
        ("factor_scores.png", plot_opt_factor_scores(post)),
    ]:
        if fig is None:
            continue
        fig.savefig(os.path.join(out_dir, "figs", name), dpi=200, bbox_inches="tight")
        plt.close(fig)
    typer.echo(f"Saved to {out_dir}/")


if __name__ == "__main__":
    app()
