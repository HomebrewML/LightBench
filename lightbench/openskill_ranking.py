import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openskill.models import PlackettLuce
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lightbench.utils import parse_opt_name

_DIFFICULTIES = ("trivial", "easy", "medium", "hard", "extreme", "nightmare")
_DIFF_IDX = {d: i for i, d in enumerate(_DIFFICULTIES)}
_NDIFF = len(_DIFFICULTIES)
_DTYPE_EDGES = (("bf16+ecc", "fp32"), ("bf16", "bf16+ecc"), ("fp16+ecc", "fp32"), ("fp16", "fp16+ecc"))
_DTYPE_PREREQ = {child: parent for child, parent in _DTYPE_EDGES}
_DTYPE_ORDER = {"fp32": 0, "bf16+ecc": 1, "bf16": 2, "fp16+ecc": 3, "fp16": 4}
_OPT_RENAMES = {
    "CachedPSGDKron": "PSGDKron(cached=True)",
    "NewtonPSGDLRA": "PSGDLRA(hessian_approx=True, hvp_interval=10000000)",
}
_DTYPE_STYLES = {"fp32": "bright_cyan", "bf16": "cyan", "bf16+ecc": "magenta", "fp16": "green", "fp16+ecc": "yellow"}


@dataclass(frozen=True)
class FrontierBuild:
    games: dict[str, dict[str, int]]
    labels: list[str]
    unresolved_by_label: dict[str, int]
    resolved_cells: int
    unresolved_cells: int
    implied_failures: int


def _dtype_key(dtype: str) -> tuple[int, str]:
    return _DTYPE_ORDER.get(dtype, len(_DTYPE_ORDER)), dtype


def normalize_optimizer_name(name: str) -> str:
    if name.startswith("Foreach") and name != "Foreach":
        name = name[len("Foreach"):]
    return _OPT_RENAMES.get(name, name)


def _parse_opt_fields(opt) -> tuple[str, str, str] | None:
    try:
        name, kwargs = parse_opt_name(opt)
    except Exception:
        return None
    caution = "Unscaled" if kwargs.get("_unscale_caution") else ("Yes" if kwargs.get("caution") else "No")
    mars = "Yes" if kwargs.get("mars") else "No"
    return normalize_optimizer_name(name), caution, mars


def load_results(path: str) -> pd.DataFrame:
    cols = ["benchmark", "difficulty", "dtype", "label", "suffix", "state"]
    try:
        raw = pd.read_json(path, lines=True)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame(columns=cols)
    if raw.empty:
        return pd.DataFrame(columns=cols)
    if "benchmark" not in raw or "difficulty" not in raw or "opt" not in raw:
        return pd.DataFrame(columns=cols)

    df = raw.copy()
    df["benchmark"] = df["benchmark"].astype(str)
    df["difficulty"] = df["difficulty"].astype(str)
    df["dtype"] = df["dtype"].astype(str) if "dtype" in df else "unknown"
    df["dtype"] = df["dtype"].fillna("unknown").astype(str)

    parsed = df["opt"].map(_parse_opt_fields)
    df = df.loc[parsed.notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    parsed = pd.DataFrame(parsed.loc[df.index].tolist(), columns=["optimizer", "caution", "mars"], index=df.index)
    df = pd.concat([df, parsed], axis=1)

    caution_tag = np.where(
        df["caution"].eq("Yes"),
        "cautious-",
        np.where(df["caution"].ne("No") & df["caution"].ne(""), df["caution"] + "-", ""),
    )
    mars_tag = np.where(
        df["mars"].eq("Yes"),
        "mars-",
        np.where(df["mars"].ne("No") & df["mars"].ne(""), df["mars"] + "-", ""),
    )
    df["label"] = df["dtype"] + ":" + caution_tag + mars_tag + df["optimizer"]
    df["suffix"] = df["label"].str.split(":", n=1).str[1]

    success = df["success"].astype(bool) if "success" in df else pd.Series(False, index=df.index)
    timed_out = df["timed_out"].astype(bool) if "timed_out" in df else pd.Series(False, index=df.index)
    error = df["error"] if "error" in df else pd.Series("", index=df.index)
    error = error.fillna("").astype(str).str.strip()
    gated_fail = error.eq("") | error.str.startswith(("Skipped", "Gated"))
    df["state"] = np.select([success, timed_out, gated_fail], [1, -1, -1], default=0).astype(np.int8)

    return (
        df[cols]
        .drop_duplicates(["benchmark", "difficulty", "label"], keep="last")
        .reset_index(drop=True)
    )


def _close_failures(states: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    observed_failures = states == -1
    fails = np.maximum.accumulate(observed_failures, axis=1)
    while True:
        prev = fails.copy()
        for parent_i, child_i in pairs:
            fails[child_i] |= fails[parent_i]
        fails = np.maximum.accumulate(fails, axis=1)
        if np.array_equal(prev, fails):
            break
    return fails, int((fails & ~observed_failures).sum())


def compute_frontiers(df: pd.DataFrame) -> FrontierBuild:
    if df.empty:
        return FrontierBuild({}, [], {}, 0, 0, 0)

    work = df[df["difficulty"].isin(_DIFF_IDX)].copy()
    if work.empty:
        return FrontierBuild({}, [], {}, 0, 0, 0)
    work["difficulty_idx"] = work["difficulty"].map(_DIFF_IDX).astype(int)

    labels = sorted(work["label"].unique().tolist(), key=lambda l: (l.split(":", 1)[1], _dtype_key(l.split(":", 1)[0]), l))
    games: dict[str, dict[str, int]] = {}
    unresolved: dict[str, int] = {}
    implied_failures = 0
    contradictions: list[str] = []
    cols = np.arange(_NDIFF)

    for (benchmark, suffix), group in work.groupby(["benchmark", "suffix"], sort=True):
        dtypes = sorted(group["dtype"].unique().tolist(), key=_dtype_key)
        dt_idx = {d: i for i, d in enumerate(dtypes)}
        states = (
            group.pivot(index="dtype", columns="difficulty_idx", values="state")
            .reindex(index=dtypes, columns=range(_NDIFF), fill_value=0)
            .fillna(0)
            .astype(np.int8)
            .to_numpy()
        )

        pairs = [(dt_idx[parent], dt_idx[child]) for child, parent in _DTYPE_PREREQ.items() if child in dt_idx and parent in dt_idx]
        fails, implied = _close_failures(states, pairs)
        implied_failures += implied
        passes = states == 1

        if np.any(fails & passes):
            i, j = np.argwhere(fails & passes)[0]
            contradictions.append(f"{benchmark}|{dtypes[i]}-{_DIFFICULTIES[j]}")
            continue

        lo = np.where(passes, cols, -1).max(axis=1)
        hi = np.where(fails, cols, _NDIFF).min(axis=1) - 1
        for dtype, i in dt_idx.items():
            label = f"{dtype}:{suffix}"
            if lo[i] > hi[i]:
                contradictions.append(f"{benchmark}|{label}[{int(lo[i])},{int(hi[i])}]")
            elif lo[i] == hi[i]:
                games.setdefault(benchmark, {})[label] = int(lo[i])
            else:
                unresolved[label] = unresolved.get(label, 0) + 1

    if contradictions:
        sample = "; ".join(contradictions[:5])
        raise ValueError(f"Inconsistent ranking constraints in {len(contradictions)} groups. Examples: {sample}")

    resolved_cells = sum(len(v) for v in games.values())
    unresolved_cells = sum(unresolved.values())
    return FrontierBuild(games=games, labels=labels, unresolved_by_label=unresolved, resolved_cells=resolved_cells,
                         unresolved_cells=unresolved_cells, implied_failures=implied_failures)


def frontier_ranks(frontiers: dict[str, int]) -> tuple[list[str], list[int]]:
    ordered = sorted(frontiers.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [label for label, _ in ordered]
    ranks, prev, rank = [], None, 0
    for i, (_, frontier) in enumerate(ordered, 1):
        if frontier != prev:
            rank, prev = i, frontier
        ranks.append(rank)
    return labels, ranks


def _resolved_frame(build: FrontierBuild) -> pd.DataFrame:
    rows = [
        (benchmark, label, frontier)
        for benchmark, per_benchmark in build.games.items()
        for label, frontier in per_benchmark.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["benchmark", "label", "frontier", "dtype", "suffix"])
    resolved = pd.DataFrame(rows, columns=["benchmark", "label", "frontier"])
    parts = resolved["label"].str.split(":", n=1, expand=True)
    resolved["dtype"] = parts[0]
    resolved["suffix"] = parts[1]
    return resolved


def _suffix_penalties(suffix_df: pd.DataFrame, suffix: str) -> dict[str, float]:
    per_dtype = {dtype: g.set_index("benchmark")["frontier"].astype(float) for dtype, g in suffix_df.groupby("dtype", sort=False)}
    penalties = {dtype: 0.0 for dtype in per_dtype}
    for child, parent in _DTYPE_EDGES:
        if child not in per_dtype or parent not in per_dtype:
            continue
        joined = pd.DataFrame({"parent": per_dtype[parent], "child": per_dtype[child]}).dropna()
        if joined.empty:
            penalties[child] = penalties[parent]
            continue
        edge = float((joined["child"] - joined["parent"]).mean())
        if edge > 1e-9:
            raise ValueError(f"Constraint violation in {suffix}: {parent}->{child} edge={edge:.4f}")
        penalties[child] = penalties[parent] + edge
    return penalties


def rank_variants(build: FrontierBuild) -> tuple[pd.DataFrame, int, int]:
    model = PlackettLuce()
    suffixes = sorted({label.split(":", 1)[1] for label in build.labels})
    suffix_ratings = {suffix: model.rating(name=suffix) for suffix in suffixes}
    resolved = _resolved_frame(build)
    by_label_bench = {
        label: g.set_index("benchmark")["frontier"].to_dict()
        for label, g in resolved.groupby("label", sort=False)
    }
    penalties_by_suffix = {
        suffix: _suffix_penalties(g[["benchmark", "dtype", "frontier"]], suffix)
        for suffix, g in resolved.groupby("suffix", sort=False)
    }
    played: dict[str, int] = {}
    skipped_singletons = 0
    skipped_ties = 0

    for benchmark in sorted(build.games):
        suffix_game = {
            suffix: frontier
            for label, frontier in build.games[benchmark].items()
            for dtype, suffix in [label.split(":", 1)]
            if dtype == "fp32"
        }
        if len(suffix_game) < 2:
            skipped_singletons += 1
            continue
        game_labels, ranks = frontier_ranks(suffix_game)
        if len(set(ranks)) < 2:
            skipped_ties += 1
            continue
        for suffix, updated in zip(game_labels, model.rate([[suffix_ratings[suffix]] for suffix in game_labels], ranks=ranks)):
            suffix_ratings[suffix] = updated[0]
            played[suffix] = played.get(suffix, 0) + 1

    x, y = [], []
    for suffix in suffixes:
        values = list(by_label_bench.get(f"fp32:{suffix}", {}).values())
        if values:
            x.append(float(np.mean(values)))
            y.append(float(suffix_ratings[suffix].ordinal()))
    if len(x) >= 2 and np.var(x) > 1e-12:
        slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    else:
        slope = 0.0
    if slope < -1e-12:
        raise ValueError(f"Invalid frontier->rating slope ({slope:.6f}); suffix ranking is inconsistent")
    slope = max(0.0, slope)

    rows = []
    for label in build.labels:
        dtype = label.split(":", 1)[0]
        suffix = label.split(":", 1)[1]
        fs_map = by_label_bench.get(label, {})
        fs = list(fs_map.values())
        suffix_rating = suffix_ratings[suffix]
        penalty = penalties_by_suffix.get(suffix, {}).get(dtype, 0.0)
        mu = suffix_rating.mu + slope * penalty
        sigma = suffix_rating.sigma
        rows.append({
            "Rank": 0,
            "Variant": label,
            "Dtype": dtype,
            "Games": played.get(suffix, 0),
            "Benchmarks": len(fs_map),
            "Unresolved": build.unresolved_by_label.get(label, 0),
            "Mean Frontier": float(np.mean(fs)) if fs else 0.0,
            "Mu": mu,
            "Sigma": sigma,
            "Ordinal": mu - 3.0 * sigma,
        })
    rank_df = pd.DataFrame(rows).sort_values(["Ordinal", "Mu", "Sigma"], ascending=[False, False, True], kind="mergesort")
    rank_df["Rank"] = np.arange(1, len(rank_df) + 1)

    ord_by_suffix: dict[str, dict[str, float]] = {}
    for row in rank_df.to_dict("records"):
        dt, suffix = row["Variant"].split(":", 1)
        ord_by_suffix.setdefault(suffix, {})[dt] = float(row["Ordinal"])
    violations = []
    for suffix, vals in ord_by_suffix.items():
        for child, parent in _DTYPE_EDGES:
            if child in vals and parent in vals and vals[parent] + 1e-12 < vals[child]:
                violations.append((suffix, parent, child, vals[parent], vals[child]))
    if violations:
        s, p, c, pv, cv = violations[0]
        raise ValueError(f"Post-rank dtype violation: {s} {p}<{c} ({pv:.4f}<{cv:.4f})")

    return rank_df, skipped_singletons, skipped_ties


def _print_legend(console: Console) -> None:
    legend = Table(title="Dtype Legend")
    legend.add_column("Color")
    legend.add_column("Dtype")
    for dtype, style in _DTYPE_STYLES.items():
        legend.add_row(Text(dtype, style=style), dtype)
    console.print(legend)


def _print_ranking(console: Console, rank_df: pd.DataFrame) -> None:
    table = Table(title="OpenSkill Ranking")
    table.add_column("#", justify="right")
    table.add_column("Variant")
    table.add_column("Dtype")
    table.add_column("Games", justify="right")
    table.add_column("Benchmarks", justify="right")
    table.add_column("Unresolved", justify="right")
    table.add_column("Mean Frontier", justify="right")
    table.add_column("Mu", justify="right")
    table.add_column("Sigma", justify="right")
    table.add_column("Ordinal", justify="right")
    for row in rank_df.to_dict("records"):
        style = _DTYPE_STYLES.get(str(row["Dtype"]), "")
        table.add_row(
            str(row["Rank"]),
            Text(str(row["Variant"]), style=style),
            str(row["Dtype"]),
            str(row["Games"]),
            str(row["Benchmarks"]),
            str(row["Unresolved"]),
            f"{row['Mean Frontier']:.2f}",
            f"{row['Mu']:.4f}",
            f"{row['Sigma']:.4f}",
            f"{row['Ordinal']:.4f}",
        )
    console.print(table)


def main() -> None:
    p = argparse.ArgumentParser(description="Rank benchmark variants using OpenSkill.")
    p.add_argument("file", nargs="?", default="benchmark_results.jsonl")
    p.add_argument("--dtype", nargs="+", help="Only include these dtypes (e.g. --dtype fp32 bf16)")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    console = Console(no_color=args.no_color)
    df = load_results(args.file)
    if args.dtype:
        df = df[df["dtype"].isin(args.dtype)].reset_index(drop=True)
    build = compute_frontiers(df)
    rank_df, skipped_singletons, skipped_ties = rank_variants(build)

    console.print(f"File: {args.file}")
    console.print("Backend: OpenSkill Plackett-Luce (fail-closure frontier per benchmark)")
    console.print(
        f"Entries: {len(df)} | Variants: {len(build.labels)} | Games: {len(build.games)} | "
        f"Resolved: {build.resolved_cells} | Unresolved: {build.unresolved_cells} | "
        f"Implied fails: {build.implied_failures} | "
        f"Skipped: {skipped_singletons} single-variant, {skipped_ties} all-tied"
    )
    console.print()
    _print_legend(console)
    console.print()
    _print_ranking(console, rank_df)


if __name__ == "__main__":
    main()
