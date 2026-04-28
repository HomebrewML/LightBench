import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openskill.models import PlackettLuce
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lightbench.utils import _DTYPE_PREREQ, _dtype_key, _parse_opt_fields

_DIFFICULTIES = ("trivial", "easy", "medium", "hard", "extreme", "nightmare")
_DIFF_IDX = {d: i for i, d in enumerate(_DIFFICULTIES)}
_NDIFF = len(_DIFFICULTIES)
_DTYPE_STYLES = {"fp32": "bright_cyan", "bf16": "cyan", "bf16+ecc": "magenta", "fp16": "green", "fp16+ecc": "yellow"}

# Transitive ancestors per dtype. Allows direct propagation even when
# intermediate dtypes are absent from a particular matrix.
_DTYPE_ANCESTORS: dict[str, tuple[str, ...]] = {}
for _dt in ("fp32", "bf16+ecc", "bf16", "fp16+ecc", "fp16"):
    _anc, _cur = [], _dt
    while _cur in _DTYPE_PREREQ:
        _cur = _DTYPE_PREREQ[_cur]
        _anc.append(_cur)
    _DTYPE_ANCESTORS[_dt] = tuple(_anc)


@dataclass
class FilledMatrix:
    dtypes: list[str]
    cells: np.ndarray  # (len(dtypes), _NDIFF), values in {-1, 0, 1}
    difficulties: tuple = _DIFFICULTIES


def _indexed_difficulties(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple]:
    if df.empty:
        return df.copy(), ()
    difficulty = df["difficulty"]
    cat = difficulty.isin(_DIFF_IDX)
    numeric = pd.to_numeric(difficulty, errors="coerce")
    num = np.isfinite(numeric.to_numpy())

    if cat.any() and not num.any():
        work = df.loc[cat].copy()
        work["diff_idx"] = work["difficulty"].map(_DIFF_IDX).astype(int)
        return work, _DIFFICULTIES

    if num.any() and not cat.any():
        work = df.loc[num].copy()
        values = tuple(float(x) for x in np.sort(numeric.loc[work.index].unique()))
        diff_idx = {d: i for i, d in enumerate(values)}
        work["diff_idx"] = numeric.loc[work.index].map(diff_idx).astype(int)
        return work, values

    values = np.full(len(df), np.nan, dtype=float)
    for label, idx in _DIFF_IDX.items():
        values[difficulty.eq(label).to_numpy()] = idx / max(_NDIFF - 1, 1)
    values[num] = numeric.to_numpy()[num]
    keep = np.isfinite(values)
    work = df.loc[keep].copy()
    levels = tuple(float(x) for x in np.unique(values[keep]))
    diff_idx = {d: i for i, d in enumerate(levels)}
    work["diff_idx"] = [diff_idx[float(v)] for v in values[keep]]
    return work, levels


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
    definite_fail = error.eq("") | error.str.startswith(("Skipped", "Gated"))
    df["state"] = np.select([success, timed_out, definite_fail], [1, -1, -1], default=0).astype(np.int8)

    return (
        df[cols]
        .drop_duplicates(["benchmark", "difficulty", "label"], keep="last")
        .reset_index(drop=True)
    )


def _fill_matrix(states: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Fill a (n_dtypes, n_diff) matrix of {-1: fail, 0: unknown, 1: pass}.

    Phase 1 — close failures (hard constraints, iterates to fixpoint):
      Dtype hierarchy:  ancestor fails at j  ->  descendant fails at j
      Difficulty mono:   fail at j            ->  fail at j+1 .. n-1
      Hierarchy overrides contradictory passes.

    Phase 2 — fill passes (never overwrites failures, iterates to fixpoint):
      Dtype hierarchy:  descendant passes at j  ->  ancestor passes at j
        (if the harder dtype passed, the easier one must have too)
      Difficulty mono:   pass at j  ->  pass at 0 .. j-1
    """
    m = states.copy()
    n_dt, _ = m.shape

    # Phase 1: close failures, overriding contradictory passes
    while True:
        prev = m.copy()
        for parent_i, child_i in pairs:
            mask = m[parent_i] == -1
            m[child_i] = np.where(mask, -1, m[child_i])
        fails = m == -1
        fails = np.maximum.accumulate(fails, axis=1)
        m = np.where(fails, -1, m)
        if np.array_equal(prev, m):
            break

    # Phase 2: fill passes (upward through hierarchy + leftward)
    while True:
        prev = m.copy()
        # Child passes -> parent passes (only into unknown cells)
        for parent_i, child_i in pairs:
            mask = (m[child_i] == 1) & (m[parent_i] == 0)
            m[parent_i] = np.where(mask, 1, m[parent_i])
        # Pass at j -> pass at 0..j-1 (only into unknown cells)
        for i in range(n_dt):
            pass_idx = np.where(m[i] == 1)[0]
            if len(pass_idx) == 0:
                continue
            max_pass = int(pass_idx.max())
            for j in range(max_pass):
                if m[i, j] == 0:
                    m[i, j] = 1
        if np.array_equal(prev, m):
            break

    return m


def _hierarchy_pairs(dtypes: list[str]) -> list[tuple[int, int]]:
    """Build (parent_idx, child_idx) pairs for the given dtype list."""
    dt_idx = {d: i for i, d in enumerate(dtypes)}
    pairs: list[tuple[int, int]] = []
    for dt in dtypes:
        for anc in _DTYPE_ANCESTORS.get(dt, ()):
            if anc in dt_idx:
                pair = (dt_idx[anc], dt_idx[dt])
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


def build_matrices(df: pd.DataFrame) -> dict[str, dict[str, FilledMatrix]]:
    """Build filled (n_dtypes, n_diff) Pareto matrices keyed by [benchmark][suffix].

    After building per-(benchmark, suffix) matrices from actual data, extends
    each matrix so that every dtype that appears for a suffix in ANY benchmark
    is present in ALL benchmarks for that suffix.  The fill then propagates
    implied results (failures down, passes up) into the new rows.
    """
    work, levels = _indexed_difficulties(df)
    if work.empty:
        return {}
    n_diff = len(levels)

    result: dict[str, dict[str, FilledMatrix]] = {}
    for (benchmark, suffix), group in work.groupby(["benchmark", "suffix"], sort=True):
        dtypes = sorted(group["dtype"].unique().tolist(), key=_dtype_key)
        states = (
            group.pivot(index="dtype", columns="diff_idx", values="state")
            .reindex(index=dtypes, columns=range(n_diff), fill_value=0)
            .fillna(0)
            .astype(np.int8)
            .to_numpy()
        )
        result.setdefault(benchmark, {})[suffix] = FilledMatrix(
            dtypes=dtypes, cells=_fill_matrix(states, _hierarchy_pairs(dtypes)), difficulties=levels,
        )

    # Extend: ensure each suffix has ALL its dtypes in EVERY benchmark.
    # Removes selection bias from partial dtype coverage across runs.
    suffix_all_dtypes: dict[str, list[str]] = {}
    for by_suffix in result.values():
        for suffix, fm in by_suffix.items():
            seen = suffix_all_dtypes.get(suffix)
            if seen is None:
                suffix_all_dtypes[suffix] = list(fm.dtypes)
            else:
                for dt in fm.dtypes:
                    if dt not in seen:
                        seen.append(dt)
    for suffix, dtypes in suffix_all_dtypes.items():
        suffix_all_dtypes[suffix] = sorted(dtypes, key=_dtype_key)

    for by_suffix in result.values():
        for suffix in list(by_suffix):
            fm = by_suffix[suffix]
            all_dtypes = suffix_all_dtypes[suffix]
            if fm.dtypes == all_dtypes:
                continue
            old_idx = {d: i for i, d in enumerate(fm.dtypes)}
            new_cells = np.zeros((len(all_dtypes), n_diff), dtype=np.int8)
            for i, dt in enumerate(all_dtypes):
                if dt in old_idx:
                    new_cells[i] = fm.cells[old_idx[dt]]
            by_suffix[suffix] = FilledMatrix(
                dtypes=all_dtypes,
                cells=_fill_matrix(new_cells, _hierarchy_pairs(all_dtypes)),
                difficulties=levels,
            )

    return result


def _build_games(matrices: dict[str, dict[str, FilledMatrix]]) -> tuple[
    dict[tuple[str, int], dict[str, bool]],
    dict[str, list[int]],
]:
    """From filled matrices, produce per-(benchmark, difficulty) game maps
    and per-label frontier lists (for display stats)."""
    cell_map: dict[tuple[str, int], dict[str, bool]] = {}
    label_frontiers: dict[str, list[int]] = {}

    for benchmark, by_suffix in matrices.items():
        for suffix, fm in by_suffix.items():
            for i, dt in enumerate(fm.dtypes):
                label = f"{dt}:{suffix}"
                row = fm.cells[i]
                pass_idx = np.where(row == 1)[0]
                frontier = int(pass_idx.max()) if len(pass_idx) > 0 else -1
                label_frontiers.setdefault(label, []).append(frontier)
                for di in range(fm.cells.shape[1]):
                    if row[di] == 0:
                        continue
                    cell_map.setdefault((benchmark, di), {})[label] = row[di] == 1

    return cell_map, label_frontiers


def rank_variants(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int, int]:
    matrices = build_matrices(df)
    if not matrices:
        return pd.DataFrame(), 0, 0, 0, 0

    cell_map, label_frontiers = _build_games(matrices)
    all_labels = sorted({l for by_label in cell_map.values() for l in by_label},
                        key=lambda l: (l.split(":", 1)[1], _dtype_key(l.split(":", 1)[0]), l))

    model = PlackettLuce()
    ratings = {l: model.rating(name=l) for l in all_labels}
    played: dict[str, int] = {}
    n_games = 0
    skipped_singletons = 0
    skipped_ties = 0

    for (bench, di) in sorted(cell_map):
        by_label = cell_map[(bench, di)]
        if len(by_label) < 2:
            skipped_singletons += 1
            continue
        game_labels = sorted(by_label)
        ranks = [1 if by_label[l] else 2 for l in game_labels]
        if len(set(ranks)) < 2:
            skipped_ties += 1
            continue
        teams = [[ratings[l]] for l in game_labels]
        updated = model.rate(teams, ranks=ranks)
        for i, l in enumerate(game_labels):
            ratings[l] = updated[i][0]
            played[l] = played.get(l, 0) + 1
        n_games += 1

    n_matrices = sum(len(by_suffix) for by_suffix in matrices.values())

    rows = []
    for label in all_labels:
        r = ratings[label]
        dtype = label.split(":", 1)[0]
        frontiers = label_frontiers.get(label, [])
        n_benchmarks = len(frontiers)
        rows.append({
            "Rank": 0,
            "Variant": label,
            "Dtype": dtype,
            "Games": played.get(label, 0),
            "Benchmarks": n_benchmarks,
            "Mean Frontier": float(np.mean(frontiers)) if frontiers else 0.0,
            "Mu": r.mu,
            "Sigma": r.sigma,
            "Ordinal": r.ordinal(),
        })

    rank_df = pd.DataFrame(rows).sort_values(
        ["Ordinal", "Mu", "Sigma"], ascending=[False, False, True], kind="mergesort"
    )
    rank_df["Rank"] = np.arange(1, len(rank_df) + 1)
    return rank_df, n_games, n_matrices, skipped_singletons, skipped_ties


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

    rank_df, n_games, n_matrices, skipped_singletons, skipped_ties = rank_variants(df)

    console.print(f"File: {args.file}")
    console.print("Backend: OpenSkill Plackett-Luce (per-difficulty binary games)")
    console.print(
        f"Entries: {len(df)} | Variants: {len(rank_df)} | "
        f"Matrices: {n_matrices} | Games: {n_games} | "
        f"Skipped: {skipped_singletons} single-variant, {skipped_ties} all-tied"
    )
    console.print()
    _print_legend(console)
    console.print()
    _print_ranking(console, rank_df)


if __name__ == "__main__":
    main()
