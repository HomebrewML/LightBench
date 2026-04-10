import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lightbench.utils import (
    _DTYPE_PREREQ,
    _compose_optimizer,
    _dtype_key,
    _parse_opt_fields,
)


@dataclass(frozen=True)
class ResultData:
    opt_names: list[str]
    bench_names: list[str]
    dtype_names: list[str]
    opt_idx: np.ndarray
    bench_idx: np.ndarray
    dtype_idx: np.ndarray
    difficulty: np.ndarray
    outcome: np.ndarray
    bench_center: np.ndarray | None = None
    bench_scale: np.ndarray | None = None


def load_results(
    path: str, *, steps: int | None = None, trials: int | None = None, timeout: int | None = None
) -> ResultData | None:
    try:
        raw = pd.read_json(path, lines=True)
    except (FileNotFoundError, ValueError):
        return None
    if raw.empty or "benchmark" not in raw or "difficulty" not in raw or "opt" not in raw:
        return None

    df = raw.copy()
    df["benchmark"] = df["benchmark"].astype(str)
    if steps is not None and "steps" in df:
        df = df[df["steps"].astype(int) == int(steps)]
    if trials is not None and "trials" in df:
        df = df[df["trials"].astype(int) == int(trials)]
    if timeout is not None and "timeout" in df:
        df = df[df["timeout"].astype(int) == int(timeout)]
    df["dtype"] = df["dtype"].fillna("fp32").astype(str) if "dtype" in df else "fp32"
    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")
    df = df[np.isfinite(df["difficulty"])]
    df["difficulty"] = df["difficulty"].clip(lower=0.0, upper=1.0)
    if df.empty:
        return None

    parsed = df["opt"].map(_parse_opt_fields)
    df = df.loc[parsed.notna()].copy()
    if df.empty:
        return None
    parsed = pd.DataFrame(parsed.loc[df.index].tolist(), columns=["optimizer", "caution", "mars"], index=df.index)
    df = pd.concat([df, parsed], axis=1)
    df["optimizer"] = df.apply(lambda r: _compose_optimizer(r["optimizer"], r["caution"], r["mars"]), axis=1)

    success = df["success"].astype(bool) if "success" in df else pd.Series(False, index=df.index)
    error = df["error"].fillna("").astype(str).str.strip() if "error" in df else pd.Series("", index=df.index)
    skipped = error.str.startswith(("Skipped", "Gated"))
    state = np.where(skipped, np.nan, np.where(success, 1, 0))
    df = df[np.isfinite(state)]
    if df.empty:
        return None

    df["outcome"] = state.astype(int)

    opt_names = sorted(df["optimizer"].unique().tolist())
    bench_names = sorted(df["benchmark"].unique().tolist())
    dtype_names = sorted(df["dtype"].unique().tolist(), key=_dtype_key)

    bench_stats = df.groupby("benchmark")["difficulty"].agg(["mean", "std"]).fillna(0.0)
    bench_center = np.array([bench_stats.loc[name, "mean"] for name in bench_names], dtype=float)
    bench_scale = np.array([bench_stats.loc[name, "std"] for name in bench_names], dtype=float)
    bench_scale = np.where(bench_scale > 1e-6, bench_scale, 1.0)

    opt_index = {name: i for i, name in enumerate(opt_names)}
    bench_index = {name: i for i, name in enumerate(bench_names)}
    dtype_index = {name: i for i, name in enumerate(dtype_names)}

    opt_idx = df["optimizer"].map(opt_index).to_numpy(dtype=int)
    bench_idx = df["benchmark"].map(bench_index).to_numpy(dtype=int)
    dtype_idx = df["dtype"].map(dtype_index).to_numpy(dtype=int)
    difficulty = df["difficulty"].to_numpy(dtype=float)
    outcome = df["outcome"].to_numpy(dtype=int)

    return ResultData(
        opt_names=opt_names,
        bench_names=bench_names,
        dtype_names=dtype_names,
        opt_idx=opt_idx,
        bench_idx=bench_idx,
        dtype_idx=dtype_idx,
        difficulty=difficulty,
        outcome=outcome,
        bench_center=bench_center,
        bench_scale=bench_scale,
    )


class IRTModel:
    def __init__(
        self,
        opt_names: list[str],
        bench_names: list[str],
        dtype_names: list[str],
        bench_center: np.ndarray | None = None,
        bench_scale: np.ndarray | None = None,
    ):
        self.opt_names = opt_names
        self.bench_names = bench_names
        self.dtype_names = dtype_names
        self.n_opts = len(opt_names)
        self.n_bench = len(bench_names)
        self.n_dtypes = len(dtype_names)
        if bench_center is None:
            bench_center = np.zeros(self.n_bench, dtype=float)
        if bench_scale is None:
            bench_scale = np.ones(self.n_bench, dtype=float)
        self.bench_center = np.asarray(bench_center, dtype=float)
        self.bench_scale = np.asarray(bench_scale, dtype=float)
        self.theta_params = max(self.n_opts - 1, 0)
        self.beta_params = max(self.n_bench - 1, 0)
        self.opt_index = {name: i for i, name in enumerate(opt_names)}
        self.bench_index = {name: i for i, name in enumerate(bench_names)}
        self.dtype_index = {name: i for i, name in enumerate(dtype_names)}
        self.dtype_parent = {}
        for name in dtype_names:
            parent = _DTYPE_PREREQ.get(name)
            while parent is not None and parent not in self.dtype_index:
                parent = _DTYPE_PREREQ.get(parent)
            if parent in self.dtype_index:
                self.dtype_parent[self.dtype_index[name]] = self.dtype_index[parent]
        self.delta_index = {}
        delta_params = 0
        for name in dtype_names:
            idx = self.dtype_index[name]
            if idx in self.dtype_parent:
                self.delta_index[idx] = delta_params
                delta_params += 1
            else:
                self.delta_index[idx] = None
        self.delta_params = delta_params
        self.delta_order = sorted(range(self.n_dtypes), key=self._dtype_depth)
        self.params = None
        self.result = None

    def _dtype_depth(self, idx: int) -> int:
        depth = 0
        parent = self.dtype_parent.get(idx)
        while parent is not None:
            depth += 1
            parent = self.dtype_parent.get(parent)
        return depth

    def _expand_centered(self, raw, total):
        if total <= 1:
            return np.zeros(total, dtype=float)
        raw = np.asarray(raw, dtype=float)
        if raw.size != total - 1:
            raise ValueError("Invalid parameter length")
        last = -raw.sum()
        return np.concatenate([raw, np.array([last], dtype=float)], axis=0)

    def _pack(self, theta, beta, log_gamma, delta_raw):
        parts = [theta, beta, log_gamma]
        if delta_raw is not None and len(delta_raw):
            parts.append(delta_raw)
        return np.concatenate(parts, axis=0)

    def _unpack(self, x):
        idx = 0
        theta_raw = x[idx : idx + self.theta_params]
        idx += self.theta_params
        beta_raw = x[idx : idx + self.beta_params]
        idx += self.beta_params
        log_gamma = x[idx : idx + self.n_bench]
        idx += self.n_bench
        delta_raw = x[idx : idx + self.delta_params] if self.delta_params else np.array([])

        theta = self._expand_centered(theta_raw, self.n_opts)
        beta = self._expand_centered(beta_raw, self.n_bench)

        gamma = np.exp(log_gamma)

        delta = np.zeros(self.n_dtypes, dtype=float)
        for i in self.delta_order:
            parent = self.dtype_parent.get(i)
            raw_idx = self.delta_index.get(i)
            if parent is None or raw_idx is None:
                delta[i] = 0.0
            else:
                delta[i] = delta[parent] + np.exp(delta_raw[raw_idx])

        return theta, beta, gamma, delta, log_gamma

    def init_params(self, data=None):
        theta = np.zeros(self.theta_params, dtype=float)
        beta = np.zeros(self.beta_params, dtype=float)
        if data is not None:
            opt_idx, bench_idx, _dtype_idx, _difficulty, outcome = data
            eps = 0.02
            if self.n_opts:
                opt_rate = np.bincount(opt_idx, weights=outcome, minlength=self.n_opts)
                opt_total = np.bincount(opt_idx, minlength=self.n_opts)
                opt_rate = opt_rate / np.maximum(opt_total, 1)
                opt_rate = np.clip(opt_rate, eps, 1 - eps)
                theta_full = np.log(opt_rate / (1 - opt_rate))
                theta_full = theta_full - theta_full.mean()
                if self.theta_params:
                    theta = theta_full[:-1]
            if self.n_bench:
                bench_rate = np.bincount(bench_idx, weights=outcome, minlength=self.n_bench)
                bench_total = np.bincount(bench_idx, minlength=self.n_bench)
                bench_rate = bench_rate / np.maximum(bench_total, 1)
                bench_rate = np.clip(bench_rate, eps, 1 - eps)
                beta_full = -np.log(bench_rate / (1 - bench_rate))
                beta_full = beta_full - beta_full.mean()
                if self.beta_params:
                    beta = beta_full[:-1]
        log_gamma = np.zeros(self.n_bench, dtype=float)
        delta_raw = np.full(self.delta_params, np.log(0.1), dtype=float) if self.delta_params else np.array([])
        return self._pack(theta, beta, log_gamma, delta_raw)

    def neg_log_likelihood(self, x, data):
        opt_idx, bench_idx, dtype_idx, difficulty, outcome = data
        theta, beta, gamma, delta, log_gamma = self._unpack(x)
        centered = (difficulty - self.bench_center[bench_idx]) / self.bench_scale[bench_idx]
        logits = theta[opt_idx] - delta[dtype_idx] - beta[bench_idx] - gamma[bench_idx] * centered
        nll = np.logaddexp(0, logits) - outcome * logits
        loss = float(np.sum(nll))
        loss += 0.5 * np.sum(theta**2)
        loss += 0.5 * np.sum(beta**2)
        if self.delta_params:
            delta_raw = x[-self.delta_params :]
            loss += 0.5 * np.sum(delta_raw**2)
        loss += 0.5 * np.sum((log_gamma / 0.5) ** 2)
        return loss

    def fit(self, data, max_iter=500):
        x0 = self.init_params(data)
        result = minimize(
            self.neg_log_likelihood,
            x0,
            args=(data,),
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        self.params = result.x
        self.result = result
        return result

    def unpacked(self):
        if self.params is None:
            return None
        return self._unpack(self.params)

    def _hess_inv_dense(self):
        if self.result is not None and self.result.hess_inv is not None:
            try:
                return np.array(self.result.hess_inv.todense())
            except Exception:
                return np.array(self.result.hess_inv)
        return getattr(self, "_cached_hess_inv", None)

    def theta_se(self):
        cov = self._hess_inv_dense()
        if cov is None or cov.ndim != 2 or cov.shape[0] < self.theta_params:
            return np.full(self.n_opts, np.nan)
        if self.n_opts <= 1:
            return np.zeros(self.n_opts, dtype=float)
        cov_theta = cov[: self.theta_params, : self.theta_params]
        if cov_theta.size == 0:
            return np.full(self.n_opts, np.nan)
        var = np.zeros(self.n_opts, dtype=float)
        var[:-1] = np.diag(cov_theta)
        var[-1] = cov_theta.sum()
        var = np.clip(var, 0, None)
        return np.sqrt(var)

    def capability_score(self, opt_idx: int, bench_idx: int, dtype_idx: int) -> float:
        theta, beta, gamma, delta, _log_gamma = self._unpack(self.params)
        return float(
            self.bench_center[bench_idx]
            + self.bench_scale[bench_idx] * (theta[opt_idx] - delta[dtype_idx] - beta[bench_idx]) / gamma[bench_idx]
        )

    def prior_for(self, opt_name, bench_name, dtype_name, precision):
        if self.params is None:
            return None
        oi = self.opt_index.get(opt_name)
        bi = self.bench_index.get(bench_name)
        di = self.dtype_index.get(dtype_name)
        if oi is None or bi is None or di is None:
            return None
        theta, beta, gamma, delta, _ = self._unpack(self.params)
        g = gamma[bi] if np.isfinite(gamma[bi]) and gamma[bi] > 0 else 1.0
        bs = max(self.bench_scale[bi], 1e-6)
        dstar = float(np.clip(self.bench_center[bi] + bs * (theta[oi] - delta[di] - beta[bi]) / g, 0.0, 1.0))
        slope = float(max(g / bs, 1e-3))
        return dstar, slope, self.theta_se()[oi], precision

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_hess_inv"] = self._hess_inv_dense()
        state["result"] = None
        return state


def fit(data: ResultData, max_iter=500) -> IRTModel:
    model = IRTModel(
        data.opt_names,
        data.bench_names,
        data.dtype_names,
        bench_center=data.bench_center,
        bench_scale=data.bench_scale,
    )
    model.fit((data.opt_idx, data.bench_idx, data.dtype_idx, data.difficulty, data.outcome), max_iter=max_iter)
    return model


def rank_optimizers(model: IRTModel, data: ResultData) -> pd.DataFrame:
    theta, _beta, _gamma, _delta, _log_gamma = model.unpacked()
    theta_se = model.theta_se()
    games = np.bincount(data.opt_idx, minlength=model.n_opts)
    benchmarks = [len(np.unique(data.bench_idx[data.opt_idx == i])) for i in range(model.n_opts)]

    order = np.argsort(-theta)
    ranks = np.empty_like(order)
    rank = 0
    prev = None
    for i, idx in enumerate(order, 1):
        if prev is None or theta[idx] != prev:
            rank = i
            prev = theta[idx]
        ranks[idx] = rank

    df = pd.DataFrame({
        "Rank": ranks,
        "Optimizer": model.opt_names,
        "Theta": theta,
        "Theta_SE": theta_se,
        "Games": games,
        "Benchmarks": benchmarks,
    })
    return df.sort_values(["Rank", "Optimizer"]).reset_index(drop=True)


def diagnostics(model: IRTModel, data: ResultData) -> dict[str, object]:
    theta, beta, gamma, delta, _log_gamma = model.unpacked()
    opt_uncertainty = pd.DataFrame({
        "Optimizer": model.opt_names,
        "Theta_SE": model.theta_se(),
    }).sort_values("Theta_SE")
    bench_stats = pd.DataFrame({
        "Benchmark": model.bench_names,
        "Gamma": gamma,
        "Beta": beta,
    }).sort_values("Gamma", ascending=False)
    dtype_stats = pd.DataFrame({"Dtype": model.dtype_names, "Delta": delta}).sort_values("Delta")

    opt_idx, bench_idx, dtype_idx, difficulty, outcome = (
        data.opt_idx,
        data.bench_idx,
        data.dtype_idx,
        data.difficulty,
        data.outcome,
    )
    centered = (difficulty - model.bench_center[bench_idx]) / model.bench_scale[bench_idx]
    logits = theta[opt_idx] - delta[dtype_idx] - beta[bench_idx] - gamma[bench_idx] * centered
    loglik = float(np.sum(outcome * logits - np.logaddexp(0, logits)))
    n_obs = len(outcome)
    n_params = len(model.params) if model.params is not None else 0
    aic = 2 * n_params - 2 * loglik if n_params else float("nan")
    bic = np.log(n_obs) * n_params - 2 * loglik if n_params else float("nan")

    fit_stats = {
        "LogLikelihood": loglik,
        "Observations": n_obs,
        "Parameters": n_params,
        "AIC": aic,
        "BIC": bic,
    }
    return {"benchmarks": bench_stats, "optimizers": opt_uncertainty, "dtypes": dtype_stats, "fit": fit_stats}


def capability_score(model: IRTModel, opt: str, bench: str, dtype: str) -> float:
    opt_idx = model.opt_names.index(opt)
    bench_idx = model.bench_names.index(bench)
    dtype_idx = model.dtype_names.index(dtype)
    return model.capability_score(opt_idx, bench_idx, dtype_idx)


def main():
    parser = argparse.ArgumentParser(description="Rank optimizers with an IRT model")
    parser.add_argument("file", nargs="?", default="benchmark_results.jsonl")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    data = load_results(args.file)
    if data is None:
        print("No usable results found.")
        return

    model = fit(data, max_iter=args.max_iter)
    ranking = rank_optimizers(model, data)
    print(ranking.to_string(index=False))

    if not args.diagnostics:
        return
    info = diagnostics(model, data)
    print("\nBenchmark discrimination:")
    print(info["benchmarks"].to_string(index=False))
    print("\nOptimizer uncertainty:")
    print(info["optimizers"].to_string(index=False))
    print("\nDtype penalties:")
    print(info["dtypes"].to_string(index=False))
    print("\nFit statistics:")
    for k, v in info["fit"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
