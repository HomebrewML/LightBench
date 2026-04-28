import argparse
import collections
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lightbench.ranking_summary import posterior_rank_table
from lightbench.utils import _compose_optimizer, _dtype_key, _parse_opt_fields


@dataclass(frozen=True)
class ResultData:
    challenger_names: list[tuple[str, str]]  # (optimizer, dtype)
    bench_names: list[str]
    challenger_idx: np.ndarray
    bench_idx: np.ndarray
    difficulty: np.ndarray
    outcome: np.ndarray


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
    # Subprocess deaths: when *all* steps for a (opt, bench) trace died, the optimizer
    # genuinely cannot run that benchmark -> conclusive failure. When only some died,
    # the deaths are likely infra hiccups; skip them as inconclusive.
    died = error.eq("Subprocess died")
    if died.any():
        per_pair = died.groupby([df["opt"], df["benchmark"]]).transform("sum")
        per_total = died.groupby([df["opt"], df["benchmark"]]).transform("size")
        intermittent = died & (per_pair < per_total)
        skipped = skipped | intermittent
    state = np.where(skipped, np.nan, np.where(success, 1, 0))
    mask = np.isfinite(state)
    df = df[mask]
    if df.empty:
        return None
    df["outcome"] = state[mask].astype(int)

    unique = df[["optimizer", "dtype"]].drop_duplicates().values.tolist()
    challenger_names = sorted([tuple(x) for x in unique], key=lambda x: (x[0], _dtype_key(x[1])))
    bench_names = sorted(df["benchmark"].unique().tolist())

    challenger_index = {name: i for i, name in enumerate(challenger_names)}
    bench_index = {name: i for i, name in enumerate(bench_names)}

    keys = list(zip(df["optimizer"].values, df["dtype"].values))
    challenger_idx = np.array([challenger_index[k] for k in keys], dtype=int)
    bench_idx = df["benchmark"].map(bench_index).to_numpy(dtype=int)
    difficulty = df["difficulty"].to_numpy(dtype=float)
    outcome = df["outcome"].to_numpy(dtype=int)

    return ResultData(
        challenger_names=challenger_names,
        bench_names=bench_names,
        challenger_idx=challenger_idx,
        bench_idx=bench_idx,
        difficulty=difficulty,
        outcome=outcome,
    )


def _derive_tags(paths: list[str]) -> list[str]:
    parents = [os.path.basename(os.path.dirname(os.path.abspath(p))) or os.path.basename(p) for p in paths]
    if len(set(parents)) == len(parents):
        return parents
    bases = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(bases)) == len(bases):
        return bases
    return [f"src{i}" for i in range(len(paths))]


def load_multi_results(
    sources: Iterable[tuple[str, str]] | dict[str, str] | list[str],
    *,
    steps: int | None = None,
    trials: int | None = None,
    timeout: int | None = None,
    namespace: bool = False,
) -> ResultData | None:
    """Load multiple jsonl files into a single ResultData.

    Optimizer/dtype challengers and benchmarks are unioned by name across
    sources, so the same task seen in two files merges into one bench
    (more cross-optimizer comparisons identify each benchmark's
    parameters). Filter by ``steps``/``trials``/``timeout`` to keep run
    conditions comparable before merging. Pass ``namespace=True`` to
    preserve the legacy ``"<tag>/<benchmark>"`` split when conditions
    genuinely differ across sources.
    """
    if isinstance(sources, dict):
        tagged = list(sources.items())
    else:
        items = list(sources)
        if items and isinstance(items[0], str):
            tagged = list(zip(_derive_tags(items), items))
        else:
            tagged = [(str(t), str(p)) for t, p in items]

    pieces: list[tuple[str, ResultData]] = []
    for tag, path in tagged:
        d = load_results(path, steps=steps, trials=trials, timeout=timeout)
        if d is not None:
            pieces.append((tag, d))
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0][1]

    chal_names = sorted({c for _, d in pieces for c in d.challenger_names},
                        key=lambda x: (x[0], _dtype_key(x[1])))
    chal_index = {c: i for i, c in enumerate(chal_names)}

    if namespace:
        bench_names: list[str] = []
        bench_remaps: list[np.ndarray] = []
        for tag, d in pieces:
            off = len(bench_names)
            bench_names.extend(f"{tag}/{b}" for b in d.bench_names)
            bench_remaps.append(np.arange(len(d.bench_names)) + off)
    else:
        bench_names = sorted({b for _, d in pieces for b in d.bench_names})
        bench_index = {b: i for i, b in enumerate(bench_names)}
        bench_remaps = [
            np.array([bench_index[b] for b in d.bench_names], dtype=int)
            for _, d in pieces
        ]

    chal_idx_parts, bench_idx_parts, diff_parts, out_parts = [], [], [], []
    for bench_remap, (_tag, d) in zip(bench_remaps, pieces):
        chal_remap = np.array([chal_index[c] for c in d.challenger_names], dtype=int)
        chal_idx_parts.append(chal_remap[d.challenger_idx])
        bench_idx_parts.append(bench_remap[d.bench_idx])
        diff_parts.append(d.difficulty)
        out_parts.append(d.outcome)

    return ResultData(
        challenger_names=chal_names,
        bench_names=bench_names,
        challenger_idx=np.concatenate(chal_idx_parts).astype(int),
        bench_idx=np.concatenate(bench_idx_parts).astype(int),
        difficulty=np.concatenate(diff_parts).astype(float),
        outcome=np.concatenate(out_parts).astype(int),
    )


class IRTModel:
    """2PL IRT with continuous difficulty covariate.

    logit P(success) = alpha[bench] * (theta[challenger] - b[bench] - c[bench] * difficulty)

    alpha > 0 : benchmark discrimination (how well it separates challengers)
    theta     : challenger ability (sum-to-zero)
    b         : benchmark difficulty intercept (sum-to-zero)
    c > 0     : difficulty slope (higher difficulty -> lower success)

    Regularisation strengths (sigma_*) are tuned via empirical Bayes.
    """

    def __init__(self, challenger_names: list[tuple[str, str]], bench_names: list[str]):
        self.challenger_names = list(challenger_names)
        self.bench_names = list(bench_names)
        self.n_challengers = len(self.challenger_names)
        self.n_bench = len(self.bench_names)
        self.theta_params = max(self.n_challengers - 1, 0)
        self.b_params = max(self.n_bench - 1, 0)
        self.challenger_index: dict[tuple[str, str], int] = {n: i for i, n in enumerate(self.challenger_names)}
        self.bench_index: dict[str, int] = {n: i for i, n in enumerate(self.bench_names)}
        self.sigma_theta = 1.0
        self.sigma_b = 1.0
        self.sigma_log_alpha = 1.0
        self.sigma_log_c = 1.0
        self.params: np.ndarray | None = None
        self.result = None

    # -- parameterisation helpers --

    @staticmethod
    def _expand_centered(raw, total):
        if total <= 1:
            return np.zeros(total, dtype=float)
        raw = np.asarray(raw, dtype=float)
        return np.concatenate([raw, np.array([-raw.sum()], dtype=float)])

    def _n_params(self):
        return self.theta_params + self.b_params + 2 * self.n_bench

    def _pack(self, theta_raw, b_raw, log_alpha, log_c):
        return np.concatenate([theta_raw, b_raw, log_alpha, log_c])

    def _unpack(self, x):
        idx = 0
        theta_raw = x[idx : idx + self.theta_params]
        idx += self.theta_params
        b_raw = x[idx : idx + self.b_params]
        idx += self.b_params
        log_alpha = x[idx : idx + self.n_bench]
        idx += self.n_bench
        log_c = x[idx : idx + self.n_bench]

        theta = self._expand_centered(theta_raw, self.n_challengers)
        b = self._expand_centered(b_raw, self.n_bench)
        alpha = np.exp(np.clip(log_alpha, -10, 10))
        c = np.exp(np.clip(log_c, -10, 10))
        return theta, b, alpha, c, log_alpha, log_c

    # -- initialisation --

    def init_params(self, data=None):
        theta = np.zeros(self.theta_params, dtype=float)
        b = np.zeros(self.b_params, dtype=float)
        if data is not None:
            challenger_idx, bench_idx, _difficulty, outcome = data
            eps = 0.02
            if self.n_challengers:
                rate = np.bincount(challenger_idx, weights=outcome, minlength=self.n_challengers)
                total = np.bincount(challenger_idx, minlength=self.n_challengers)
                rate = np.clip(rate / np.maximum(total, 1), eps, 1 - eps)
                theta_full = np.log(rate / (1 - rate))
                theta_full -= theta_full.mean()
                if self.theta_params:
                    theta = theta_full[:-1]
            if self.n_bench:
                rate = np.bincount(bench_idx, weights=outcome, minlength=self.n_bench)
                total = np.bincount(bench_idx, minlength=self.n_bench)
                rate = np.clip(rate / np.maximum(total, 1), eps, 1 - eps)
                b_full = -np.log(rate / (1 - rate))
                b_full -= b_full.mean()
                if self.b_params:
                    b = b_full[:-1]
        log_alpha = np.zeros(self.n_bench, dtype=float)
        log_c = np.zeros(self.n_bench, dtype=float)
        return self._pack(theta, b, log_alpha, log_c)

    def _random_init(self, rng: np.random.Generator):
        theta = rng.normal(0, self.sigma_theta, self.theta_params)
        b = rng.normal(0, self.sigma_b, self.b_params)
        log_alpha = rng.normal(0, self.sigma_log_alpha, self.n_bench)
        log_c = rng.normal(0, self.sigma_log_c, self.n_bench)
        return self._pack(theta, b, log_alpha, log_c)

    # -- likelihood --

    def neg_log_likelihood(self, x, data):
        challenger_idx, bench_idx, difficulty, outcome = data
        theta, b, alpha, c, log_alpha, log_c = self._unpack(x)
        logits = alpha[bench_idx] * (theta[challenger_idx] - b[bench_idx] - c[bench_idx] * difficulty)
        nll = np.logaddexp(0, logits) - outcome * logits
        loss = float(np.sum(nll))
        loss += 0.5 * np.sum((theta / self.sigma_theta) ** 2)
        loss += 0.5 * np.sum((b / self.sigma_b) ** 2)
        loss += 0.5 * np.sum((log_alpha / self.sigma_log_alpha) ** 2)
        loss += 0.5 * np.sum((log_c / self.sigma_log_c) ** 2)
        return loss

    # -- fitting --

    def fit(self, data, max_iter: int = 500, eb_rounds: int = 10, restarts: int = 10):
        # Phase 1: empirical Bayes — alternate MAP estimation and hyperparameter updates
        x0 = self.init_params(data)
        result = None
        for _ in range(eb_rounds):
            result = minimize(self.neg_log_likelihood, x0, args=(data,), method="L-BFGS-B",
                              options={"maxiter": max_iter})
            x0 = result.x
            theta, b, _a, _c, log_alpha, log_c = self._unpack(result.x)

            new_st = max(np.sqrt(np.mean(theta**2)), 0.1)
            new_sb = max(np.sqrt(np.mean(b**2)), 0.1)
            new_sla = max(np.sqrt(np.mean(log_alpha**2)), 0.1)
            new_slc = max(np.sqrt(np.mean(log_c**2)), 0.1)

            converged = (
                abs(new_st - self.sigma_theta) < 0.01 * self.sigma_theta
                and abs(new_sb - self.sigma_b) < 0.01 * self.sigma_b
            )
            self.sigma_theta = new_st
            self.sigma_b = new_sb
            self.sigma_log_alpha = new_sla
            self.sigma_log_c = new_slc
            if converged:
                break

        best = result

        # Phase 2: random restarts with learned hyperparameters
        rng = np.random.default_rng(42)
        for _ in range(restarts):
            x0r = self._random_init(rng)
            r = minimize(self.neg_log_likelihood, x0r, args=(data,), method="L-BFGS-B",
                         options={"maxiter": max_iter})
            if r.fun < best.fun:
                best = r

        self.params = best.x
        self.result = best
        self._compute_observed_information(data)
        return best

    def _compute_observed_information(self, data):
        """Analytical Hessian of the penalised NLL at the mode.

        H = J^T diag(w) J + H_penalty

        where J is the Jacobian of logits w.r.t. free parameters and
        w_k = p_k(1 - p_k) are the Fisher information weights.
        """
        challenger_idx, bench_idx, difficulty, outcome = data
        theta, b, alpha, c, _la, _lc = self._unpack(self.params)
        n_obs = len(outcome)
        n = self._n_params()

        inner = theta[challenger_idx] - b[bench_idx] - c[bench_idx] * difficulty
        logits = alpha[bench_idx] * inner
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        w = p * (1.0 - p)
        a = alpha[bench_idx]

        # --- build Jacobian J (n_obs × n_params) ---
        J = np.zeros((n_obs, n))
        obs = np.arange(n_obs)
        last_c = self.n_challengers - 1
        last_b = self.n_bench - 1

        # theta_raw block
        free_c = challenger_idx < last_c
        J[obs[free_c], challenger_idx[free_c]] = a[free_c]
        last_c_mask = challenger_idx == last_c
        if self.theta_params > 0 and last_c_mask.any():
            J[np.ix_(obs[last_c_mask], np.arange(self.theta_params))] = -a[last_c_mask, None]

        # b_raw block
        off_b = self.theta_params
        free_b = bench_idx < last_b
        J[obs[free_b], off_b + bench_idx[free_b]] = -a[free_b]
        last_b_mask = bench_idx == last_b
        if self.b_params > 0 and last_b_mask.any():
            J[np.ix_(obs[last_b_mask], off_b + np.arange(self.b_params))] = a[last_b_mask, None]

        # log_alpha block: d(logit)/d(log_alpha_b) = alpha_b * inner = logit
        off_a = self.theta_params + self.b_params
        J[obs, off_a + bench_idx] = a * inner

        # log_c block: d(logit)/d(log_c_b) = -alpha_b * c_b * d
        off_c = off_a + self.n_bench
        J[obs, off_c + bench_idx] = -a * c[bench_idx] * difficulty

        # --- H = J^T W J ---
        H = (J * w[:, None]).T @ J

        # --- add penalty Hessian ---
        # theta penalty: 0.5*sum(theta_full^2)/sigma^2
        #   d^2/d(raw_i)(raw_j) = (delta_ij + 1) / sigma^2
        tp = self.theta_params
        if tp:
            H[:tp, :tp] += (np.eye(tp) + 1.0) / self.sigma_theta**2
        bp = self.b_params
        if bp:
            H[off_b : off_b + bp, off_b : off_b + bp] += (np.eye(bp) + 1.0) / self.sigma_b**2
        idx_a = np.arange(off_a, off_a + self.n_bench)
        H[idx_a, idx_a] += 1.0 / self.sigma_log_alpha**2
        idx_c = np.arange(off_c, off_c + self.n_bench)
        H[idx_c, idx_c] += 1.0 / self.sigma_log_c**2

        try:
            self._cached_hess_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self._cached_hess_inv = np.linalg.pinv(H)

    # -- accessors --

    def unpacked(self):
        if self.params is None:
            return None
        theta, b, alpha, c, _la, _lc = self._unpack(self.params)
        return theta, b, alpha, c

    def _hess_inv_dense(self):
        return getattr(self, "_cached_hess_inv", None)

    def theta_se(self):
        cov = self._hess_inv_dense()
        if cov is None or cov.ndim != 2 or cov.shape[0] < self.theta_params:
            return np.full(self.n_challengers, np.nan)
        if self.n_challengers <= 1:
            return np.zeros(self.n_challengers, dtype=float)
        cov_theta = cov[: self.theta_params, : self.theta_params]
        if cov_theta.size == 0:
            return np.full(self.n_challengers, np.nan)
        var = np.zeros(self.n_challengers, dtype=float)
        var[:-1] = np.diag(cov_theta)
        var[-1] = cov_theta.sum()
        return np.sqrt(np.clip(var, 0, None))

    def theta_samples(self, n_samples: int = 2000, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw theta from the Laplace posterior at the MAP.

        Returns shape (n_samples, n_challengers). The last challenger is
        determined by the sum-to-zero constraint. Falls back to MAP-only
        samples if the Hessian inverse is unavailable.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        if self.n_challengers <= 1 or self.params is None:
            theta_map = self._unpack(self.params)[0] if self.params is not None else np.zeros(self.n_challengers)
            return np.broadcast_to(theta_map, (n_samples, self.n_challengers)).copy()
        cov = self._hess_inv_dense()
        theta_raw_map = self.params[: self.theta_params]
        if cov is None or cov.ndim != 2 or cov.shape[0] < self.theta_params or self.theta_params == 0:
            theta_map = self._unpack(self.params)[0]
            return np.broadcast_to(theta_map, (n_samples, self.n_challengers)).copy()
        cov_theta = cov[: self.theta_params, : self.theta_params]
        cov_theta = 0.5 * (cov_theta + cov_theta.T) + 1e-10 * np.eye(self.theta_params)
        try:
            samples_raw = rng.multivariate_normal(theta_raw_map, cov_theta, size=n_samples)
        except np.linalg.LinAlgError:
            diag = np.clip(np.diag(cov_theta), 1e-12, None)
            samples_raw = theta_raw_map + rng.standard_normal((n_samples, self.theta_params)) * np.sqrt(diag)
        last = -samples_raw.sum(axis=1, keepdims=True)
        return np.concatenate([samples_raw, last], axis=1)

    def capability_score(self, challenger_idx: int, bench_idx: int) -> float:
        theta, b, _alpha, c, _la, _lc = self._unpack(self.params)
        ci = c[bench_idx]
        if ci < 1e-10:
            return float("nan")
        return float((theta[challenger_idx] - b[bench_idx]) / ci)

    def prior_for(self, challenger: tuple[str, str], bench: str, precision: float):
        if self.params is None:
            return None
        ci = self.challenger_index.get(challenger)
        bi = self.bench_index.get(bench)
        if ci is None or bi is None:
            return None
        theta, b, alpha, c, _la, _lc = self._unpack(self.params)
        cval = c[bi]
        if not (np.isfinite(cval) and cval > 0):
            return None
        dstar = float(np.clip((theta[ci] - b[bi]) / cval, 0.0, 1.0))
        slope = float(max(alpha[bi] * cval, 1e-3))
        return dstar, slope, self.theta_se()[ci], precision

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_hess_inv"] = self._hess_inv_dense()
        state["result"] = None
        return state


# -- module-level API --


def fit(data: ResultData, max_iter: int = 500, restarts: int = 10) -> IRTModel:
    model = IRTModel(data.challenger_names, data.bench_names)
    model.fit(
        (data.challenger_idx, data.bench_idx, data.difficulty, data.outcome),
        max_iter=max_iter,
        restarts=restarts,
    )
    return model


def rank_optimizers(
    model: IRTModel,
    data: ResultData,
    *,
    n_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """One shareable ranking under the Laplace approximation.

    Shares the per-draw → ranking summary with the Bayesian backend
    (``bayes_rating.rating_table``) via ``ranking_summary.posterior_rank_table``;
    the only frequentist-specific bit is the Laplace covariance source for
    ``theta_samples``. Argsorting the MAP theta treats all theta gaps as
    equally informative, which is wrong when theta_SE varies by orders of
    magnitude across challengers (heavy boundary censoring → wide posterior;
    dense interior → narrow). Sampling theta from the Laplace posterior and
    averaging the per-draw rank vector accounts for that automatically.
    """
    theta_map, _b, _alpha, _c = model.unpacked()
    n = model.n_challengers
    rng = np.random.default_rng(seed)
    samples = model.theta_samples(n_samples=n_samples, rng=rng)
    extras = {
        "Theta_SE": model.theta_se(),
        "Games": np.bincount(data.challenger_idx, minlength=n),
        "Benchmarks": np.array(
            [len(np.unique(data.bench_idx[data.challenger_idx == i])) for i in range(n)],
            dtype=int,
        ),
    }
    return posterior_rank_table(
        samples,
        model.challenger_names,
        theta_map=theta_map,
        extras=extras,
    )


def pairwise_dominance(
    model: IRTModel,
    *,
    n_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Posterior P(theta_i > theta_j) for every challenger pair."""
    rng = np.random.default_rng(seed)
    samples = model.theta_samples(n_samples=n_samples, rng=rng)
    diff = samples[:, :, None] - samples[:, None, :]
    p = (diff > 0).mean(axis=0)
    labels = [f"{opt}@{dt}" for opt, dt in model.challenger_names]
    return pd.DataFrame(p, index=labels, columns=labels)


class MIRT:
    """K-factor 2PL IRT, used for dimensionality CV — not for ranking.

    logit P(success) = sum_k alpha[b, k] * theta[c, k] - b[b] - c[b] * difficulty

    Theta and alpha are not separately identified (rotation-free), but the
    likelihood is rotation-invariant, so held-out predictive log-prob is a
    well-defined comparator across K. Gaussian priors on every block keep
    the MAP bounded.
    """

    def __init__(
        self,
        n_chal: int,
        n_bench: int,
        K: int = 1,
        sigma_theta: float = 1.0,
        sigma_alpha: float = 1.0,
        sigma_b: float = 1.0,
        sigma_log_c: float = 1.0,
    ):
        self.n_chal = int(n_chal)
        self.n_bench = int(n_bench)
        self.K = int(K)
        self.sigma_theta = float(sigma_theta)
        self.sigma_alpha = float(sigma_alpha)
        self.sigma_b = float(sigma_b)
        self.sigma_log_c = float(sigma_log_c)
        self.params: np.ndarray | None = None
        self.result = None

    def _n_params(self) -> int:
        return self.n_chal * self.K + self.n_bench * self.K + 2 * self.n_bench

    def _unpack(self, x: np.ndarray):
        idx = 0
        theta = x[idx : idx + self.n_chal * self.K].reshape(self.n_chal, self.K)
        idx += self.n_chal * self.K
        alpha = x[idx : idx + self.n_bench * self.K].reshape(self.n_bench, self.K)
        idx += self.n_bench * self.K
        b = x[idx : idx + self.n_bench]
        idx += self.n_bench
        log_c = x[idx : idx + self.n_bench]
        c = np.exp(np.clip(log_c, -10, 10))
        return theta, alpha, b, c, log_c

    def neg_log_likelihood(self, x: np.ndarray, data) -> float:
        challenger_idx, bench_idx, difficulty, outcome = data
        theta, alpha, b, c, log_c = self._unpack(x)
        contributions = np.einsum("ok,ok->o", theta[challenger_idx], alpha[bench_idx])
        logits = contributions - b[bench_idx] - c[bench_idx] * difficulty
        nll = np.logaddexp(0, logits) - outcome * logits
        loss = float(np.sum(nll))
        loss += 0.5 * np.sum((theta / self.sigma_theta) ** 2)
        loss += 0.5 * np.sum((alpha / self.sigma_alpha) ** 2)
        loss += 0.5 * np.sum((b / self.sigma_b) ** 2)
        loss += 0.5 * np.sum((log_c / self.sigma_log_c) ** 2)
        return loss

    def fit(self, data, *, max_iter: int = 500, n_restarts: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        best = None
        for r in range(max(n_restarts, 1)):
            x0 = rng.standard_normal(self._n_params()) * 0.1
            result = minimize(
                self.neg_log_likelihood, x0, args=(data,),
                method="L-BFGS-B", options={"maxiter": max_iter},
            )
            if best is None or result.fun < best.fun:
                best = result
        self.params = best.x
        self.result = best
        return best

    def predict_log_likelihood(self, data) -> float:
        challenger_idx, bench_idx, difficulty, outcome = data
        theta, alpha, b, c, _log_c = self._unpack(self.params)
        contributions = np.einsum("ok,ok->o", theta[challenger_idx], alpha[bench_idx])
        logits = contributions - b[bench_idx] - c[bench_idx] * difficulty
        log_p_one = -np.logaddexp(0, -logits)
        log_p_zero = -np.logaddexp(0, logits)
        return float(np.sum(np.where(outcome == 1, log_p_one, log_p_zero)))


def _subset_data(data: ResultData, mask: np.ndarray) -> ResultData:
    return ResultData(
        challenger_names=data.challenger_names,
        bench_names=data.bench_names,
        challenger_idx=data.challenger_idx[mask],
        bench_idx=data.bench_idx[mask],
        difficulty=data.difficulty[mask],
        outcome=data.outcome[mask],
    )


def _stratified_cell_folds(
    challenger_idx: np.ndarray,
    bench_idx: np.ndarray,
    n_folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign each (chal, bench) cell to a fold, stratified per challenger.

    Holding out random *cells* (not random trials) is what tests whether
    the K-factor model generalises across benchmarks for a given optimizer.
    Holding out trials within a cell would test in-cell prediction, which
    is uninteresting given 8 trials per cell come from the same posterior.

    Stratifying per challenger guarantees every challenger appears in
    every training fold, so theta is identifiable for every entity.
    """
    cells = list({(int(c), int(b)) for c, b in zip(challenger_idx, bench_idx)})
    cells_per_chal: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for cell in cells:
        cells_per_chal[cell[0]].append(cell)
    fold_of: dict[tuple[int, int], int] = {}
    for chal, chal_cells in cells_per_chal.items():
        idxs = np.arange(len(chal_cells))
        rng.shuffle(idxs)
        for i, j in enumerate(idxs):
            fold_of[chal_cells[j]] = int(i % n_folds)
    return np.array([fold_of[(int(c), int(b))] for c, b in zip(challenger_idx, bench_idx)], dtype=int)


def dimensionality_cv(
    data: ResultData,
    *,
    K_max: int = 2,
    n_folds: int = 5,
    max_iter: int = 300,
    n_restarts: int = 2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold CV comparing held-out predictive log-prob across MIRT factor counts.

    A meaningful gap of K=2 over K=1 (in mean per-observation held-out
    log-likelihood, beyond 1 SE) means unidimensional theta is dropping
    benchmark structure that a 2-factor model recovers — the scalar
    ranking is then misspecified and the K=2 loadings are the answer.
    """
    rng = np.random.default_rng(seed)
    fold_idx = _stratified_cell_folds(data.challenger_idx, data.bench_idx, n_folds, rng)
    n_chal = len(data.challenger_names)
    n_bench = len(data.bench_names)

    rows = []
    for K in range(1, K_max + 1):
        for f in range(n_folds):
            train = _subset_data(data, fold_idx != f)
            test = _subset_data(data, fold_idx == f)
            if len(test.outcome) == 0 or len(train.outcome) == 0:
                continue
            model = MIRT(n_chal, n_bench, K=K)
            model.fit(
                (train.challenger_idx, train.bench_idx, train.difficulty, train.outcome),
                max_iter=max_iter, n_restarts=n_restarts, seed=seed * 101 + f * 7 + K * 13,
            )
            logp = model.predict_log_likelihood(
                (test.challenger_idx, test.bench_idx, test.difficulty, test.outcome)
            )
            rows.append({
                "K": K, "fold": f, "n_test": int(len(test.outcome)),
                "logp": logp, "logp_per_obs": logp / max(len(test.outcome), 1),
            })

    folds = pd.DataFrame(rows)
    if folds.empty:
        return folds, pd.DataFrame(columns=["K", "mean_logp_per_obs", "se_logp_per_obs", "total_logp"])
    summary = (
        folds.groupby("K")
        .agg(
            mean_logp_per_obs=("logp_per_obs", "mean"),
            se_logp_per_obs=("logp_per_obs", lambda x: float(np.std(x, ddof=1) / np.sqrt(max(len(x), 1)))),
            total_logp=("logp", "sum"),
        )
        .reset_index()
    )
    return folds, summary


def diagnostics(model: IRTModel, data: ResultData) -> dict[str, object]:
    theta, b, alpha, c = model.unpacked()

    challenger_stats = pd.DataFrame({
        "Challenger": [f"{opt}@{dt}" for opt, dt in model.challenger_names],
        "Theta": theta,
        "Theta_SE": model.theta_se(),
    }).sort_values("Theta_SE")

    bench_stats = pd.DataFrame({
        "Benchmark": model.bench_names,
        "Alpha": alpha,
        "B": b,
        "C": c,
    }).sort_values("Alpha", ascending=False)

    challenger_idx, bench_idx, difficulty, outcome = (
        data.challenger_idx, data.bench_idx, data.difficulty, data.outcome,
    )
    logits = alpha[bench_idx] * (theta[challenger_idx] - b[bench_idx] - c[bench_idx] * difficulty)
    loglik = float(np.sum(outcome * logits - np.logaddexp(0, logits)))
    n_obs = len(outcome)
    n_params = len(model.params) if model.params is not None else 0

    return {
        "benchmarks": bench_stats,
        "challengers": challenger_stats,
        "fit": {
            "LogLikelihood": loglik,
            "Observations": n_obs,
            "Parameters": n_params,
            "AIC": 2 * n_params - 2 * loglik,
            "BIC": np.log(n_obs) * n_params - 2 * loglik,
        },
        "hyperparameters": {
            "sigma_theta": model.sigma_theta,
            "sigma_b": model.sigma_b,
            "sigma_log_alpha": model.sigma_log_alpha,
            "sigma_log_c": model.sigma_log_c,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Rank optimizers with a 2PL IRT model")
    parser.add_argument("file", nargs="*", default=["benchmark_results.jsonl"],
                        help="One or more jsonl files. With multiple files, benchmarks are "
                             "namespaced per source ('<tag>/<benchmark>') while optimizers are shared.")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Posterior draws for ranking and dominance.")
    parser.add_argument("--dominance", action="store_true",
                        help="Print pairwise posterior dominance matrix P(theta_i > theta_j).")
    parser.add_argument("--dim-cv", action="store_true",
                        help="K-fold MIRT cross-validation: K=1 vs K=2 held-out log-prob.")
    parser.add_argument("--dim-cv-folds", type=int, default=5)
    parser.add_argument("--dim-cv-kmax", type=int, default=2)
    parser.add_argument("--dim-cv-restarts", type=int, default=2)
    args = parser.parse_args()

    data = load_results(args.file[0]) if len(args.file) == 1 else load_multi_results(args.file)
    if data is None:
        print("No usable results found.")
        return

    model = fit(data, max_iter=args.max_iter, restarts=args.restarts)
    ranking = rank_optimizers(model, data, n_samples=args.samples)
    print(ranking.to_string(index=False))

    if args.dominance:
        print("\nPairwise posterior dominance P(row > col):")
        dom = pairwise_dominance(model, n_samples=args.samples)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(dom.to_string())

    if args.dim_cv:
        folds, summary = dimensionality_cv(
            data, K_max=args.dim_cv_kmax, n_folds=args.dim_cv_folds,
            max_iter=args.max_iter, n_restarts=args.dim_cv_restarts,
        )
        print(f"\nMIRT dimensionality CV ({args.dim_cv_folds}-fold, stratified by optimizer):")
        print(summary.to_string(index=False))
        if len(summary) >= 2:
            best = summary.loc[summary["mean_logp_per_obs"].idxmax(), "K"]
            print(f"  best K (mean held-out log-prob): {int(best)}")

    if not args.diagnostics:
        return
    info = diagnostics(model, data)
    print("\nBenchmark parameters (Alpha=discrimination, B=difficulty, C=slope):")
    print(info["benchmarks"].to_string(index=False))
    print("\nChallenger uncertainty:")
    print(info["challengers"].to_string(index=False))
    print("\nFit statistics:")
    for k, v in info["fit"].items():
        print(f"  {k}: {v}")
    print("\nEmpirical Bayes hyperparameters:")
    for k, v in info["hyperparameters"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
