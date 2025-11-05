from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

ArrayLike = Sequence[float] | NDArray[np.float64]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _make_pd(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    sym = (matrix + matrix.T) * 0.5
    eye = np.eye(sym.shape[0])
    jitter = 1e-8
    for _ in range(8):
        try:
            np.linalg.cholesky(sym)
        except np.linalg.LinAlgError:
            sym = sym + eye * jitter
            jitter *= 10.0
        else:
            return sym
    raise np.linalg.LinAlgError("Matrix not positive definite")


def _normalise_weights(
    raw: ArrayLike | Mapping[str, float] | None,
    axes: Sequence[str],
    *,
    kind: str = "direction",
) -> NDArray[np.float64]:
    if raw is None:
        weights = np.ones(len(axes), dtype=float)
    elif isinstance(raw, Mapping):
        weights = np.array([float(raw.get(axis, 0.0)) for axis in axes], dtype=float)
    else:
        weights = np.array(raw, dtype=float)
        if weights.shape != (len(axes),):
            raise ValueError("Weight vector must match number of axes")

    if not np.isfinite(weights).all():
        raise ValueError("Weights must be finite")

    if kind == "simplex":
        if np.allclose(weights, 0.0):
            weights = np.ones(len(axes), dtype=float)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("Weights must sum to a positive number")
        return weights / total

    norm = float(np.linalg.norm(weights))
    if norm == 0.0:
        weights = np.ones(len(axes), dtype=float)
        norm = float(np.linalg.norm(weights))
    return weights / norm


@dataclass(frozen=True)
class Item:
    dataset: str
    model_size: str
    budget: int
    loadings: ArrayLike
    difficulty: float
    cost: float
    tags: Tuple[str, ...] = field(default_factory=tuple)
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.cost <= 0:
            raise ValueError("Item cost must be positive")
        loadings = np.asarray(self.loadings, dtype=float)
        if loadings.ndim != 1 or loadings.shape[0] == 0:
            raise ValueError("Loadings must be a 1-D, non-empty vector")
        object.__setattr__(self, "loadings", loadings)

        if self.name is None:
            derived = f"{self.dataset}|{self.model_size}|{self.budget}"
            object.__setattr__(self, "name", derived)


@dataclass
class Observation:
    optimizer: str
    item: Item
    score: float
    threshold: float = 0.0
    weight: float = 1.0

    @classmethod
    def from_metric(
        cls,
        optimizer: str,
        item: Item,
        metric: float,
        reference_mean: float,
        reference_std: float,
        *,
        direction: str = "min",
        threshold: float = 0.0,
        weight: float = 1.0,
    ) -> "Observation":
        if reference_std <= 0.0:
            raise ValueError("reference_std must be positive")

        z = (metric - reference_mean) / reference_std
        if direction == "min":
            score = -z
        elif direction == "max":
            score = z
        else:
            raise ValueError("direction must be 'min' or 'max'")
        return cls(optimizer=optimizer, item=item, score=score, threshold=threshold, weight=weight)

    def binary_outcome(self) -> float:
        return 1.0 if self.score >= self.threshold else 0.0


class GaussianPosterior:
    def __init__(
        self,
        axes: Sequence[str],
        mean: Optional[ArrayLike] = None,
        covariance: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if not axes:
            raise ValueError("At least one axis must be specified")
        self.axes: Tuple[str, ...] = tuple(axes)
        dim = len(self.axes)

        self.mean = np.zeros(dim, dtype=float) if mean is None else np.asarray(mean, dtype=float)
        if self.mean.shape != (dim,):
            raise ValueError("Mean vector must align with axes")

        if covariance is None:
            covariance = np.eye(dim, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (dim, dim):
            raise ValueError("Covariance matrix must be square with size equal to number of axes")
        self.covariance = _make_pd(cov)

    @property
    def dim(self) -> int:
        return len(self.axes)

    def fisher_information(self, item: Item) -> NDArray[np.float64]:
        if item.loadings.shape[0] != self.dim:
            raise ValueError("Item loadings do not match posterior dimensionality")
        z = float(item.loadings @ self.mean - item.difficulty)
        p = _sigmoid(z)
        scale = p * (1.0 - p)
        return scale * np.outer(item.loadings, item.loadings)

    def expected_information(
        self,
        item: Item,
        *,
        weights: ArrayLike | Mapping[str, float] | None = None,
    ) -> float:
        fisher = self.fisher_information(item)
        if weights is None:
            return float(np.trace(fisher))
        direction = _normalise_weights(weights, self.axes, kind="direction")
        return float(direction @ fisher @ direction)

    def update(
        self,
        item: Item,
        outcome: float,
        *,
        weight: float = 1.0,
        max_iter: int = 25,
        tol: float = 1e-6,
    ) -> None:
        if not (0.0 <= outcome <= 1.0):
            raise ValueError("Outcome must be a probability in [0, 1]")
        if item.loadings.shape[0] != self.dim:
            raise ValueError("Item loadings do not match posterior dimensionality")
        if weight <= 0:
            raise ValueError("Observation weight must be positive")

        prior_mean = self.mean.copy()
        prior_cov = self.covariance
        prior_precision = np.linalg.inv(prior_cov)

        s = prior_mean.copy()
        a = item.loadings
        last_hessian = prior_precision.copy()
        for _ in range(max_iter):
            z = float(a @ s - item.difficulty)
            p = _sigmoid(z)
            grad = prior_precision @ (s - prior_mean) + (p - outcome) * a * weight
            hessian = prior_precision + weight * p * (1.0 - p) * np.outer(a, a)
            last_hessian = hessian
            try:
                delta = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                hessian = _make_pd(hessian)
                last_hessian = hessian
                delta = np.linalg.solve(hessian, grad)
            s -= delta
            if float(np.linalg.norm(delta)) < tol:
                break

        posterior_cov = np.linalg.inv(last_hessian)
        self.mean = s
        self.covariance = _make_pd(posterior_cov)

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        cov = _make_pd(self.covariance)
        return rng.multivariate_normal(self.mean, cov)


class CostAwareCAT:
    def __init__(self, posterior: GaussianPosterior) -> None:
        self.posterior = posterior

    def select_next(
        self,
        items: Iterable[Item],
        *,
        weights: ArrayLike | Mapping[str, float] | None = None,
        excluded: Optional[set[str]] = None,
        min_cost: float | None = None,
    ) -> Tuple[Item, float]:
        best_item: Optional[Item] = None
        best_ratio = float("-inf")
        excluded = excluded or set()

        for item in items:
            if item.name in excluded:
                continue
            if min_cost is not None and item.cost < min_cost:
                continue
            info = self.posterior.expected_information(item, weights=weights)
            if info <= 0.0:
                continue
            ratio = info / item.cost
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item

        if best_item is None:
            raise ValueError("No candidate yielded positive information")
        return best_item, best_ratio


class SubmissionState:
    def __init__(self, name: str, posterior: GaussianPosterior) -> None:
        self.name = name
        self.posterior = posterior
        self.observations: list[Observation] = []
        self.cost_spent: float = 0.0

    def record(self, observation: Observation) -> None:
        if observation.optimizer != self.name:
            raise ValueError("Observation optimizer does not match submission")
        outcome = observation.binary_outcome()
        self.posterior.update(observation.item, outcome, weight=observation.weight)
        self.observations.append(observation)
        self.cost_spent += observation.item.cost

    def expected_score(self, weights: ArrayLike | Mapping[str, float] | None = None) -> float:
        direction = _normalise_weights(weights, self.posterior.axes, kind="direction")
        return float(direction @ self.posterior.mean)

    def sample_skill(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.posterior.sample(rng)


class SuccessiveHalvingGate:
    def __init__(self, eta: float = 3.0, *, min_promotions: int = 1) -> None:
        if eta <= 1.0:
            raise ValueError("eta must be > 1")
        if min_promotions < 1:
            raise ValueError("At least one submission must be promoted")
        self.eta = float(eta)
        self.min_promotions = int(min_promotions)

    def promote(
        self,
        submissions: Sequence[SubmissionState],
        *,
        weights: ArrayLike | Mapping[str, float] | None = None,
    ) -> list[SubmissionState]:
        if not submissions:
            return []
        count = max(self.min_promotions, int(math.ceil(len(submissions) / self.eta)))
        scored = sorted(
            submissions,
            key=lambda sub: sub.expected_score(weights),
            reverse=True,
        )
        return scored[:count]


@dataclass(frozen=True)
class BestResult:
    optimizer: str
    score: float
    uncertainty: float
    weights: NDArray[np.float64]


class Leaderboard:
    def __init__(
        self,
        axes: Sequence[str],
        *,
        dirichlet_alpha: Optional[ArrayLike] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not axes:
            raise ValueError("Leaderboard requires at least one axis")
        self.axes: Tuple[str, ...] = tuple(axes)
        self._submissions: MutableMapping[str, SubmissionState] = {}
        if dirichlet_alpha is not None:
            alpha = _normalise_weights(dirichlet_alpha, self.axes, kind="simplex")
        else:
            alpha = np.ones(len(self.axes), dtype=float)
            alpha /= np.sum(alpha)
        self._dirichlet_alpha = alpha
        self._rng = rng or np.random.default_rng()

    def register(
        self,
        optimizer: str,
        *,
        prior_mean: Optional[ArrayLike] = None,
        prior_covariance: Optional[NDArray[np.float64]] = None,
    ) -> SubmissionState:
        if optimizer in self._submissions:
            raise ValueError(f"Optimizer '{optimizer}' is already registered")
        posterior = GaussianPosterior(self.axes, prior_mean, prior_covariance)
        state = SubmissionState(optimizer, posterior)
        self._submissions[optimizer] = state
        return state

    def ensure(self, optimizer: str) -> SubmissionState:
        return self._submissions.get(optimizer) or self.register(optimizer)

    def record(self, observation: Observation) -> None:
        state = self.ensure(observation.optimizer)
        state.record(observation)

    def best(self, **axis_flags: bool) -> BestResult:
        weights = {axis: 1.0 if axis_flags.get(axis, False) else 0.0 for axis in self.axes}
        direction = _normalise_weights(weights, self.axes, kind="direction")
        scored = [
            (
                state.posterior.mean @ direction,
                math.sqrt(direction @ state.posterior.covariance @ direction),
                name,
            )
            for name, state in self._submissions.items()
        ]
        if not scored:
            raise ValueError("Leaderboard is empty")
        scored.sort(reverse=True)
        best_score, uncertainty, name = scored[0]
        return BestResult(optimizer=name, score=float(best_score), uncertainty=float(uncertainty), weights=direction)

    def snapshot(self) -> Dict[str, Dict[str, NDArray[np.float64] | float]]:
        return {
            name: {
                "mean": state.posterior.mean.copy(),
                "covariance": state.posterior.covariance.copy(),
            }
            for name, state in self._submissions.items()
        }

    def frontier_statistics(
        self,
        *,
        num_weight_samples: int = 128,
        num_skill_samples: int = 8,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        if not self._submissions:
            return {}

        rng = np.random.default_rng(seed)
        alpha = self._dirichlet_alpha
        optimizers = list(self._submissions.items())
        names = [name for name, _ in optimizers]
        best_counts = {name: 0 for name in names}
        nondominated_counts = {name: 0 for name in names}
        total = 0

        for _ in range(num_weight_samples):
            weights = rng.dirichlet(alpha)
            direction = _normalise_weights(weights, self.axes, kind="direction")
            for _ in range(num_skill_samples):
                samples = {name: state.sample_skill(rng) for name, state in optimizers}
                projections = {name: float(direction @ sample) for name, sample in samples.items()}
                best_name = max(projections, key=projections.get)
                best_counts[best_name] += 1

                for name, skill in samples.items():
                    dominated = any(
                        np.all(other >= skill - 1e-9) and np.any(other > skill + 1e-9)
                        for other_name, other in samples.items()
                        if other_name != name
                    )
                    if not dominated:
                        nondominated_counts[name] += 1
                total += 1

        return {
            name: {
                "frontier_share": best_counts[name] / total,
                "prob_non_dominated": nondominated_counts[name] / total,
            }
            for name in names
        }

    def trueskill_summary(self) -> Dict[str, Tuple[float, float]]:
        uniform = _normalise_weights(np.ones(len(self.axes), dtype=float), self.axes, kind="direction")
        summary: Dict[str, Tuple[float, float]] = {}
        for name, state in self._submissions.items():
            mu = float(uniform @ state.posterior.mean)
            sigma = float(math.sqrt(uniform @ state.posterior.covariance @ uniform))
            summary[name] = (mu, sigma)
        return summary
