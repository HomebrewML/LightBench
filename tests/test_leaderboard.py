from __future__ import annotations

import numpy as np

from lightbench.leaderboard import (
    BestResult,
    CostAwareCAT,
    GaussianPosterior,
    Item,
    Leaderboard,
    Observation,
    SubmissionState,
    SuccessiveHalvingGate,
)


def test_item_validates_positive_cost_and_loadings() -> None:
    with np.testing.assert_raises(ValueError):
        Item("dataset", "small", 32, loadings=[], difficulty=0.0, cost=1.0)

    with np.testing.assert_raises(ValueError):
        Item("dataset", "small", 32, loadings=[1.0, 0.0], difficulty=0.0, cost=0.0)


def test_gaussian_posterior_update_moves_mean_towards_success() -> None:
    posterior = GaussianPosterior(["big_data", "dense"])
    submission = SubmissionState("adam", posterior)
    item = Item("cifar10", "small", 128, loadings=[1.0, 0.5], difficulty=0.0, cost=1.0)
    observation = Observation("adam", item, score=1.5)

    submission.record(observation)

    assert submission.posterior.mean[0] > 0.0
    assert submission.posterior.mean[1] > 0.0
    assert np.all(np.linalg.eigvals(submission.posterior.covariance) > 0)


def test_cost_aware_cat_prefers_information_per_cost() -> None:
    posterior = GaussianPosterior(["big_data"])
    cat = CostAwareCAT(posterior)
    item_high_info = Item("ds", "small", 32, loadings=[1.0], difficulty=0.0, cost=1.0)
    item_low_info = Item("ds", "small", 64, loadings=[1.0], difficulty=2.0, cost=0.1)

    selected, ratio = cat.select_next([item_high_info, item_low_info])
    assert selected is item_low_info
    assert ratio > 0

    # Ensure excluded items are skipped
    selected, _ = cat.select_next([item_high_info, item_low_info], excluded={item_low_info.name})
    assert selected is item_high_info


def test_leaderboard_best_respects_axis_flags() -> None:
    lb = Leaderboard(["big_data", "dense"])
    state_a = lb.register("optimizer_a")
    state_b = lb.register("optimizer_b")

    state_a.posterior.mean = np.array([2.0, -1.0])
    state_b.posterior.mean = np.array([1.0, 2.0])

    best_big_data: BestResult = lb.best(big_data=True, dense=False)
    assert best_big_data.optimizer == "optimizer_a"

    best_dense: BestResult = lb.best(big_data=False, dense=True)
    assert best_dense.optimizer == "optimizer_b"


def test_frontier_statistics_returns_probabilities() -> None:
    lb = Leaderboard(["big_data", "dense"], rng=np.random.default_rng(0))
    state_a = lb.register("opt_a")
    state_b = lb.register("opt_b")

    state_a.posterior.mean = np.array([1.5, 0.5])
    state_b.posterior.mean = np.array([0.8, 1.2])
    state_a.posterior.covariance = np.eye(2) * 0.05
    state_b.posterior.covariance = np.eye(2) * 0.05

    stats = lb.frontier_statistics(num_weight_samples=32, num_skill_samples=4, seed=0)

    assert set(stats.keys()) == {"opt_a", "opt_b"}
    for entry in stats.values():
        assert 0.0 <= entry["frontier_share"] <= 1.0
        assert 0.0 <= entry["prob_non_dominated"] <= 1.0


def test_successive_halving_promotes_top_scoring_submissions() -> None:
    states = [SubmissionState(f"opt{i}", GaussianPosterior(["axis"])) for i in range(4)]
    for idx, state in enumerate(states):
        state.posterior.mean = np.array([float(idx)])

    gate = SuccessiveHalvingGate(eta=2.0)
    promoted = gate.promote(states)

    # eta=2 promotes ceil(4/2)=2 submissions – the highest means
    assert [state.name for state in promoted] == ["opt3", "opt2"]
