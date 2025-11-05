"""Evaluation pipeline used by the web infrastructure."""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from lightbench.leaderboard import Leaderboard, Observation

from .items import DEFAULT_AXES, DEFAULT_ITEM_BANK, serialize_item
from .models import EvaluationSummary, Submission
from .storage import SubmissionRepository

LOGGER = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.utcnow().replace(tzinfo=None)


class OptimizerAdapter:
    """Runtime adapter that turns a submission entrypoint into callables."""

    def __init__(self, submission: Submission) -> None:
        self.submission = submission
        self._callable, self.metadata = self._load_callable()

    def _load_callable(self) -> tuple[Callable[[Any], float], Dict[str, Any]]:
        entry = self.submission.entrypoint
        if ":" not in entry:
            raise ValueError("Entrypoint must be of the form 'module:callable'. Received: " + entry)
        module_name, attr_path = entry.split(":", 1)
        module = importlib.import_module(module_name)
        target = module
        for attr in attr_path.split("."):
            target = getattr(target, attr)

        if callable(target):
            instantiated = target()
        else:
            instantiated = target

        metadata: Dict[str, Any] = {}

        evaluator: Optional[Callable[[Any], float]] = None
        if callable(instantiated):
            evaluator = instantiated  # type: ignore[assignment]
        elif hasattr(instantiated, "evaluate_item") and callable(instantiated.evaluate_item):
            evaluator = instantiated.evaluate_item  # type: ignore[assignment]

        if evaluator is None:
            raise TypeError("Entry point must be callable or expose 'evaluate_item' method")

        if hasattr(instantiated, "metadata"):
            meta_attr = getattr(instantiated, "metadata")
            try:
                metadata_candidate = meta_attr() if callable(meta_attr) else meta_attr
                if isinstance(metadata_candidate, dict):
                    metadata = {str(k): metadata_candidate[k] for k in metadata_candidate}
            except Exception:  # noqa: BLE001
                LOGGER.warning("Failed to collect metadata from submission %s", self.submission.id)

        return evaluator, metadata

    def evaluate(self, item) -> float:
        return float(self._callable(item))


class EvaluationRunner:
    """Executes the expensive benchmark workload and returns raw metrics."""

    def __init__(self, *, items: Iterable = DEFAULT_ITEM_BANK) -> None:
        self.items = list(items)

    def run(
        self,
        submission: Submission,
        *,
        progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        adapter = OptimizerAdapter(submission)
        metadata = dict(adapter.metadata)
        metrics: Dict[str, float] = {}

        for item in self.items:
            metric = adapter.evaluate(item)
            if not math.isfinite(metric):
                raise ValueError(f"Non-finite metric returned for {item.name}")
            metrics[item.name] = float(metric)
            if progress:
                progress(item.name)

        return metrics, metadata


class BenchmarkEvaluationEngine:
    """Coordinates leaderboard updates and result persistence."""

    def __init__(
        self,
        repository: SubmissionRepository,
        *,
        axes: Sequence[str] = DEFAULT_AXES,
        items: Iterable = DEFAULT_ITEM_BANK,
        leaderboard_state_path: Path | None = None,
    ) -> None:
        self.repository = repository
        self.axes = tuple(axes)
        self.items = list(items)
        self._items_by_name = {item.name: item for item in self.items}
        self.leaderboard_state_path = leaderboard_state_path or self._default_leaderboard_state_path()
        self.leaderboard_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._leaderboard = self._load_leaderboard()

    def evaluate_submission(
        self,
        submission_id: int,
        *,
        runner: Optional[EvaluationRunner] = None,
        worker_id: str = "server",
    ) -> EvaluationSummary:
        runner = runner or EvaluationRunner(items=self.items)
        submission = self.repository.claim_submission(submission_id, worker_id)
        LOGGER.info("Evaluating submission %s locally", submission.short_identifier)
        try:
            metrics, metadata = runner.run(submission)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Local evaluation failed for %s", submission.short_identifier)
            self.repository.mark_failed(submission_id, worker_id, str(exc))
            raise
        return self.record_results(submission_id, metrics, metadata, worker_id=worker_id)

    def record_results(
        self,
        submission_id: int,
        metrics: Mapping[str, float],
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        worker_id: str = "unknown",
    ) -> EvaluationSummary:
        submission = self.repository.get_submission(submission_id)
        submission_key = submission.short_identifier
        state = self._leaderboard.ensure(submission_key)

        for item_name, value in metrics.items():
            item = self._items_by_name.get(item_name)
            if item is None:
                LOGGER.warning("Received metric for unknown item '%s'", item_name)
                continue
            observation = Observation(
                optimizer=submission_key,
                item=item,
                score=float(value),
                threshold=item.difficulty,
            )
            state.record(observation)

        snapshot = self._leaderboard.snapshot()
        summary = EvaluationSummary(
            submission_id=submission_id,
            metrics=dict(metrics),
            metadata=dict(metadata or {}),
            leaderboard_snapshot={k: _numpy_to_serialisable(v) for k, v in snapshot.items()},
            completed_at=_now(),
        )

        self.repository.store_evaluation_summary(summary)
        self.repository.mark_completed(submission_id, worker_id)
        self._persist_leaderboard(self._leaderboard)
        LOGGER.info("Recorded results for %s from worker %s", submission.short_identifier, worker_id)
        return summary

    def fail_submission(self, submission_id: int, worker_id: str, error: str) -> None:
        LOGGER.error("Worker %s reported failure for submission %s: %s", worker_id, submission_id, error)
        self.repository.mark_failed(submission_id, worker_id, error)

    def serialized_items(self) -> list[Dict[str, Any]]:
        return [serialize_item(item) for item in self.items]

    def _default_leaderboard_state_path(self) -> Path:
        env = os.environ.get("LIGHTBENCH_LEADERBOARD_STATE")
        if env:
            path = Path(env).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return Path.cwd() / "state" / "leaderboard.json"

    def _load_leaderboard(self) -> Leaderboard:
        leaderboard = Leaderboard(self.axes)
        if not self.leaderboard_state_path.exists():
            return leaderboard

        try:
            payload = json.loads(self.leaderboard_state_path.read_text())
        except json.JSONDecodeError:
            LOGGER.warning("Leaderboard state corrupted; starting fresh")
            return leaderboard

        for optimizer, stats in payload.items():
            state = leaderboard.register(optimizer)
            state.posterior.mean = np.asarray(stats["mean"], dtype=float)
            state.posterior.covariance = np.asarray(stats["covariance"], dtype=float)
        return leaderboard

    def _persist_leaderboard(self, leaderboard: Leaderboard) -> None:
        snapshot = leaderboard.snapshot()
        serialisable = {k: _numpy_to_serialisable(v) for k, v in snapshot.items()}
        self.leaderboard_state_path.write_text(json.dumps(serialisable, indent=2))

    def leaderboard_snapshot(self) -> Dict[str, Dict[str, Any]]:
        snapshot = self._leaderboard.snapshot()
        return {k: _numpy_to_serialisable(v) for k, v in snapshot.items()}

    def leaderboard_frontier(self) -> Dict[str, Dict[str, float]]:
        return self._leaderboard.frontier_statistics()


def _numpy_to_serialisable(payload: Dict[str, Any]) -> Dict[str, Any]:
    serialised: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            serialised[key] = value.tolist()
        else:
            serialised[key] = value
    return serialised
