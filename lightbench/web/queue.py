"""Queue abstractions for orchestrating evaluations."""

from __future__ import annotations

import atexit
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional

from .evaluation import BenchmarkEvaluationEngine


class SubmissionQueue:
    """Interface for enqueuing submissions for evaluation."""

    def enqueue(self, submission_id: int) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LocalThreadQueue(SubmissionQueue):
    """Simple thread-based queue useful for development and testing."""

    def __init__(self, engine: BenchmarkEvaluationEngine, max_workers: int = 2) -> None:
        self._engine = engine
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        atexit.register(self._executor.shutdown, wait=False)

    def enqueue(self, submission_id: int) -> str:
        job_id = uuid.uuid4().hex

        def _task() -> None:
            self._engine.evaluate_submission(submission_id, worker_id=job_id)

        future = self._executor.submit(_task)
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def status(self, job_id: str) -> Optional[str]:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            return None
        if future.done():
            return "completed" if future.exception() is None else "failed"
        return "running"
