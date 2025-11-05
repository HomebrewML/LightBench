"""Persistence helpers for the LightBench web service."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional

from .models import EvaluationSummary, Submission, SubmissionCreate, SubmissionStatus


def _now() -> datetime:
    return datetime.utcnow().replace(tzinfo=None)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


class SubmissionRepository:
    """SQLite-backed submission store."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        finally:
            conn.close()

    def _initialise(self) -> None:
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimizer_name TEXT NOT NULL,
                    contact_email TEXT NOT NULL,
                    contact_name TEXT NOT NULL,
                    repository_url TEXT NOT NULL,
                    entrypoint TEXT NOT NULL,
                    description TEXT NOT NULL,
                    resource_limits TEXT NOT NULL,
                    extra_metadata TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    modal_task_id TEXT,
                    last_error TEXT,
                    claimed_by TEXT,
                    claimed_at TEXT,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_summaries (
                    submission_id INTEGER PRIMARY KEY,
                    metrics TEXT NOT NULL,
                    leaderboard_snapshot TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(submission_id) REFERENCES submissions(id) ON DELETE CASCADE
                )
                """
            )
            for statement in (
                "ALTER TABLE submissions ADD COLUMN claimed_by TEXT",
                "ALTER TABLE submissions ADD COLUMN claimed_at TEXT",
                "ALTER TABLE submissions ADD COLUMN started_at TEXT",
                "ALTER TABLE submissions ADD COLUMN completed_at TEXT",
                "ALTER TABLE submissions ADD COLUMN modal_task_id TEXT",
                "ALTER TABLE submissions ADD COLUMN last_error TEXT",
                "ALTER TABLE evaluation_summaries ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'",
            ):
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError:
                    continue

    def create_submission(self, payload: SubmissionCreate) -> Submission:
        created_at = _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            cursor = conn.execute(
                """
                INSERT INTO submissions (
                    optimizer_name, contact_email, contact_name, repository_url, entrypoint, description,
                    resource_limits, extra_metadata, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.optimizer_name.strip(),
                    payload.contact_email.strip(),
                    payload.contact_name.strip(),
                    payload.repository_url.strip(),
                    payload.entrypoint.strip(),
                    payload.description.strip(),
                    json.dumps(payload.resource_limits or {}),
                    json.dumps(payload.extra_metadata or {}),
                    SubmissionStatus.QUEUED.value,
                    created_at.isoformat(),
                    created_at.isoformat(),
                ),
            )
            submission_id = int(cursor.lastrowid)
        return self.get_submission(submission_id)

    def get_submission(self, submission_id: int) -> Submission:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        if row is None:
            raise KeyError(f"No submission with id={submission_id}")
        return self._row_to_submission(row)

    def list_submissions(self, limit: Optional[int] = None) -> List[Submission]:
        query = "SELECT * FROM submissions ORDER BY created_at DESC"
        params: tuple[object, ...]
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        else:
            params = ()
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_submission(row) for row in rows]

    def update_status(
        self,
        submission_id: int,
        status: SubmissionStatus,
        *,
        modal_task_id: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> Submission:
        updated_at = _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                UPDATE submissions
                SET status = ?, updated_at = ?, modal_task_id = COALESCE(?, modal_task_id), last_error = ?
                WHERE id = ?
                """,
                (status.value, updated_at.isoformat(), modal_task_id, last_error, submission_id),
            )
        return self.get_submission(submission_id)

    def update_submission(self, submission: Submission) -> None:
        updated_at = _now()
        payload = asdict(submission)
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                UPDATE submissions SET
                    optimizer_name = ?,
                    contact_email = ?,
                    contact_name = ?,
                    repository_url = ?,
                    entrypoint = ?,
                    description = ?,
                    resource_limits = ?,
                    extra_metadata = ?,
                    status = ?,
                    updated_at = ?,
                    modal_task_id = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (
                    payload["optimizer_name"],
                    payload["contact_email"],
                    payload["contact_name"],
                    payload["repository_url"],
                    payload["entrypoint"],
                    payload["description"],
                    json.dumps(payload["resource_limits"]),
                    json.dumps(payload["extra_metadata"]),
                    submission.status.value,
                    updated_at.isoformat(),
                    payload["modal_task_id"],
                    payload["last_error"],
                    payload["id"],
                ),
            )

    def update_modal_task(self, submission_id: int, modal_task_id: str) -> None:
        updated_at = _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                UPDATE submissions
                SET modal_task_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (modal_task_id, updated_at.isoformat(), submission_id),
            )

    def claim_next_submission(
        self,
        worker_id: str,
        *,
        lease_seconds: int = 3600,
    ) -> Optional[Submission]:
        claim_time = _now()
        lease_cutoff = (claim_time - timedelta(seconds=lease_seconds)).isoformat()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM submissions
                WHERE status = ?
                  AND (claimed_at IS NULL OR claimed_at <= ?)
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (SubmissionStatus.QUEUED.value, lease_cutoff),
            ).fetchone()
            if row is None:
                return None
            submission_id = int(row["id"])
            job_id = uuid.uuid4().hex
            conn.execute(
                """
                UPDATE submissions
                SET status = ?, claimed_by = ?, claimed_at = ?, started_at = COALESCE(started_at, ?),
                    updated_at = ?, modal_task_id = ?, last_error = NULL
                WHERE id = ?
                """,
                (
                    SubmissionStatus.RUNNING.value,
                    worker_id,
                    claim_time.isoformat(),
                    claim_time.isoformat(),
                    claim_time.isoformat(),
                    job_id,
                    submission_id,
                ),
            )
        submission = self.get_submission(submission_id)
        return submission

    def claim_submission(self, submission_id: int, worker_id: str) -> Submission:
        claim_time = _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, claimed_by FROM submissions WHERE id = ?",
                (submission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"No submission with id={submission_id}")
            claimed_by = row["claimed_by"]
            status = SubmissionStatus(row["status"])
            if status not in {SubmissionStatus.QUEUED, SubmissionStatus.RUNNING}:
                raise ValueError("Submission is not claimable")
            if claimed_by not in {None, worker_id}:
                raise ValueError("Submission already claimed by another worker")
            job_id = uuid.uuid4().hex
            conn.execute(
                """
                UPDATE submissions
                SET status = ?, claimed_by = ?, claimed_at = ?, started_at = COALESCE(started_at, ?),
                    updated_at = ?, modal_task_id = ?, last_error = NULL
                WHERE id = ?
                """,
                (
                    SubmissionStatus.RUNNING.value,
                    worker_id,
                    claim_time.isoformat(),
                    claim_time.isoformat(),
                    claim_time.isoformat(),
                    job_id,
                    submission_id,
                ),
            )
        return self.get_submission(submission_id)

    def heartbeat_submission(self, submission_id: int, worker_id: str) -> None:
        heartbeat_time = _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                UPDATE submissions
                SET claimed_at = ?, updated_at = ?
                WHERE id = ? AND claimed_by = ? AND status = ?
                """,
                (
                    heartbeat_time.isoformat(),
                    heartbeat_time.isoformat(),
                    submission_id,
                    worker_id,
                    SubmissionStatus.RUNNING.value,
                ),
            )

    def mark_completed(self, submission_id: int, worker_id: str) -> Submission:
        completed_at = _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                UPDATE submissions
                SET status = ?, updated_at = ?, completed_at = ?, claimed_by = NULL,
                    claimed_at = NULL, modal_task_id = NULL
                WHERE id = ? AND (claimed_by IS NULL OR claimed_by = ?)
                """,
                (
                    SubmissionStatus.COMPLETED.value,
                    completed_at.isoformat(),
                    completed_at.isoformat(),
                    submission_id,
                    worker_id,
                ),
            )
        return self.get_submission(submission_id)

    def mark_failed(self, submission_id: int, worker_id: str, error: str) -> Submission:
        failed_at = _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                UPDATE submissions
                SET status = ?, updated_at = ?, last_error = ?, claimed_by = NULL,
                    claimed_at = NULL, modal_task_id = NULL
                WHERE id = ? AND (claimed_by IS NULL OR claimed_by = ?)
                """,
                (
                    SubmissionStatus.FAILED.value,
                    failed_at.isoformat(),
                    error,
                    submission_id,
                    worker_id,
                ),
            )
        return self.get_submission(submission_id)

    def release_submission(self, submission_id: int) -> None:
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                UPDATE submissions
                SET status = ?, claimed_by = NULL, claimed_at = NULL, modal_task_id = NULL
                WHERE id = ?
                """,
                (SubmissionStatus.QUEUED.value, submission_id),
            )

    def store_evaluation_summary(self, summary: EvaluationSummary) -> None:
        completed_at = summary.completed_at or _now()
        with self._connection() as conn, conn:  # type: ignore[arg-type]
            conn.execute(
                """
                INSERT INTO evaluation_summaries (submission_id, metrics, leaderboard_snapshot, metadata, completed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(submission_id) DO UPDATE SET
                    metrics = excluded.metrics,
                    leaderboard_snapshot = excluded.leaderboard_snapshot,
                    metadata = excluded.metadata,
                    completed_at = excluded.completed_at
                """,
                (
                    summary.submission_id,
                    json.dumps(summary.metrics),
                    json.dumps(summary.leaderboard_snapshot),
                    json.dumps(summary.metadata),
                    completed_at.isoformat(),
                ),
            )

    def load_evaluation_summary(self, submission_id: int) -> Optional[EvaluationSummary]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_summaries WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        if row is None:
            return None
        completed_at = datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return EvaluationSummary(
            submission_id=int(row["submission_id"]),
            metrics=json.loads(row["metrics"]),
            leaderboard_snapshot=json.loads(row["leaderboard_snapshot"]),
            metadata=metadata,
            completed_at=completed_at,
        )

    def _row_to_submission(self, row: sqlite3.Row) -> Submission:
        keys = row.keys()
        modal_task_id = row["modal_task_id"] if "modal_task_id" in keys else None
        return Submission(
            id=int(row["id"]),
            optimizer_name=row["optimizer_name"],
            contact_email=row["contact_email"],
            contact_name=row["contact_name"],
            repository_url=row["repository_url"],
            entrypoint=row["entrypoint"],
            description=row["description"],
            resource_limits=json.loads(row["resource_limits"] or "{}"),
            extra_metadata=json.loads(row["extra_metadata"] or "{}"),
            status=SubmissionStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            modal_task_id=modal_task_id,
            last_error=row["last_error"] if "last_error" in keys else None,
            claimed_by=row["claimed_by"] if "claimed_by" in keys else None,
            claimed_at=_parse_datetime(row["claimed_at"]) if "claimed_at" in keys else None,
            started_at=_parse_datetime(row["started_at"]) if "started_at" in keys else None,
            completed_at=_parse_datetime(row["completed_at"]) if "completed_at" in keys else None,
        )


def default_repository_path() -> Path:
    env_value = os.environ.get("LIGHTBENCH_DB_PATH")
    if env_value:
        target = Path(env_value).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    env_path = Path.cwd() / "state"
    env_path.mkdir(parents=True, exist_ok=True)
    return env_path / "lightbench_submissions.db"


def build_default_repository() -> SubmissionRepository:
    return SubmissionRepository(default_repository_path())
