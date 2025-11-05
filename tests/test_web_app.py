from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lightbench.web.app import create_app
from lightbench.web.evaluation import BenchmarkEvaluationEngine
from lightbench.web.models import SubmissionCreate, SubmissionStatus
from lightbench.web.queue import SubmissionQueue
from lightbench.web.storage import SubmissionRepository


def make_repository(tmp_path: Path) -> SubmissionRepository:
    return SubmissionRepository(tmp_path / "submissions.db")


def test_submission_repository_roundtrip(tmp_path: Path) -> None:
    repo = make_repository(tmp_path)
    payload = SubmissionCreate(
        optimizer_name="opt",
        contact_email="opt@example.com",
        contact_name="Opt",
        repository_url="https://example.com/repo.git",
        entrypoint="tests.dummy_optimizer:build_optimizer",
        description="test",
    )
    created = repo.create_submission(payload)

    fetched = repo.get_submission(created.id)
    assert fetched.optimizer_name == "opt"
    assert fetched.status is SubmissionStatus.QUEUED

    claimed = repo.claim_next_submission("worker", lease_seconds=60)
    assert claimed is not None
    assert claimed.status is SubmissionStatus.RUNNING
    repo.mark_completed(claimed.id, "worker")
    assert repo.get_submission(created.id).status is SubmissionStatus.COMPLETED


def test_evaluation_updates_leaderboard(tmp_path: Path) -> None:
    repo = make_repository(tmp_path)
    payload = SubmissionCreate(
        optimizer_name="dummy",
        contact_email="dummy@example.com",
        contact_name="Tester",
        repository_url="https://example.com/repo.git",
        entrypoint="tests.dummy_optimizer:build_optimizer",
    )
    submission = repo.create_submission(payload)
    engine = BenchmarkEvaluationEngine(repo, leaderboard_state_path=tmp_path / "leaderboard.json")

    summary = engine.evaluate_submission(submission.id)

    updated = repo.get_submission(submission.id)
    assert updated.status is SubmissionStatus.COMPLETED
    assert summary.metrics
    assert len(engine.leaderboard_snapshot()) >= 1


class ImmediateQueue(SubmissionQueue):
    def __init__(self, engine: BenchmarkEvaluationEngine) -> None:
        self.engine = engine

    def enqueue(self, submission_id: int) -> str:
        self.engine.evaluate_submission(submission_id)
        return "sync-job"

    def status(self, job_id: str) -> str:
        return "completed"


def test_fastapi_submission_flow(tmp_path: Path) -> None:
    repo = make_repository(tmp_path)
    engine = BenchmarkEvaluationEngine(repo, leaderboard_state_path=tmp_path / "leaderboard.json")
    queue = ImmediateQueue(engine)
    client = TestClient(create_app(repository=repo, engine=engine, queue=queue))

    payload = {
        "optimizer_name": "api-opt",
        "contact_email": "user@example.com",
        "contact_name": "User",
        "repository_url": "https://example.com/repo.git",
        "entrypoint": "tests.dummy_optimizer:build_optimizer",
        "description": "via api",
    }

    response = client.post("/api/submissions", json=payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["status"] == SubmissionStatus.COMPLETED.value

    page = client.get("/")
    assert page.status_code == 200

    detail = client.get(f"/api/submissions/{data['id']}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["status"] == SubmissionStatus.COMPLETED.value
    assert detail_payload["evaluation_summary"]["metrics"]

    job_status = client.get("/api/jobs/sync-job")
    assert job_status.status_code == 200
    assert job_status.json()["status"] == "completed"


def test_worker_api_claim_and_complete(tmp_path: Path) -> None:
    repo = make_repository(tmp_path)
    engine = BenchmarkEvaluationEngine(repo, leaderboard_state_path=tmp_path / "leaderboard.json")
    token = "secret-token"
    app = create_app(repository=repo, engine=engine, auto_start_local_queue=False, worker_token=token)
    client = TestClient(app)

    submission = repo.create_submission(
        SubmissionCreate(
            optimizer_name="worker-opt",
            contact_email="worker@example.com",
            contact_name="Worker",
            repository_url="https://example.com/repo.git",
            entrypoint="tests.dummy_optimizer:build_optimizer",
        )
    )

    claim = client.post(
        "/api/worker/claim",
        headers={"x-worker-token": token},
        json={"worker_id": "w1", "lease_seconds": 600},
    )
    assert claim.status_code == 200
    job = claim.json()
    assert job["submission"]["status"] == SubmissionStatus.RUNNING.value

    metrics = {item["name"]: 1.0 for item in job["items"]}
    complete = client.post(
        f"/api/worker/{submission.id}/complete",
        headers={"x-worker-token": token},
        json={"worker_id": "w1", "metrics": metrics, "metadata": {"note": "ok"}},
    )
    assert complete.status_code == 200
    payload = complete.json()
    assert payload["submission"]["status"] == SubmissionStatus.COMPLETED.value
    summary = repo.load_evaluation_summary(submission.id)
    assert summary is not None
    assert summary.metadata["note"] == "ok"


def test_worker_claim_none_when_queue_empty(tmp_path: Path) -> None:
    repo = make_repository(tmp_path)
    engine = BenchmarkEvaluationEngine(repo, leaderboard_state_path=tmp_path / "leaderboard.json")
    token = "secret"
    client = TestClient(create_app(repository=repo, engine=engine, auto_start_local_queue=False, worker_token=token))

    resp = client.post(
        "/api/worker/claim",
        headers={"x-worker-token": token},
        json={"worker_id": "worker", "lease_seconds": 60},
    )
    assert resp.status_code == 204
