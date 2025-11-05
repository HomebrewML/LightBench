from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

from fastapi import FastAPI, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, HttpUrl

from .evaluation import BenchmarkEvaluationEngine
from .models import Submission, SubmissionCreate, SubmissionStatus
from .queue import LocalThreadQueue, SubmissionQueue
from .storage import SubmissionRepository, build_default_repository


class SubmissionPayload(BaseModel):
    optimizer_name: str
    contact_email: EmailStr
    contact_name: str
    repository_url: HttpUrl
    entrypoint: str
    description: str = ""
    resource_limits: Dict[str, Any] | None = None
    extra_metadata: Dict[str, Any] | None = None


class WorkerClaimPayload(BaseModel):
    worker_id: str
    lease_seconds: int | None = None


class WorkerResultPayload(BaseModel):
    worker_id: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] | None = None


class WorkerHeartbeatPayload(BaseModel):
    worker_id: str


class WorkerFailurePayload(BaseModel):
    worker_id: str
    error: str


def _serialise_submission(repo: SubmissionRepository, submission: Submission) -> Dict[str, Any]:
    payload = submission.to_dict()
    summary = repo.load_evaluation_summary(submission.id)
    if summary:
        payload["evaluation_summary"] = {
            "metrics": summary.metrics,
            "metadata": summary.metadata,
            "completed_at": summary.completed_at.isoformat() + "Z" if summary.completed_at else None,
        }
    else:
        payload["evaluation_summary"] = None
    return payload


def _list_context(repo: SubmissionRepository, limit: Optional[int] = 50) -> Iterable[Dict[str, Any]]:
    for submission in repo.list_submissions(limit=limit):
        yield {
            "submission": submission,
            "summary": repo.load_evaluation_summary(submission.id),
        }


def create_app(
    *,
    repository: SubmissionRepository | None = None,
    engine: BenchmarkEvaluationEngine | None = None,
    queue: SubmissionQueue | None = None,
    auto_start_local_queue: bool = False,
    worker_token: Optional[str] = None,
) -> FastAPI:
    repo = repository or build_default_repository()
    eval_engine = engine or BenchmarkEvaluationEngine(repo)
    job_queue = queue or (LocalThreadQueue(eval_engine) if auto_start_local_queue else None)
    worker_token = worker_token or os.environ.get("LIGHTBENCH_WORKER_TOKEN")

    app = FastAPI(title="LightBench")

    def _require_worker_token(request: Request) -> None:
        if worker_token is None:
            return
        if request.headers.get("x-worker-token") != worker_token:
            raise HTTPException(status_code=403, detail="Invalid worker token")

    def _enqueue_if_possible(submission: Submission) -> Submission:
        if job_queue is None:
            return submission
        job_id = job_queue.enqueue(submission.id)
        repo.update_modal_task(submission.id, job_id)
        return repo.get_submission(submission.id)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        items = list(_list_context(repo))
        if not items:
            body = "<p>No submissions yet.</p>"
        else:
            rows = []
            for entry in items:
                submission = entry["submission"]
                summary = entry["summary"]
                summary_hint = " – results available" if summary else ""
                rows.append(f"<li>{submission.optimizer_name} ({submission.status.value}){summary_hint}</li>")
            body = "<ul>" + "".join(rows) + "</ul>"
        html = "<html><body><h1>LightBench</h1>{}</body></html>".format(body)
        return HTMLResponse(html)

    @app.post("/submit")
    async def submit_form(
        request: Request,
        optimizer_name: str = Form(...),
        contact_email: str = Form(...),
        contact_name: str = Form(...),
        repository_url: str = Form(...),
        entrypoint: str = Form(...),
        description: str = Form(""),
        resource_limits_json: str = Form(""),
        extra_metadata_json: str = Form(""),
    ) -> RedirectResponse:
        try:
            resource_limits = json.loads(resource_limits_json) if resource_limits_json else {}
            extra_metadata = json.loads(extra_metadata_json) if extra_metadata_json else {}
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

        submission = repo.create_submission(
            SubmissionCreate(
                optimizer_name=optimizer_name,
                contact_email=contact_email,
                contact_name=contact_name,
                repository_url=repository_url,
                entrypoint=entrypoint,
                description=description,
                resource_limits=resource_limits,
                extra_metadata=extra_metadata,
            )
        )
        submission = _enqueue_if_possible(submission)
        url = request.url_for("submission_detail", submission_id=submission.id)
        return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/submissions/{submission_id}", response_class=HTMLResponse)
    async def submission_detail(submission_id: int) -> HTMLResponse:
        try:
            submission = repo.get_submission(submission_id)
        except KeyError as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail="Submission not found") from exc
        summary = repo.load_evaluation_summary(submission_id)
        metrics = summary.metrics if summary else {}
        html = "<html><body><h1>{name}</h1><p>Status: {status}</p><pre>{metrics}</pre></body></html>".format(
            name=submission.optimizer_name,
            status=submission.status.value,
            metrics=json.dumps(metrics, indent=2),
        )
        return HTMLResponse(html)

    @app.get("/api/submissions")
    async def api_list_submissions() -> JSONResponse:
        payload = [_serialise_submission(repo, sub) for sub in repo.list_submissions()]
        return JSONResponse(payload)

    @app.post("/api/submissions", status_code=status.HTTP_201_CREATED)
    async def api_create_submission(payload: SubmissionPayload) -> JSONResponse:
        submission = repo.create_submission(
            SubmissionCreate(
                optimizer_name=payload.optimizer_name,
                contact_email=payload.contact_email,
                contact_name=payload.contact_name,
                repository_url=str(payload.repository_url),
                entrypoint=payload.entrypoint,
                description=payload.description,
                resource_limits=payload.resource_limits,
                extra_metadata=payload.extra_metadata,
            )
        )
        submission = _enqueue_if_possible(submission)
        return JSONResponse(_serialise_submission(repo, submission), status_code=status.HTTP_201_CREATED)

    @app.get("/api/submissions/{submission_id}")
    async def api_submission_detail(submission_id: int) -> JSONResponse:
        try:
            submission = repo.get_submission(submission_id)
        except KeyError as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail="Submission not found") from exc
        return JSONResponse(_serialise_submission(repo, submission))

    @app.get("/api/jobs/{job_id}")
    async def api_job_status(job_id: str) -> JSONResponse:
        if job_queue is None or not hasattr(job_queue, "status"):
            raise HTTPException(status_code=404, detail="Job queue disabled")
        status_text = job_queue.status(job_id)  # type: ignore[call-arg]
        if status_text is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse({"job_id": job_id, "status": status_text})

    @app.post("/api/worker/claim")
    async def api_worker_claim(payload: WorkerClaimPayload, request: Request) -> Response:
        _require_worker_token(request)
        lease = payload.lease_seconds or 3600
        submission = repo.claim_next_submission(payload.worker_id, lease_seconds=lease)
        if submission is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        body = {
            "job_id": submission.modal_task_id,
            "submission": _serialise_submission(repo, submission),
            "items": eval_engine.serialized_items(),
            "axes": list(eval_engine.axes),
        }
        return JSONResponse(body)

    @app.post("/api/worker/{submission_id}/heartbeat")
    async def api_worker_heartbeat(
        submission_id: int, payload: WorkerHeartbeatPayload, request: Request
    ) -> JSONResponse:
        _require_worker_token(request)
        repo.heartbeat_submission(submission_id, payload.worker_id)
        return JSONResponse({"status": "ok"})

    @app.post("/api/worker/{submission_id}/complete")
    async def api_worker_complete(
        submission_id: int,
        payload: WorkerResultPayload,
        request: Request,
    ) -> JSONResponse:
        _require_worker_token(request)
        submission = repo.get_submission(submission_id)
        if submission.status == SubmissionStatus.COMPLETED:
            summary = repo.load_evaluation_summary(submission_id)
            return JSONResponse({
                "submission": _serialise_submission(repo, submission),
                "summary": {
                    "metrics": summary.metrics,
                    "metadata": summary.metadata,
                    "completed_at": summary.completed_at.isoformat() + "Z"
                    if summary and summary.completed_at
                    else None,
                }
                if summary
                else None,
            })
        if submission.claimed_by not in {None, payload.worker_id}:
            raise HTTPException(status_code=409, detail="Submission claimed by another worker")
        summary = eval_engine.record_results(
            submission_id,
            payload.metrics,
            payload.metadata,
            worker_id=payload.worker_id,
        )
        submission = repo.get_submission(submission_id)
        return JSONResponse({
            "submission": _serialise_submission(repo, submission),
            "summary": {
                "metrics": summary.metrics,
                "metadata": summary.metadata,
                "completed_at": summary.completed_at.isoformat() + "Z" if summary.completed_at else None,
            },
        })

    @app.post("/api/worker/{submission_id}/fail")
    async def api_worker_fail(
        submission_id: int,
        payload: WorkerFailurePayload,
        request: Request,
    ) -> JSONResponse:
        _require_worker_token(request)
        eval_engine.fail_submission(submission_id, payload.worker_id, payload.error)
        return JSONResponse({"status": "failed"})

    return app
