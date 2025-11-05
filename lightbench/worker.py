"""Command-line worker for executing LightBench submissions on remote GPUs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx
import typer

from lightbench.web.evaluation import EvaluationRunner
from lightbench.web.items import DEFAULT_ITEM_BANK, deserialize_item
from lightbench.web.models import Submission

app = typer.Typer(add_completion=False)


def _run_command(args: list[str], cwd: Optional[Path] = None) -> None:
    typer.echo(f"$ {' '.join(args)}", err=True)
    subprocess.run(args, cwd=cwd, check=True)


def _prepare_environment(submission: Submission, workspace: Path) -> None:
    repo_url = submission.repository_url.strip()
    if not repo_url:
        return
    repo_dir = workspace / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    if repo_url.endswith(".git"):
        _run_command(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])
        sys.path.insert(0, str(repo_dir))
        _run_command([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)])
    else:
        _run_command([sys.executable, "-m", "pip", "install", repo_url])


def _post_json(client: httpx.Client, url: str, payload: dict, *, timeout: float = 30.0) -> httpx.Response:
    response = client.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response


@app.command()
def run(
    api_url: str = typer.Option(..., help="Base URL of the LightBench control plane"),
    worker_id: Optional[str] = typer.Option(None, help="Unique identifier for this worker"),
    token: Optional[str] = typer.Option(None, help="Shared secret expected by the server"),
    poll_interval: int = typer.Option(30, help="Seconds between queue polls when idle"),
    lease_seconds: int = typer.Option(1800, help="Lease duration requested when claiming jobs"),
    workspace: Path = typer.Option(Path("/tmp/lightbench-worker"), help="Directory for temporary job data"),
    once: bool = typer.Option(False, help="Exit after processing a single job"),
) -> None:
    worker_id = worker_id or os.environ.get("LIGHTBENCH_WORKER_ID") or f"worker-{uuid4().hex[:8]}"
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    headers = {}
    token = token or os.environ.get("LIGHTBENCH_WORKER_TOKEN")
    if token:
        headers["x-worker-token"] = token

    api_url = os.environ.get("LIGHTBENCH_CONTROL_URL", api_url)
    if api_url is None:
        raise typer.BadParameter("--api-url must be provided or LIGHTBENCH_CONTROL_URL must be set")

    typer.echo(f"LightBench worker {worker_id} polling {api_url}")

    with httpx.Client(base_url=api_url, headers=headers, timeout=60.0) as client:
        while True:
            try:
                response = client.post(
                    "/api/worker/claim",
                    json={"worker_id": worker_id, "lease_seconds": lease_seconds},
                    timeout=30.0,
                )
            except httpx.HTTPError as exc:  # noqa: BLE001
                typer.echo(f"Queue request failed: {exc}", err=True)
                time.sleep(poll_interval)
                if once:
                    break
                continue

            if response.status_code == 204:
                time.sleep(poll_interval)
                if once:
                    break
                continue

            response.raise_for_status()
            payload = response.json()
            submission = Submission.from_dict(payload["submission"])
            items_payload = payload.get("items")
            if items_payload is None:
                items = list(DEFAULT_ITEM_BANK)
            else:
                items = [deserialize_item(item) for item in items_payload]
            job_workspace = workspace / f"submission-{submission.id}-{int(time.time())}"
            job_workspace.mkdir(parents=True, exist_ok=True)

            typer.echo(f"[{worker_id}] claimed submission {submission.id}")

            try:
                _prepare_environment(submission, job_workspace)
                runner = EvaluationRunner(items=items)

                def _progress(_: str) -> None:
                    try:
                        client.post(
                            f"/api/worker/{submission.id}/heartbeat",
                            json={"worker_id": worker_id},
                            timeout=10.0,
                        )
                    except httpx.HTTPError:
                        pass

                metrics, metadata = runner.run(submission, progress=_progress)
                _post_json(
                    client,
                    f"/api/worker/{submission.id}/complete",
                    {"worker_id": worker_id, "metrics": metrics, "metadata": metadata},
                )
                typer.echo(f"[{worker_id}] completed submission {submission.id}")
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"[{worker_id}] failed submission {submission.id}: {exc}", err=True)
                try:
                    _post_json(
                        client,
                        f"/api/worker/{submission.id}/fail",
                        {"worker_id": worker_id, "error": str(exc)},
                    )
                except httpx.HTTPError as report_exc:  # noqa: BLE001
                    typer.echo(f"Failed to report failure: {report_exc}", err=True)
            finally:
                shutil.rmtree(job_workspace, ignore_errors=True)

            if once:
                break

            time.sleep(1)


if __name__ == "__main__":  # pragma: no cover
    app()
