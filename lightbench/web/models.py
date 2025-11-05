"""Data models used by the LightBench web service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Mapping, MutableMapping, Optional


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.rstrip("Z")
    return datetime.fromisoformat(value)


class SubmissionStatus(str, Enum):
    """Lifecycle states for a submission."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class SubmissionCreate:
    """Payload required to register a new optimizer submission."""

    optimizer_name: str
    contact_email: str
    contact_name: str
    repository_url: str
    entrypoint: str
    description: str = ""
    resource_limits: Mapping[str, Any] | None = None
    extra_metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class Submission:
    """Persisted submission record."""

    id: int
    optimizer_name: str
    contact_email: str
    contact_name: str
    repository_url: str
    entrypoint: str
    description: str
    resource_limits: MutableMapping[str, Any]
    extra_metadata: MutableMapping[str, Any]
    status: SubmissionStatus
    created_at: datetime
    updated_at: datetime
    modal_task_id: Optional[str] = None
    last_error: Optional[str] = None
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def short_identifier(self) -> str:
        return f"{self.optimizer_name}-{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "optimizer_name": self.optimizer_name,
            "contact_email": self.contact_email,
            "contact_name": self.contact_name,
            "repository_url": self.repository_url,
            "entrypoint": self.entrypoint,
            "description": self.description,
            "resource_limits": dict(self.resource_limits),
            "extra_metadata": dict(self.extra_metadata),
            "status": self.status.value,
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
            "job_reference": self.modal_task_id,
            "last_error": self.last_error,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() + "Z" if self.claimed_at else None,
            "started_at": self.started_at.isoformat() + "Z" if self.started_at else None,
            "completed_at": self.completed_at.isoformat() + "Z" if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Submission":
        return cls(
            id=int(payload["id"]),
            optimizer_name=str(payload["optimizer_name"]),
            contact_email=str(payload["contact_email"]),
            contact_name=str(payload["contact_name"]),
            repository_url=str(payload["repository_url"]),
            entrypoint=str(payload["entrypoint"]),
            description=str(payload.get("description", "")),
            resource_limits=dict(payload.get("resource_limits", {})),
            extra_metadata=dict(payload.get("extra_metadata", {})),
            status=SubmissionStatus(payload["status"]),
            created_at=_parse_dt(payload.get("created_at")) or datetime.utcnow(),
            updated_at=_parse_dt(payload.get("updated_at")) or datetime.utcnow(),
            modal_task_id=payload.get("job_reference") or payload.get("modal_task_id"),
            last_error=payload.get("last_error"),
            claimed_by=payload.get("claimed_by"),
            claimed_at=_parse_dt(payload.get("claimed_at")),
            started_at=_parse_dt(payload.get("started_at")),
            completed_at=_parse_dt(payload.get("completed_at")),
        )


@dataclass(slots=True)
class EvaluationSummary:
    """Aggregate view of an evaluation outcome for display."""

    submission_id: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    leaderboard_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[datetime] = None
