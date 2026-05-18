"""In-memory lifecycle management for asynchronous training jobs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from app.schemas.graph import ProjectGraph
from app.schemas.responses import TrainingDiagnosticsResponse, TrainingJobResponse
from app.services.diagnostics import (
    build_training_insights,
    diagnose_training_graph,
)
from app.services.graph_ir import project_to_ir
from app.services.training import run_training

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingJobRecord:
    """Mutable state for one training job."""

    job_id: str
    project: ProjectGraph
    status: str
    progress: float
    created_at: str
    updated_at: str
    diagnostics: TrainingDiagnosticsResponse | None = None
    logs: list[dict[str, float | int | None]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    insights: dict[str, object] | None = None
    training_metadata: dict[str, object] | None = None
    cancel_requested: bool = False
    future: Future | None = field(default=None, repr=False)

    def to_response(self) -> TrainingJobResponse:
        ok = self.status == "completed"
        return TrainingJobResponse(
            jobId=self.job_id,
            ok=ok,
            status=self.status,
            progress=self.progress,
            cancelRequested=self.cancel_requested,
            logs=self.logs,
            errors=self.errors,
            diagnostics=self.diagnostics,
            insights=self.insights,
            trainingMetadata=self.training_metadata,
            createdAt=self.created_at,
            updatedAt=self.updated_at,
        )


class TrainingJobStore:
    """Small single-process job store for portfolio/demo deployments."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vmb-training")
        self._jobs: dict[str, TrainingJobRecord] = {}
        self._lock = RLock()

    def create(self, project: ProjectGraph) -> TrainingJobResponse:
        job_id = uuid4().hex
        now = _utc_now()
        ir_graph = project_to_ir(project)
        diagnostics = diagnose_training_graph(ir_graph)
        record = TrainingJobRecord(
            job_id=job_id,
            project=project,
            status="queued",
            progress=0.0,
            created_at=now,
            updated_at=now,
            diagnostics=diagnostics,
        )

        if not diagnostics.ok:
            record.status = "failed"
            record.errors = list(diagnostics.errors)
            record.insights = build_training_insights(
                diagnostics=diagnostics,
                logs=[],
                status="diagnostics_failed",
                runtime_messages=diagnostics.errors,
                training_metadata=None,
            ).model_dump(by_alias=True)
            record.updated_at = _utc_now()
            with self._lock:
                self._jobs[job_id] = record
            return record.to_response()

        with self._lock:
            self._jobs[job_id] = record
            record.future = self._executor.submit(self._run_job, job_id)
        return record.to_response()

    def get(self, job_id: str) -> TrainingJobResponse | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.to_response() if record else None

    def cancel(self, job_id: str) -> TrainingJobResponse | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None

            if record.status not in TERMINAL_STATUSES:
                record.cancel_requested = True
                if record.status == "queued" and record.future and record.future.cancel():
                    record.status = "cancelled"
                    record.progress = 0.0
                    record.errors = ["Training was cancelled before it started."]
                record.updated_at = _utc_now()
            return record.to_response()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if record.cancel_requested:
                record.status = "cancelled"
                record.errors = ["Training was cancelled before it started."]
                record.updated_at = _utc_now()
                return
            record.status = "running"
            record.updated_at = _utc_now()

        ir_graph = project_to_ir(record.project)

        def should_cancel() -> bool:
            with self._lock:
                current = self._jobs[job_id]
                return current.cancel_requested

        def on_progress(payload: dict[str, Any]) -> None:
            with self._lock:
                current = self._jobs[job_id]
                current.status = str(payload.get("status", "running"))
                current.progress = float(payload.get("progress", current.progress))
                current.logs = [*current.logs, payload["log"]] if "log" in payload else current.logs
                current.updated_at = _utc_now()

        try:
            status, logs, warnings, training_metadata, insights = run_training(
                ir_graph,
                project_name=record.project.metadata.name,
                diagnostics_payload=record.diagnostics,
                progress_callback=on_progress,
                should_cancel=should_cancel,
            )
        except RuntimeError as exc:
            self._fail_job(job_id, "failed", [str(exc)], "runtime_unavailable")
            return
        except Exception as exc:
            self._fail_job(job_id, "failed", [str(exc)], "runtime_failed")
            return

        with self._lock:
            current = self._jobs[job_id]
            current.status = "cancelled" if status == "cancelled" else "completed"
            current.progress = 1.0 if status == "completed" else current.progress
            current.logs = logs
            current.errors = warnings
            current.training_metadata = training_metadata
            current.insights = insights
            current.updated_at = _utc_now()

    def _fail_job(self, job_id: str, status: str, errors: list[str], insight_status: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = status
            record.errors = errors
            record.insights = build_training_insights(
                diagnostics=record.diagnostics,
                logs=record.logs,
                status=insight_status,
                runtime_messages=errors,
                training_metadata=None,
            ).model_dump(by_alias=True)
            record.updated_at = _utc_now()


training_job_store = TrainingJobStore()
