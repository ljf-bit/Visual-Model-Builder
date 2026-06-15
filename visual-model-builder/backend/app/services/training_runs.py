"""Read persisted training run summaries from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.training import _get_training_runs_root


def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _metadata_value(metadata: dict[str, Any], camel_key: str, snake_key: str, default: Any = None) -> Any:
    if camel_key in metadata:
        return metadata[camel_key]
    return metadata.get(snake_key, default)


def _metric_value(snapshot: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _record_from_summary(run_id: str, summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("trainingMetadata") if isinstance(summary.get("trainingMetadata"), dict) else {}
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    final_train = evaluation.get("finalTrain") if isinstance(evaluation.get("finalTrain"), dict) else {}
    final_validation = evaluation.get("finalValidation") if isinstance(evaluation.get("finalValidation"), dict) else None
    display_final = final_validation if isinstance(final_validation, dict) else final_train

    return {
        "runId": str(_metadata_value(metadata, "runId", "run_id", run_id) or run_id),
        "ok": bool(summary.get("ok", False)),
        "status": str(summary.get("status", "")),
        "projectName": str(_metadata_value(metadata, "projectName", "project_name", "")),
        "createdAt": str(_metadata_value(metadata, "startedAt", "started_at", "")),
        "completedAt": str(_metadata_value(metadata, "completedAt", "completed_at", "")),
        "datasetUsed": str(_metadata_value(metadata, "datasetUsed", "dataset_used", "")),
        "datasetMode": str(_metadata_value(metadata, "datasetMode", "dataset_mode", "")),
        "durationSeconds": _metadata_value(metadata, "durationSeconds", "duration_seconds", None),
        "finalLoss": _metric_value(display_final, "loss"),
        "finalAccuracy": _metric_value(display_final, "accuracy"),
        "macroF1": _metric_value(evaluation, "macroF1"),
        "weightedF1": _metric_value(evaluation, "weightedF1"),
        "summaryPath": str(summary_path.resolve()),
    }


def list_training_run_records(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent persisted runs sorted newest first."""

    root = _get_training_runs_root()
    if not root.exists():
        return []

    records: list[dict[str, Any]] = []
    summary_paths = sorted(
        root.glob("*/training_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summary_paths[: max(limit, 1)]:
        summary = _read_summary(summary_path)
        if summary is None:
            continue
        records.append(_record_from_summary(summary_path.parent.name, summary_path, summary))
    return records


def get_training_run_summary(run_id: str) -> dict[str, Any] | None:
    """Return the raw persisted summary for one run id."""

    if not run_id or "/" in run_id or "\\" in run_id:
        return None

    summary_path = _get_training_runs_root() / run_id / "training_summary.json"
    if not summary_path.exists():
        return None
    return _read_summary(summary_path)
