"""API route definitions."""

from fastapi import APIRouter, HTTPException

from app.schemas.requests import InspectDatasetRequest, ProjectRequest
from app.schemas.responses import (
    GenerateCodeResponse,
    InferShapesResponse,
    InspectDatasetResponse,
    RunTrainingResponse,
    TrainingDiagnosticsResponse,
    TrainingJobResponse,
    TrainingRunDetailResponse,
    TrainingRunListResponse,
    ValidateGraphResponse,
)
from app.services.codegen import generate_code as do_generate_code, generate_model_code as do_generate_model_code
from app.services.dataset_inspection import inspect_dataset_config
from app.services.diagnostics import (
    build_training_insights,
    diagnose_training_graph as do_diagnose_training_graph,
)
from app.services.graph_ir import project_to_ir
from app.services.shape_infer import infer_graph_shapes
from app.services.training import run_training as do_run_training
from app.services.training_jobs import training_job_store
from app.services.training_runs import get_training_run_summary, list_training_run_records
from app.services.validator import validate_graph as do_validate_graph

router = APIRouter()


def _collect_errors(global_errors, node_errors, node_results) -> list[str]:
    errors = list(global_errors)
    for errs in node_errors.values():
        errors.extend(errs)
    for result in node_results.values():
        errors.extend(result.errors)
    return errors


@router.post("/validate-graph", response_model=ValidateGraphResponse)
async def validate_graph(request: ProjectRequest):
    """Validate the project graph structure."""

    ir_graph = project_to_ir(request.project)
    global_errors, node_errors, warnings = do_validate_graph(ir_graph)
    return ValidateGraphResponse(
        ok=len(global_errors) == 0 and sum(len(errs) for errs in node_errors.values()) == 0,
        global_errors=global_errors,
        node_errors=node_errors,
        warnings=warnings,
    )


@router.post("/infer-shapes", response_model=InferShapesResponse)
async def infer_shapes(request: ProjectRequest):
    """Infer input/output shapes for all nodes in the graph."""

    ir_graph = project_to_ir(request.project)
    node_results = infer_graph_shapes(ir_graph)
    ok = all(len(result.errors) == 0 for result in node_results.values()) if node_results else False
    return InferShapesResponse(ok=ok, nodes=node_results)


@router.post("/generate-code", response_model=GenerateCodeResponse)
async def generate_code(request: ProjectRequest):
    """Generate PyTorch model code from the project graph."""

    ir_graph = project_to_ir(request.project)
    global_errors, node_errors, _warnings = do_validate_graph(ir_graph)
    node_results = infer_graph_shapes(ir_graph)
    all_errors = _collect_errors(global_errors, node_errors, node_results)

    if all_errors:
        return GenerateCodeResponse(ok=False, code="", errors=all_errors)

    return GenerateCodeResponse(ok=True, code=do_generate_model_code(ir_graph), errors=[])


@router.post("/validate-training-graph", response_model=ValidateGraphResponse)
async def validate_training_graph(request: ProjectRequest):
    """Validate the Phase 2 training graph structure."""

    ir_graph = project_to_ir(request.project)
    global_errors, node_errors, warnings = do_validate_graph(ir_graph, require_training=True)
    return ValidateGraphResponse(
        ok=len(global_errors) == 0 and sum(len(errs) for errs in node_errors.values()) == 0,
        global_errors=global_errors,
        node_errors=node_errors,
        warnings=warnings,
    )


@router.post("/generate-training-code", response_model=GenerateCodeResponse)
async def generate_training_code(request: ProjectRequest):
    """Generate a full training script from the project graph."""

    ir_graph = project_to_ir(request.project)
    global_errors, node_errors, _warnings = do_validate_graph(ir_graph, require_training=True)
    node_results = infer_graph_shapes(ir_graph)
    all_errors = _collect_errors(global_errors, node_errors, node_results)

    if all_errors:
        return GenerateCodeResponse(ok=False, code="", errors=all_errors)

    return GenerateCodeResponse(ok=True, code=do_generate_code(ir_graph, force_training=True), errors=[])


@router.post("/diagnose-training-graph", response_model=TrainingDiagnosticsResponse)
async def diagnose_training_graph(request: ProjectRequest):
    """Return teaching-oriented diagnostics for the training graph."""

    ir_graph = project_to_ir(request.project)
    return do_diagnose_training_graph(ir_graph)


@router.post("/inspect-dataset", response_model=InspectDatasetResponse)
async def inspect_dataset(request: InspectDatasetRequest):
    """Inspect a Dataset node config and return resolved metadata."""

    inspection = inspect_dataset_config(request.config)
    return InspectDatasetResponse(**inspection.to_dict())


@router.post("/run-training", response_model=RunTrainingResponse)
async def run_training(request: ProjectRequest):
    """Execute the synchronous training loop for compatibility."""

    ir_graph = project_to_ir(request.project)
    diagnostics = do_diagnose_training_graph(ir_graph)

    if not diagnostics.ok:
        return RunTrainingResponse(
            ok=False,
            status="diagnostics_failed",
            logs=[],
            errors=diagnostics.errors,
            diagnostics=diagnostics,
            insights=build_training_insights(
                diagnostics=diagnostics,
                logs=[],
                status="diagnostics_failed",
                runtime_messages=diagnostics.errors,
                training_metadata=None,
            ),
            training_metadata=None,
        )

    try:
        status, logs, warnings, training_metadata, insights, evaluation = do_run_training(
            ir_graph,
            project_name=request.project.metadata.name,
            diagnostics_payload=diagnostics,
            project_snapshot=request.project.model_dump(by_alias=True),
        )
    except RuntimeError as exc:
        return RunTrainingResponse(
            ok=False,
            status="runtime_unavailable",
            logs=[],
            errors=[str(exc)],
            diagnostics=diagnostics,
            insights=build_training_insights(
                diagnostics=diagnostics,
                logs=[],
                status="runtime_unavailable",
                runtime_messages=[str(exc)],
                training_metadata=None,
            ),
            training_metadata=None,
        )
    except Exception as exc:
        return RunTrainingResponse(
            ok=False,
            status="runtime_failed",
            logs=[],
            errors=[str(exc)],
            diagnostics=diagnostics,
            insights=build_training_insights(
                diagnostics=diagnostics,
                logs=[],
                status="runtime_failed",
                runtime_messages=[str(exc)],
                training_metadata=None,
            ),
            training_metadata=None,
        )

    return RunTrainingResponse(
        ok=True,
        status=status,
        logs=logs,
        errors=warnings,
        diagnostics=diagnostics,
        insights=insights,
        evaluation=evaluation,
        training_metadata=training_metadata,
    )


@router.post("/training-jobs", response_model=TrainingJobResponse)
async def create_training_job(request: ProjectRequest):
    """Create an asynchronous training job and return its initial state."""

    return training_job_store.create(request.project)


@router.get("/training-jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """Return the latest state for a training job."""

    job = training_job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


@router.post("/training-jobs/{job_id}/cancel", response_model=TrainingJobResponse)
async def cancel_training_job(job_id: str):
    """Request best-effort cancellation for a queued or running training job."""

    job = training_job_store.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return job


@router.get("/training-runs", response_model=TrainingRunListResponse)
async def list_training_runs(limit: int = 50):
    """List persisted training runs from the artifact directory."""

    return TrainingRunListResponse(runs=list_training_run_records(limit=limit))


@router.get("/training-runs/{run_id}", response_model=TrainingRunDetailResponse)
async def get_training_run(run_id: str):
    """Return a persisted training run summary."""

    summary = get_training_run_summary(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return TrainingRunDetailResponse(runId=run_id, summary=summary)
