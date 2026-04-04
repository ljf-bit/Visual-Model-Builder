'''
"""
API route definitions.

All REST endpoints are registered here and mounted in main.py.
"""

from fastapi import APIRouter
from app.schemas.requests import ProjectRequest
from app.schemas.responses import (
    ValidateGraphResponse,
    InferShapesResponse,
    GenerateCodeResponse,
)
from app.services.graph_ir import project_to_ir
from app.services.validator import validate_graph as do_validate_graph
from app.services.shape_infer import infer_graph_shapes
from app.services.codegen import generate_model_code as do_generate_code

router = APIRouter()


@router.post("/validate-graph", response_model=ValidateGraphResponse)
async def validate_graph(request: ProjectRequest):
    """
    Validate the project graph structure.
    """
    ir_graph = project_to_ir(request.project)
    global_errors, node_errors = do_validate_graph(ir_graph)
    
    return ValidateGraphResponse(
        ok=len(global_errors) == 0 and sum(len(errs) for errs in node_errors.values()) == 0,
        global_errors=global_errors,
        node_errors=node_errors,
    )


@router.post("/infer-shapes", response_model=InferShapesResponse)
async def infer_shapes(request: ProjectRequest):
    """
    Infer input/output shapes for all nodes in the graph.
    """
    ir_graph = project_to_ir(request.project)
    node_results = infer_graph_shapes(ir_graph)
    
    # Check if there are any errors in any node's inference
    ok = all(len(res.errors) == 0 for res in node_results.values()) if node_results else False
    
    return InferShapesResponse(
        ok=ok,
        nodes=node_results,
    )


@router.post("/generate-code", response_model=GenerateCodeResponse)
async def generate_code(request: ProjectRequest):
    """
    Generate PyTorch model code from the project graph.
    """
    ir_graph = project_to_ir(request.project)
    global_errors, node_errors = do_validate_graph(ir_graph)
    node_results = infer_graph_shapes(ir_graph)
    
    # Collect all errors
    all_errors = list(global_errors)
    for errs in node_errors.values():
        all_errors.extend(errs)
    for res in node_results.values():
        all_errors.extend(res.errors)
        
    if all_errors:
        return GenerateCodeResponse(
            ok=False,
            code="",
            errors=all_errors
        )
        
    code = do_generate_code(ir_graph)
    if code.startswith("# Error"):
        return GenerateCodeResponse(
            ok=False,
            code="",
            errors=["生成代码失败，图结构不合法"]
        )
        
    return GenerateCodeResponse(
        ok=True,
        code=code,
        errors=[],
    )
'''

"""API route definitions."""

from fastapi import APIRouter

from app.schemas.requests import ProjectRequest
from app.schemas.responses import (
    GenerateCodeResponse,
    InferShapesResponse,
    RunTrainingResponse,
    TrainingDiagnosticsResponse,
    ValidateGraphResponse,
)
from app.services.codegen import generate_code as do_generate_code, generate_model_code as do_generate_model_code
from app.services.diagnostics import (
    build_training_insights,
    diagnose_training_graph as do_diagnose_training_graph,
)
from app.services.graph_ir import project_to_ir
from app.services.shape_infer import infer_graph_shapes
from app.services.training import run_training as do_run_training
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
    global_errors, node_errors = do_validate_graph(ir_graph)
    return ValidateGraphResponse(
        ok=len(global_errors) == 0 and sum(len(errs) for errs in node_errors.values()) == 0,
        global_errors=global_errors,
        node_errors=node_errors,
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
    """Generate PyTorch code from the project graph."""

    ir_graph = project_to_ir(request.project)
    global_errors, node_errors = do_validate_graph(ir_graph)
    node_results = infer_graph_shapes(ir_graph)
    all_errors = _collect_errors(global_errors, node_errors, node_results)

    if all_errors:
        return GenerateCodeResponse(ok=False, code="", errors=all_errors)

    return GenerateCodeResponse(ok=True, code=do_generate_model_code(ir_graph), errors=[])


@router.post("/validate-training-graph", response_model=ValidateGraphResponse)
async def validate_training_graph(request: ProjectRequest):
    """Validate the Phase 2 training graph structure."""

    ir_graph = project_to_ir(request.project)
    global_errors, node_errors = do_validate_graph(ir_graph, require_training=True)
    return ValidateGraphResponse(
        ok=len(global_errors) == 0 and sum(len(errs) for errs in node_errors.values()) == 0,
        global_errors=global_errors,
        node_errors=node_errors,
    )


@router.post("/generate-training-code", response_model=GenerateCodeResponse)
async def generate_training_code(request: ProjectRequest):
    """Generate a full training script from the project graph."""

    ir_graph = project_to_ir(request.project)
    global_errors, node_errors = do_validate_graph(ir_graph, require_training=True)
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


@router.post("/run-training", response_model=RunTrainingResponse)
async def run_training(request: ProjectRequest):
    """Execute the minimal synchronous Phase 2 training loop."""

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
        status, logs, warnings, training_metadata, insights = do_run_training(
            ir_graph,
            project_name=request.project.metadata.name,
            diagnostics_payload=diagnostics,
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

    return RunTrainingResponse(
        ok=True,
        status=status,
        logs=logs,
        errors=warnings,
        diagnostics=diagnostics,
        insights=insights,
        training_metadata=training_metadata,
    )
