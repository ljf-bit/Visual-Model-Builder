'''
"""
Shape Inference service.

Propagates tensor shapes through the graph following topological order.
Each node type has an independent inference rule.
"""

from typing import Any

from pydantic import BaseModel


class ShapeResult(BaseModel):
    """Result of shape inference for a single node."""
    ok: bool
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    errors: list[str] = []


# ============================================================
# Shape inference rule registry
# ============================================================

ShapeInferFn = type(lambda input_shape, params: ShapeResult(ok=True))

_shape_rules: dict[str, Any] = {}


def register_shape_rule(op: str):
    """Decorator to register a shape inference rule for a node op."""
    def wrapper(fn):
        _shape_rules[op] = fn
        return fn
    return wrapper


def get_shape_rule(op: str):
    """Get the shape inference function for a node op."""
    return _shape_rules.get(op)


# ============================================================
# Phase 1 shape rules (placeholder implementations)
# ============================================================

@register_shape_rule("Input")
def infer_input(input_shape: list[int] | None, params: dict) -> ShapeResult:
    """Input node: output shape is the user-specified inputShape."""
    shape = params.get("inputShape")
    if not shape or not isinstance(shape, list):
        return ShapeResult(ok=False, errors=["输入 shape 格式非法"])
    return ShapeResult(ok=True, output_shape=shape)


@register_shape_rule("Conv2d")
def infer_conv2d(input_shape: list[int] | None, params: dict) -> ShapeResult:
    """Conv2d: validate channels and compute spatial dims."""
    if not input_shape or len(input_shape) != 3:
        return ShapeResult(ok=False, input_shape=input_shape,
                           errors=["Conv2d 需要 [C, H, W] 的 3 维输入"])
                           
    in_channels = params.get("in_channels", 1)
    if input_shape[0] != in_channels:
        return ShapeResult(ok=False, input_shape=input_shape,
                           errors=[f"输入通道数 ({input_shape[0]}) 与参数 in_channels ({in_channels}) 不匹配"])
    
    out_channels = params.get("out_channels", 16)
    kernel_size = params.get("kernel_size", 3)
    stride = params.get("stride", 1)
    padding = params.get("padding", 0)

    h = (input_shape[1] + 2 * padding - kernel_size) // stride + 1
    w = (input_shape[2] + 2 * padding - kernel_size) // stride + 1

    if h <= 0 or w <= 0:
        return ShapeResult(ok=False, input_shape=input_shape,
                           errors=["卷积后的空间尺寸小于等于 0，请检查参数"])

    return ShapeResult(ok=True, input_shape=input_shape, output_shape=[out_channels, h, w])


@register_shape_rule("ReLU")
def infer_relu(input_shape: list[int] | None, params: dict) -> ShapeResult:
    """ReLU: pass-through shape."""
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=input_shape)


@register_shape_rule("MaxPool2d")
def infer_maxpool2d(input_shape: list[int] | None, params: dict) -> ShapeResult:
    """MaxPool2d: compute pooled spatial dims."""
    if not input_shape or len(input_shape) != 3:
        return ShapeResult(ok=False, input_shape=input_shape,
                           errors=["MaxPool2d 需要 [C, H, W] 的 3 维输入"])
                           
    kernel_size = params.get("kernel_size", 2)
    stride = params.get("stride", 2)
    padding = params.get("padding", 0)

    h = (input_shape[1] + 2 * padding - kernel_size) // stride + 1
    w = (input_shape[2] + 2 * padding - kernel_size) // stride + 1

    if h <= 0 or w <= 0:
        return ShapeResult(ok=False, input_shape=input_shape,
                           errors=["池化后的空间尺寸小于等于 0，请检查参数"])
                           
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=[input_shape[0], h, w])


@register_shape_rule("Flatten")
def infer_flatten(input_shape: list[int] | None, params: dict) -> ShapeResult:
    """Flatten: collapse dimensions."""
    if not input_shape:
        return ShapeResult(ok=False, errors=["缺少输入 shape"])
    
    start_dim = params.get("start_dim", 1)
    end_dim = params.get("end_dim", -1)
    
    L = len(input_shape)
    s = start_dim if start_dim >= 0 else L + start_dim
    e = end_dim if end_dim >= 0 else L + end_dim
    
    if s < 0: s = 0
    if e >= L: e = L - 1
    
    if s > e or s >= L:
        return ShapeResult(ok=False, input_shape=input_shape, 
                           errors=[f"Flatten 维度边界无效: start_dim={start_dim}, end_dim={end_dim}, 输入维度={L}"])

    flat_size = 1
    for d in input_shape[s:e+1]:
        flat_size *= d
        
    out_shape = input_shape[:s] + [flat_size] + input_shape[e+1:]
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=out_shape)


@register_shape_rule("Linear")
def infer_linear(input_shape: list[int] | None, params: dict) -> ShapeResult:
    """Linear: validate in_features against the last dimension, output [..., out_features]."""
    if not input_shape:
        return ShapeResult(ok=False, errors=["缺少输入 shape"])
                           
    in_features = params.get("in_features", 128)
    if input_shape[-1] != in_features:
        return ShapeResult(ok=False, input_shape=input_shape,
                           errors=[f"输入特征数 (最后侧维度 {input_shape[-1]}) 与参数 in_features ({in_features}) 不匹配"])
                           
    out_features = params.get("out_features", 10)
    out_shape = list(input_shape)
    out_shape[-1] = out_features
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=out_shape)


@register_shape_rule("Output")
def infer_output(input_shape: list[int] | None, params: dict) -> ShapeResult:
    """Output: pass-through shape."""
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=input_shape)


# ============================================================
# Main inference loop
# ============================================================

from app.services.graph_ir import IRGraph, topological_sort
from app.schemas.responses import NodeShapeResult

def infer_graph_shapes(ir_graph: IRGraph) -> dict[str, NodeShapeResult]:
    """
    Run shape inference over the entire graph following topological order.
    Returns a dict mapping node_id to its NodeShapeResult.
    """
    sorted_ids = topological_sort(ir_graph)
    if sorted_ids is None:
        # Cannot infer shapes if there are cycles
        return {}

    node_map = {n.node_id: n for n in ir_graph.nodes}
    results: dict[str, NodeShapeResult] = {}
    
    # Environment to keep track of output shapes propagated
    shapes_env: dict[str, list[int] | None] = {}

    for node_id in sorted_ids:
        node = node_map[node_id]
        
        # Determine input shape
        input_shape = None
        if node.input_refs:
            # Phase 1: assume single input per node for layers
            upstream_id = node.input_refs[0]
            input_shape = shapes_env.get(upstream_id)
            
        rule = get_shape_rule(node.op)
        if rule:
            res = rule(input_shape, node.params)
            results[node_id] = NodeShapeResult(
                input_shape=res.input_shape,
                output_shape=res.output_shape,
                errors=res.errors
            )
            node.inferred_input_shape = res.input_shape
            node.inferred_output_shape = res.output_shape
            node.errors = res.errors
            
            if res.ok and res.output_shape:
                shapes_env[node_id] = res.output_shape
            else:
                shapes_env[node_id] = None
        else:
            results[node_id] = NodeShapeResult(
                input_shape=input_shape,
                output_shape=None,
                errors=[f"未知的节点类型: {node.op}"]
            )
            shapes_env[node_id] = None
            
    return results
'''

"""Shape inference for model nodes."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.responses import NodeShapeResult
from app.services.graph_ir import IRGraph, topological_sort
from app.services.training import get_node_map


class ShapeResult(BaseModel):
    """Result of shape inference for a single node."""

    ok: bool
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    errors: list[str] = Field(default_factory=list)


ShapeInferFn = type(lambda input_shape, params: ShapeResult(ok=True))

_shape_rules: dict[str, Any] = {}


def register_shape_rule(op: str):
    """Decorator to register a shape inference rule for a node op."""

    def wrapper(fn):
        _shape_rules[op] = fn
        return fn

    return wrapper


def get_shape_rule(op: str):
    """Get the shape inference function for a node op."""

    return _shape_rules.get(op)


@register_shape_rule("Input")
def infer_input(input_shape: list[int] | None, params: dict) -> ShapeResult:
    shape = params.get("inputShape")
    if not shape or not isinstance(shape, list) or len(shape) != 3:
        return ShapeResult(ok=False, errors=["Input requires inputShape as [C, H, W]."])
    if not all(isinstance(value, int) and value > 0 for value in shape):
        return ShapeResult(ok=False, errors=["Input shape values must be positive integers."])
    return ShapeResult(ok=True, output_shape=shape)


@register_shape_rule("Conv2d")
def infer_conv2d(input_shape: list[int] | None, params: dict) -> ShapeResult:
    if not input_shape or len(input_shape) != 3:
        return ShapeResult(ok=False, input_shape=input_shape, errors=["Conv2d expects a [C, H, W] input tensor."])

    in_channels = params.get("in_channels", 1)
    if input_shape[0] != in_channels:
        return ShapeResult(
            ok=False,
            input_shape=input_shape,
            errors=[f"Conv2d expected {in_channels} input channels but received {input_shape[0]}."],
        )

    out_channels = params.get("out_channels", 16)
    kernel_size = params.get("kernel_size", 3)
    stride = params.get("stride", 1)
    padding = params.get("padding", 0)

    h = (input_shape[1] + 2 * padding - kernel_size) // stride + 1
    w = (input_shape[2] + 2 * padding - kernel_size) // stride + 1

    if h <= 0 or w <= 0:
        return ShapeResult(
            ok=False,
            input_shape=input_shape,
            errors=["Conv2d produced a non-positive spatial size. Check kernel, stride, and padding."],
        )

    return ShapeResult(ok=True, input_shape=input_shape, output_shape=[out_channels, h, w])


@register_shape_rule("ReLU")
def infer_relu(input_shape: list[int] | None, params: dict) -> ShapeResult:
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=input_shape)


@register_shape_rule("MaxPool2d")
def infer_maxpool2d(input_shape: list[int] | None, params: dict) -> ShapeResult:
    if not input_shape or len(input_shape) != 3:
        return ShapeResult(ok=False, input_shape=input_shape, errors=["MaxPool2d expects a [C, H, W] input tensor."])

    kernel_size = params.get("kernel_size", 2)
    stride = params.get("stride", 2)
    padding = params.get("padding", 0)

    h = (input_shape[1] + 2 * padding - kernel_size) // stride + 1
    w = (input_shape[2] + 2 * padding - kernel_size) // stride + 1

    if h <= 0 or w <= 0:
        return ShapeResult(
            ok=False,
            input_shape=input_shape,
            errors=["MaxPool2d produced a non-positive spatial size. Check kernel, stride, and padding."],
        )

    return ShapeResult(ok=True, input_shape=input_shape, output_shape=[input_shape[0], h, w])


@register_shape_rule("Flatten")
def infer_flatten(input_shape: list[int] | None, params: dict) -> ShapeResult:
    if not input_shape:
        return ShapeResult(ok=False, errors=["Flatten is missing an input shape."])

    start_dim = params.get("start_dim", 1)
    end_dim = params.get("end_dim", -1)
    rank = len(input_shape)
    start = start_dim if start_dim >= 0 else rank + start_dim
    end = end_dim if end_dim >= 0 else rank + end_dim

    if start < 0:
        start = 0
    if end >= rank:
        end = rank - 1

    if start > end or start >= rank:
        return ShapeResult(
            ok=False,
            input_shape=input_shape,
            errors=[f"Flatten dimensions are invalid for input rank {rank}: start_dim={start_dim}, end_dim={end_dim}."],
        )

    flat_size = 1
    for dim in input_shape[start : end + 1]:
        flat_size *= dim

    out_shape = input_shape[:start] + [flat_size] + input_shape[end + 1 :]
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=out_shape)


@register_shape_rule("Linear")
def infer_linear(input_shape: list[int] | None, params: dict) -> ShapeResult:
    if not input_shape:
        return ShapeResult(ok=False, errors=["Linear is missing an input shape."])

    in_features = params.get("in_features", 128)
    if input_shape[-1] != in_features:
        return ShapeResult(
            ok=False,
            input_shape=input_shape,
            errors=[f"Linear expected in_features={in_features} but received {input_shape[-1]} features."],
        )

    out_features = params.get("out_features", 10)
    out_shape = list(input_shape)
    out_shape[-1] = out_features
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=out_shape)


@register_shape_rule("Output")
def infer_output(input_shape: list[int] | None, params: dict) -> ShapeResult:
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=input_shape)


@register_shape_rule("Dataset")
@register_shape_rule("DataLoader")
@register_shape_rule("Optimizer")
@register_shape_rule("Trainer")
@register_shape_rule("Metric")
def infer_semantic_node(input_shape: list[int] | None, params: dict) -> ShapeResult:
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=None)


@register_shape_rule("Loss")
def infer_loss(input_shape: list[int] | None, params: dict) -> ShapeResult:
    return ShapeResult(ok=True, input_shape=input_shape, output_shape=None)


def infer_graph_shapes(ir_graph: IRGraph) -> dict[str, NodeShapeResult]:
    """Run shape inference over the entire graph following topological order."""

    sorted_ids = topological_sort(ir_graph)
    if sorted_ids is None:
        return {}

    node_map = get_node_map(ir_graph)
    results: dict[str, NodeShapeResult] = {}
    shapes_env: dict[str, list[int] | None] = {}

    for node_id in sorted_ids:
        node = node_map[node_id]
        input_shape = None

        if node.category == "model" and node.op != "Input":
            model_inputs = [ref for ref in node.input_refs if node_map.get(ref) and node_map[ref].category == "model"]
            if model_inputs:
                input_shape = shapes_env.get(model_inputs[0])
        elif node.input_refs:
            input_shape = shapes_env.get(node.input_refs[0])

        rule = get_shape_rule(node.op)
        if rule:
            result = rule(input_shape, node.params)
            results[node_id] = NodeShapeResult(
                input_shape=result.input_shape,
                output_shape=result.output_shape,
                errors=result.errors,
            )
            node.inferred_input_shape = result.input_shape
            node.inferred_output_shape = result.output_shape
            node.errors = result.errors
            shapes_env[node_id] = result.output_shape if result.ok else None
            continue

        results[node_id] = NodeShapeResult(
            input_shape=input_shape,
            output_shape=None,
            errors=[f"Unknown node type `{node.op}`."],
        )
        shapes_env[node_id] = None

    return results
