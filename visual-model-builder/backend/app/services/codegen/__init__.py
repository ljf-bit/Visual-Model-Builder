"""Code generation entrypoints for model and training scripts."""

from app.services.graph_ir import IRGraph, IRNode, topological_sort
from app.services.training import has_training_nodes

from .training import generate_training_code


_codegen_rules: dict[str, object] = {}


def register_codegen_rule(op: str):
    """Decorator to register a code generation rule for a node op."""

    def wrapper(fn):
        _codegen_rules[op] = fn
        return fn

    return wrapper


def get_codegen_rule(op: str):
    return _codegen_rules.get(op)


@register_codegen_rule("Conv2d")
def codegen_conv2d(node: IRNode, var_name: str) -> tuple[str, str]:
    p = node.params
    init = (
        f"self.{var_name} = nn.Conv2d("
        f"{p.get('in_channels', 1)}, {p.get('out_channels', 16)}, "
        f"kernel_size={p.get('kernel_size', 3)}, stride={p.get('stride', 1)}, "
        f"padding={p.get('padding', 0)})"
    )
    return init, f"x = self.{var_name}(x)"


@register_codegen_rule("ReLU")
def codegen_relu(node: IRNode, var_name: str) -> tuple[str, str]:
    inplace = bool(node.params.get("inplace", False))
    init = f"self.{var_name} = nn.ReLU(inplace={str(inplace)})"
    return init, f"x = self.{var_name}(x)"


@register_codegen_rule("MaxPool2d")
def codegen_maxpool2d(node: IRNode, var_name: str) -> tuple[str, str]:
    p = node.params
    init = (
        f"self.{var_name} = nn.MaxPool2d("
        f"kernel_size={p.get('kernel_size', 2)}, stride={p.get('stride', 2)}, "
        f"padding={p.get('padding', 0)})"
    )
    return init, f"x = self.{var_name}(x)"


@register_codegen_rule("Flatten")
def codegen_flatten(node: IRNode, var_name: str) -> tuple[str, str]:
    p = node.params
    start_dim = int(p.get("start_dim", 0))
    if start_dim >= 0:
        # Graph params are sample-relative; generated runtime code receives batched tensors.
        start_dim += 1
    init = f"self.{var_name} = nn.Flatten(start_dim={start_dim}, end_dim={p.get('end_dim', -1)})"
    return init, f"x = self.{var_name}(x)"


@register_codegen_rule("Linear")
def codegen_linear(node: IRNode, var_name: str) -> tuple[str, str]:
    p = node.params
    bias = bool(p.get("bias", True))
    init = f"self.{var_name} = nn.Linear({p.get('in_features', 128)}, {p.get('out_features', 10)}, bias={str(bias)})"
    return init, f"x = self.{var_name}(x)"


def generate_model_code(ir_graph: IRGraph) -> str:
    """Generate PyTorch model code from an IR graph."""
    sorted_ids = topological_sort(ir_graph)
    if sorted_ids is None:
        return "# Error: Graph contains a cycle, cannot generate code.\n"

    node_map = {node.node_id: node for node in ir_graph.nodes}
    init_lines: list[str] = []
    forward_lines: list[str] = []
    counter: dict[str, int] = {}

    for node_id in sorted_ids:
        node = node_map[node_id]
        if node.op in ("Input", "Output"):
            continue

        op_lower = node.op.lower()
        counter[op_lower] = counter.get(op_lower, 0) + 1
        var_name = f"{op_lower}_{counter[op_lower]}"

        rule = get_codegen_rule(node.op)
        if rule:
            init_line, forward_line = rule(node, var_name)
            init_lines.append(init_line)
            forward_lines.append(forward_line)

    code = "import torch\nimport torch.nn as nn\n\n\n"
    code += "class Model(nn.Module):\n"
    code += "    def __init__(self):\n"
    code += "        super().__init__()\n"
    for line in init_lines:
        code += f"        {line}\n"
    code += "\n"
    code += "    def forward(self, x):\n"
    if forward_lines:
        for line in forward_lines:
            code += f"        {line}\n"
    else:
        code += "        pass\n"
    code += "        return x\n"
    return code


def generate_code(ir_graph: IRGraph, force_training: bool = False) -> str:
    """Generate model code or training code depending on the graph."""

    if force_training or has_training_nodes(ir_graph):
        return generate_training_code(ir_graph)
    return generate_model_code(ir_graph)
