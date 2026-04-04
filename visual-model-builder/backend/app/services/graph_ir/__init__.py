"""Intermediate representation helpers for project graphs."""

from typing import Any

from pydantic import BaseModel, Field

MODEL_NODE_TYPES = {
    "Input",
    "Conv2d",
    "ReLU",
    "MaxPool2d",
    "Flatten",
    "Linear",
    "Output",
}
DATA_NODE_TYPES = {"Dataset", "DataLoader"}
TRAINING_NODE_TYPES = {"Loss", "Optimizer", "Trainer", "Metric"}


class IRNode(BaseModel):
    """A single node in the intermediate representation."""

    node_id: str
    op: str
    category: str | None = None
    training_role: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    inferred_input_shape: list[int] | None = None
    inferred_output_shape: list[int] | None = None
    errors: list[str] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        if self.category is None:
            self.category = get_node_category(self.op)
        if self.training_role is None:
            self.training_role = get_training_role(self.op)
        if not self.runtime_config and self.category != "model":
            self.runtime_config = dict(self.params)


class IRGraph(BaseModel):
    """Full intermediate representation of the graph."""

    version: str
    nodes: list[IRNode]
    edges: list[dict[str, str]]


def get_node_category(op: str) -> str:
    """Map a node op to a high-level category."""

    if op in MODEL_NODE_TYPES:
        return "model"
    if op in DATA_NODE_TYPES:
        return "data"
    if op in TRAINING_NODE_TYPES:
        return "training"
    return "unknown"


def get_training_role(op: str) -> str | None:
    """Map a node op to its training role."""

    role_map = {
        "Dataset": "dataset",
        "DataLoader": "dataloader",
        "Loss": "loss",
        "Optimizer": "optimizer",
        "Trainer": "trainer",
        "Metric": "metric",
    }
    return role_map.get(op)


def project_to_ir(project) -> IRGraph:
    """Convert a ProjectGraph to an IRGraph."""

    from app.schemas.graph import ProjectGraph

    if not isinstance(project, ProjectGraph):
        raise TypeError("Expected ProjectGraph instance")

    forward_adj: dict[str, list[str]] = {node.id: [] for node in project.nodes}
    reverse_adj: dict[str, list[str]] = {node.id: [] for node in project.nodes}

    for edge in project.edges:
        forward_adj[edge.source].append(edge.target)
        reverse_adj[edge.target].append(edge.source)

    ir_nodes = []
    for node in project.nodes:
        category = get_node_category(node.type)
        ir_nodes.append(
            IRNode(
                node_id=node.id,
                op=node.type,
                category=category,
                training_role=get_training_role(node.type),
                params=node.data.params,
                runtime_config=node.data.params if category != "model" else {},
                input_refs=reverse_adj.get(node.id, []),
                output_refs=forward_adj.get(node.id, []),
            )
        )

    ir_edges = [{"source": edge.source, "target": edge.target} for edge in project.edges]

    return IRGraph(version=project.version, nodes=ir_nodes, edges=ir_edges)


def topological_sort(ir_graph: IRGraph) -> list[str] | None:
    """
    Topological sort using Kahn's algorithm.
    Returns ordered node IDs, or None if cycle detected.
    """
    in_degree: dict[str, int] = {n.node_id: 0 for n in ir_graph.nodes}
    adjacency: dict[str, list[str]] = {n.node_id: [] for n in ir_graph.nodes}

    for edge in ir_graph.edges:
        src, tgt = edge["source"], edge["target"]
        adjacency[src].append(tgt)
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_ids: list[str] = []

    while queue:
        current = queue.pop(0)
        sorted_ids.append(current)
        for neighbor in adjacency.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_ids if len(sorted_ids) == len(ir_graph.nodes) else None
