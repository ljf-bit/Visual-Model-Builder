"""
Pydantic schemas for graph data structures.

These mirror the frontend TypeScript types and define the
shared contract between frontend and backend.
"""

from pydantic import BaseModel, Field
from typing import Any


class GraphNodeData(BaseModel):
    """Data payload attached to each node instance."""
    label: str
    params: dict[str, Any] = Field(default_factory=dict)
    inferred_input_shape: list[int] | None = Field(None, alias="inferredInputShape")
    inferred_output_shape: list[int] | None = Field(None, alias="inferredOutputShape")
    errors: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class Position(BaseModel):
    """2D position on canvas."""
    x: float
    y: float


class GraphNode(BaseModel):
    """A node instance on the canvas."""
    id: str
    type: str
    position: Position
    data: GraphNodeData


class GraphEdge(BaseModel):
    """An edge connecting two nodes."""
    id: str
    source: str
    target: str
    source_handle: str | None = Field(None, alias="sourceHandle")
    target_handle: str | None = Field(None, alias="targetHandle")

    model_config = {"populate_by_name": True}


class ProjectMetadata(BaseModel):
    """Project metadata."""
    name: str
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")

    model_config = {"populate_by_name": True}


class ProjectGraph(BaseModel):
    """Full project graph (serializable)."""
    version: str
    metadata: ProjectMetadata
    nodes: list[GraphNode]
    edges: list[GraphEdge]
