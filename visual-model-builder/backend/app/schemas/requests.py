"""Request schemas for API endpoints."""

from typing import Any

from pydantic import BaseModel, Field
from app.schemas.graph import ProjectGraph


class ProjectRequest(BaseModel):
    """Wrapper for project graph in API requests."""
    project: ProjectGraph


class InspectDatasetRequest(BaseModel):
    """Wrapper for Dataset node config in dataset inspection requests."""

    config: dict[str, Any] = Field(default_factory=dict)
