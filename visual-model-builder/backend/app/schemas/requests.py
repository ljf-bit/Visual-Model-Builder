"""Request schemas for API endpoints."""

from pydantic import BaseModel
from app.schemas.graph import ProjectGraph


class ProjectRequest(BaseModel):
    """Wrapper for project graph in API requests."""
    project: ProjectGraph
