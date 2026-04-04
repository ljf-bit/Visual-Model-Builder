"""
Placeholder test for backend health check.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_validate_graph_placeholder():
    """Test that validate-graph endpoint returns valid response structure."""
    project = {
        "project": {
            "version": "1.0.0",
            "metadata": {
                "name": "Test",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z"
            },
            "nodes": [],
            "edges": []
        }
    }
    response = client.post("/validate-graph", json=project)
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data


def test_generate_code_placeholder():
    """Test that generate-code endpoint returns valid response structure."""
    project = {
        "project": {
            "version": "1.0.0",
            "metadata": {
                "name": "Test",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z"
            },
            "nodes": [],
            "edges": []
        }
    }
    response = client.post("/generate-code", json=project)
    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert "code" in data
