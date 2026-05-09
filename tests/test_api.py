"""Tests para la API FastAPI."""
import pytest
from fastapi.testclient import TestClient


def test_api_root():
    from app.main import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_api_stats():
    from app.main import app
    client = TestClient(app)
    response = client.get("/stats")
    assert response.status_code == 200
    assert "chunks_indexed" in response.json()


def test_api_ask_empty_query():
    from app.main import app
    client = TestClient(app)
    response = client.post("/ask", json={"query": ""})
    assert response.status_code == 400
