"""
Tests for public utility endpoints and metadata routes.
"""

from fastapi.testclient import TestClient

def test_root_dashboard(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cogito-0.9.1-15B API Dashboard" in response.text

def test_health_endpoint_not_loaded(client: TestClient, mock_engine):
    mock_engine.model_loaded = False
    mock_engine.model_loading = False
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert data["model"] == "Cogito-0.9.1-15B"
    assert "uptime" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data

def test_health_endpoint_loaded(client: TestClient, mock_engine):
    mock_engine.model_loaded = True
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["model_loaded"] is True

def test_ping_endpoint(client: TestClient):
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["pong"] is True
    assert "ts" in data
    assert isinstance(data["ts"], int)

def test_models_list_endpoint(client: TestClient, user_headers):
    response = client.get("/v1/models", headers=user_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "Cogito-0.9.1-15B"
    assert data["data"][0]["owned_by"] == "ozaa77"

def test_embeddings_stub_endpoint(client: TestClient, user_headers):
    response = client.post("/v1/embeddings", headers=user_headers, json={
        "model": "Cogito-0.9.1-15B",
        "input": "test input string"
    })
    assert response.status_code == 501
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 501
