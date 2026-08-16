"""
Tests for API rate limiting, per-key RPM limits, and usage accounting.
"""

from fastapi.testclient import TestClient
from src.core.key_manager import APIKeyManager

def test_rate_limit_exceeded_returns_429(client: TestClient, isolated_key_manager: APIKeyManager):
    key_record = isolated_key_manager.create_key(name="rate-limit-test", role="user", rate_limit_rpm=3)
    headers = {"Authorization": f"Bearer {key_record['key']}"}

    for _ in range(3):
        res = client.get("/v1/models", headers=headers)
        assert res.status_code == 200

    res_4 = client.get("/v1/models", headers=headers)
    assert res_4.status_code == 429
    data = res_4.json()
    assert "error" in data
    assert data["error"]["type"] == "rate_limit_error"
    assert data["error"]["code"] == 429

def test_admin_unlimited_rate(client: TestClient, admin_headers):
    for _ in range(20):
        res = client.get("/v1/models", headers=admin_headers)
        assert res.status_code == 200

def test_isolated_rate_counters_per_key(client: TestClient, isolated_key_manager: APIKeyManager):
    key1 = isolated_key_manager.create_key(name="k1", role="user", rate_limit_rpm=2)
    key2 = isolated_key_manager.create_key(name="k2", role="user", rate_limit_rpm=5)

    h1 = {"Authorization": f"Bearer {key1['key']}"}
    h2 = {"Authorization": f"Bearer {key2['key']}"}

    assert client.get("/v1/models", headers=h1).status_code == 200
    assert client.get("/v1/models", headers=h1).status_code == 200
    assert client.get("/v1/models", headers=h1).status_code == 429

    assert client.get("/v1/models", headers=h2).status_code == 200
    assert client.get("/v1/models", headers=h2).status_code == 200
