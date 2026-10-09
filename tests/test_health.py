from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "health" in data
    assert "version" in data


def test_health_check_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    components = data["components"]
    assert "api" in components
    assert components["api"]["status"] == "ok"
    assert "database" in components
    assert "storage" in components
