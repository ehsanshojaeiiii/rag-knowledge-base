from fastapi.testclient import TestClient


def test_health_reports_database_and_pgvector(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pgvector"].startswith("0.8")
