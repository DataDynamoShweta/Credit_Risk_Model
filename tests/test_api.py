from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint(project):
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_card_discloses_demo_status(project):
    response = TestClient(app).get("/model-card")
    assert response.status_code == 200
    assert "not-approved-for-production" in response.json()["status"]
