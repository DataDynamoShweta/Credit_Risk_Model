import time

from fastapi.testclient import TestClient

from api.main import app


def test_planner_endpoint_served():
    client = TestClient(app)
    response = client.get("/planner")
    assert response.status_code == 200
    assert "Credit Risk" in response.text
    assert "Agent Orchestrator" in response.text


def test_workflow_lifecycle_end_to_end():
    client = TestClient(app)

    # 1. Trigger workflow run
    run_payload = {
        "dataset": "synthetic",
        "target": "target",
        "champion": "logistic_regression",
        "challenger": "hist_gradient_boosting",
        "oot_pct": 20,
        "seed": 42,
    }
    resp = client.post("/workflows/credit-risk/run", json=run_payload)
    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data
    run_id = data["run_id"]
    assert data["status"] == "running"

    # 2. Wait until graph reaches governance interrupt (awaiting_approval)
    awaiting = False
    for _ in range(30):
        time.sleep(0.5)
        st_resp = client.get(f"/runs/{run_id}")
        assert st_resp.status_code == 200
        st = st_resp.json()
        if st.get("awaiting_approval"):
            awaiting = True
            break
        if st.get("status") in ("completed", "failed"):
            break

    assert awaiting, f"status={st.get('status')}, error={st.get('error_message')}, metrics={st.get('validation_metrics')}"
    assert "auroc" in (st.get("validation_metrics") or {})

    # 3. Verify event stream
    ev_resp = client.get(f"/runs/{run_id}/events")
    assert ev_resp.status_code == 200
    events = ev_resp.json().get("events", [])
    assert len(events) >= 3

    # 4. Approve model promotion
    approve_resp = client.post(
        f"/workflows/credit-risk/{run_id}/approve",
        json={
            "decision": "APPROVE",
            "comment": "Scorecard exceeds minimum discrimination hurdles. Approved.",
            "reviewer_id": "MRO-Lead-Reviewer",
        },
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json().get("governance_status") == "APPROVED"

    # 5. Verify run reached completion
    final_resp = client.get(f"/runs/{run_id}")
    final_data = final_resp.json()
    assert final_data.get("status") == "completed"
    assert final_data.get("progress_pct") == 100

    # 6. Verify artifacts manifest
    art_resp = client.get(f"/runs/{run_id}/artifacts")
    assert art_resp.status_code == 200
    artifacts = [a["name"] for a in art_resp.json().get("artifacts", [])]
    assert "quality_summary.json" in artifacts
    assert "model_card.json" in artifacts


def test_flat_feature_predict_scoring(trained_project):
    client = TestClient(app)
    flat_payload = {
        "loan_amnt": 12000,
        "int_rate": 10.5,
        "annual_inc": 80000,
        "dti": 14.5,
        "emp_length": 5,
        "home_ownership": "MORTGAGE",
    }
    resp = client.post("/predict", json=flat_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "probability_of_default" in body
    assert "score" in body
    assert 300 <= body["score"] <= 850

