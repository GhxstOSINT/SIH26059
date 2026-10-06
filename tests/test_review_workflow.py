import base64
import hashlib
import hmac
import sqlite3
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import service.main as main
from service.review_workflow import canonical, verify_gateway_assertion


client = TestClient(main.app)
SECRET = "review-gateway-secret-for-tests-only-2026"
ROUTE = {"route_id": "Research route", "observation_date": "2024-08-31",
         "analysis_half_width_km": 5,
         "geometry": {"type": "LineString", "coordinates": [[-60.2, -66.0], [-59.8, -66.0]]}}
SUBMISSION = {"route": ROUTE, "expected_source_sha256": "e" * 64}


def assertion(subject, roles, method, path, request_id="test-request"):
    now = int(time.time())
    payload = {"iss": "https://gateway.example.test", "aud": "southern-passage-review",
               "sub": subject, "roles": roles, "iat": now, "exp": now + 90,
               "method": method, "path": path, "request_id": request_id}
    encoded = base64.urlsafe_b64encode(canonical(payload)).rstrip(b"=").decode()
    signature = base64.urlsafe_b64encode(hmac.new(SECRET.encode(), encoded.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    return f"{encoded}.{signature}"


def headers(subject, roles, method, path):
    return {"X-Request-ID": "test-request",
            "X-SP-Actor-Assertion": assertion(subject, roles, method, path)}


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "REVIEW_WORKFLOW_ENABLED", True)
    monkeypatch.setattr(main, "REVIEW_ASSERTION_SECRET", SECRET)
    monkeypatch.setattr(main, "REVIEW_ASSERTION_ISSUER", "https://gateway.example.test")
    monkeypatch.setattr(main, "REVIEW_ASSERTION_AUDIENCE", "southern-passage-review")
    monkeypatch.setattr(main, "REVIEW_DB_PATH", tmp_path / "review" / "cases.sqlite3")
    def packet_for_route(route, request):
        packet = {"schema_version": 1, "analysis_key": "a" * 64,
                  "review": {"navigation_suitability": "not_assessed", "decision": None},
                  "observation": {"source_provenance": {"source_sha256": "e" * 64}},
                  "route": route.model_dump(mode="json")}
        packet["integrity"] = {"sha256": hashlib.sha256(canonical(packet)).hexdigest()}
        return packet
    monkeypatch.setattr(main, "create_route_review_packet", packet_for_route)


def test_identity_assertion_rejects_spoofing_and_request_replay():
    token = assertion("analyst-1", ["analyst"], "POST", "/api/v1/review-workflow/cases")
    actor = verify_gateway_assertion(token, secret=SECRET, issuer="https://gateway.example.test",
                                     audience="southern-passage-review", method="POST",
                                     path="/api/v1/review-workflow/cases", request_id="test-request")
    assert actor["sub"] == "analyst-1"
    for changed in ({"request_id": "another"}, {"method": "GET"}, {"path": "/api/v1/review-workflow/me"}):
        arguments = {"secret": SECRET, "issuer": "https://gateway.example.test",
                     "audience": "southern-passage-review", "method": "POST",
                     "path": "/api/v1/review-workflow/cases", "request_id": "test-request"}
        arguments.update(changed)
        with pytest.raises(HTTPException) as failure:
            verify_gateway_assertion(token, **arguments)
        assert failure.value.status_code == 401
    with pytest.raises(HTTPException):
        verify_gateway_assertion("X" + token[1:], secret=SECRET, issuer="https://gateway.example.test",
                                 audience="southern-passage-review", method="POST",
                                 path="/api/v1/review-workflow/cases", request_id="test-request")
    with pytest.raises(HTTPException):
        verify_gateway_assertion(token, secret=SECRET, issuer="https://gateway.example.test",
                                 audience="southern-passage-review", method="POST",
                                 path="/api/v1/review-workflow/cases", request_id="test-request",
                                 now=int(time.time()) + 400)


def test_workflow_fails_closed_without_gateway_identity(configured):
    assert client.get("/api/v1/review-workflow/capabilities").json()["enabled"] is True
    assert client.get("/api/v1/review-workflow/me").status_code == 401
    path = "/api/v1/review-workflow/cases"
    assert client.post(path, json=SUBMISSION).status_code == 401
    response = client.post(path, headers=headers("observer-1", ["auditor"], "POST", path), json=SUBMISSION)
    assert response.status_code == 403
    changed = {**SUBMISSION, "expected_source_sha256": "f" * 64}
    assert client.post(path, headers=headers("author-1", ["analyst"], "POST", path), json=changed).status_code == 409


def test_production_requires_workload_bearer_and_actor_assertion(configured, monkeypatch):
    monkeypatch.setattr(main, "APP_ENV", "production")
    monkeypatch.setattr(main, "API_AUTH_TOKEN", "service-workload-secret-for-tests-only-2026")
    path = "/api/v1/review-workflow/me"
    identity = headers("analyst-1", ["analyst"], "GET", path)
    assert client.get(path, headers=identity).status_code == 401
    identity["Authorization"] = "Bearer service-workload-secret-for-tests-only-2026"
    assert client.get(path, headers=identity).json()["sub"] == "analyst-1"
    assert client.get(path, headers={"Authorization": identity["Authorization"]}).status_code == 401


def test_independent_gate_signoffs_and_handoff_never_grant_clearance(configured):
    path = "/api/v1/review-workflow/cases"
    created = client.post(path, headers=headers("author-1", ["analyst"], "POST", path), json=SUBMISSION)
    assert created.status_code == 201
    case = created.json()
    assert case["status"] == "in_review"
    assert case["navigation_clearance"] is False
    detail_path = f"{path}/{case['case_id']}"
    assert client.get(detail_path, headers=headers("audit-1", ["auditor"], "GET", detail_path)).status_code == 200
    decision_path = f"{detail_path}/decisions"
    evidence = [{"record_id": "DMS/REVIEW-2026-01", "sha256": "c" * 64}]
    body = {"gate": "science", "decision": "reviewed", "rationale": "Dataset lineage and known limits have been reviewed.", "evidence": evidence}
    assert client.post(decision_path, headers=headers("author-1", ["science_reviewer", "analyst"], "POST", decision_path), json=body).status_code == 403
    assert client.post(decision_path, headers=headers("science-1", ["science_reviewer"], "POST", decision_path), json={**body, "evidence": []}).status_code == 422
    first = client.post(decision_path, headers=headers("science-1", ["science_reviewer"], "POST", decision_path), json=body)
    assert first.status_code == 200
    assert first.json()["events"][0]["previous_hash"] == case["packet_sha256"]
    assert client.post(decision_path, headers=headers("science-1", ["science_reviewer"], "POST", decision_path), json=body).status_code == 403
    assert client.post(decision_path, headers=headers("science-2", ["science_reviewer"], "POST", decision_path), json=body).status_code == 409
    for gate, role, subject in (("maritime", "maritime_reviewer", "maritime-1"),
                                ("security", "security_reviewer", "security-1"),
                                ("operations", "operations_reviewer", "operations-1")):
        body["gate"] = gate
        response = client.post(decision_path, headers=headers(subject, [role], "POST", decision_path), json=body)
        assert response.status_code == 200
    assert response.json()["status"] == "awaiting_release_handoff"
    body.update(gate="handoff", decision="package")
    assert client.post(decision_path, headers=headers("science-1", ["release_authority"], "POST", decision_path), json=body).status_code == 403
    final = client.post(decision_path, headers=headers("release-1", ["release_authority"], "POST", decision_path), json=body)
    assert final.status_code == 200
    assert final.json()["status"] == "pilot_review_packaged"
    assert final.json()["navigation_clearance"] is False
    assert len({event["actor_subject"] for event in final.json()["events"]}) == 5
    assert client.get(path, headers=headers("audit-1", ["auditor"], "GET", path)).json()["items"][0]["case_id"] == case["case_id"]


def test_blocked_gate_closes_case(configured):
    path = "/api/v1/review-workflow/cases"
    case = client.post(path, headers=headers("author-1", ["analyst"], "POST", path), json=SUBMISSION).json()
    decision_path = f"{path}/{case['case_id']}/decisions"
    blocked = client.post(decision_path, headers=headers("science-1", ["science_reviewer"], "POST", decision_path),
                          json={"gate": "science", "decision": "blocked", "rationale": "Source coverage is insufficient for the requested review.", "evidence": []})
    assert blocked.status_code == 200
    assert blocked.json()["status"] == "changes_required"
    another = client.post(decision_path, headers=headers("maritime-1", ["maritime_reviewer"], "POST", decision_path),
                          json={"gate": "maritime", "decision": "reviewed", "rationale": "This would be a separate review, but the case is closed.",
                                "evidence": [{"record_id": "DMS/REVIEW-2026-02", "sha256": "d" * 64}]})
    assert another.status_code == 409


def test_tampered_event_is_detected_on_read(configured):
    path = "/api/v1/review-workflow/cases"
    case = client.post(path, headers=headers("author-1", ["analyst"], "POST", path), json=SUBMISSION).json()
    decision_path = f"{path}/{case['case_id']}/decisions"
    response = client.post(decision_path, headers=headers("science-1", ["science_reviewer"], "POST", decision_path),
                           json={"gate": "science", "decision": "blocked",
                                 "rationale": "The observed source is too old for this purpose.", "evidence": []})
    assert response.status_code == 200
    with sqlite3.connect(main.REVIEW_DB_PATH) as db:
        db.execute("UPDATE review_events SET rationale='altered' WHERE case_id=?", (case["case_id"],))
    detail_path = f"{path}/{case['case_id']}"
    assert client.get(detail_path, headers=headers("audit-1", ["auditor"], "GET", detail_path)).status_code == 503
