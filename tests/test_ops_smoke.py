import pytest

from ops.smoke import smoke, validate_base_url


def test_smoke_origin_rejects_remote_plain_http_and_embedded_credentials():
    assert validate_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    for invalid in ("http://example.org", "https://user:pass@example.org", "https://example.org/portal/",
                    "https://example.org/?token=x"):
        with pytest.raises(ValueError):
            validate_base_url(invalid)


def test_smoke_does_not_mark_weak_or_missing_evidence_ready(monkeypatch):
    def probe(_origin, path, _token, *, json_response=True):
        if path == "/readyz":
            return 200, {"ready": True}
        if path == "/api/v1/model/selection":
            return 200, {"verified": True, "operational_decision_ready": False}
        if path == "/api/v1/evaluations/multiplatform-corridors":
            return 200, {"quality_gate": "insufficient_evidence", "operational_decision_ready": False}
        if path == "/api/v1/integration/status":
            return 200, {"operational_decision_ready": False}
        return 200, {}
    monkeypatch.setattr("ops.smoke.request_status", probe)
    result = smoke("http://localhost:8000")
    assert result["research_service_ready"] is True
    assert result["operational_decision_ready"] is False
    monkeypatch.setattr("ops.smoke.request_status", lambda *_args, **_kwargs: (401, {}))
    assert smoke("http://localhost:8000")["research_service_ready"] is False
