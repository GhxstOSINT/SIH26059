"""Check a running Southern Passage research service without exposing secrets."""
from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def validate_base_url(value: str) -> str:
    url = urlsplit(value)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username
            or url.password or url.path not in {"", "/"} or url.query or url.fragment):
        raise ValueError("Provide an origin URL without credentials, path, query or fragment")
    if url.scheme != "https" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote smoke checks require HTTPS")
    return value.rstrip("/")


def request_status(origin: str, path: str, token: str, *, json_response: bool = True) -> tuple[int, dict]:
    headers = {"Accept": "application/json" if json_response else "text/html"}
    if token and path.startswith("/api/"):
        headers["Authorization"] = f"Bearer {token}"
    request = Request(origin + path, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response) if json_response else {}
            return response.status, payload
    except HTTPError as exc:
        return exc.code, {}
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return 0, {}


def smoke(origin: str, token: str = "") -> dict:
    origin = validate_base_url(origin)
    checks = {}
    for name, path, json_response in (
        ("health", "/healthz", True),
        ("readiness", "/readyz", True),
        ("portal", "/portal/", False),
        ("model_selection", "/api/v1/model/selection", True),
        ("multiplatform_evidence", "/api/v1/evaluations/multiplatform-corridors", True),
        ("integration", "/api/v1/integration/status", True),
        ("review_capabilities", "/api/v1/review-workflow/capabilities", True),
    ):
        status, payload = request_status(origin, path, token, json_response=json_response)
        checks[name] = {"http_status": status, "ok": status == 200}
        if name == "readiness":
            checks[name]["model_ready"] = payload.get("ready") is True
        elif name == "model_selection":
            checks[name]["verified_research_model"] = (payload.get("verified") is True
                                                         and payload.get("operational_decision_ready") is False)
        elif name == "multiplatform_evidence":
            checks[name]["quality_gate"] = payload.get("quality_gate")
            checks[name]["operational_decision_ready"] = payload.get("operational_decision_ready")
        elif name == "integration":
            checks[name]["active_observation_status"] = ((payload.get("research_components") or {})
                                                           .get("gated_active_observations") or {}).get("status")
            checks[name]["operational_decision_ready"] = payload.get("operational_decision_ready")
        elif name == "review_capabilities":
            checks[name]["enabled"] = payload.get("enabled")
    essential = ("health", "readiness", "portal", "model_selection", "multiplatform_evidence", "integration")
    ready = (all(checks[name]["ok"] for name in essential)
             and checks["readiness"]["model_ready"]
             and checks["model_selection"]["verified_research_model"]
             and checks["multiplatform_evidence"]["quality_gate"] == "insufficient_evidence"
             and checks["multiplatform_evidence"]["operational_decision_ready"] is False
             and checks["integration"]["operational_decision_ready"] is False)
    return {"schema_version": 1, "origin": origin, "research_service_ready": ready,
            "operational_decision_ready": False, "checks": checks,
            "warning": "This checks service packaging and research evidence, not source freshness, SSO assurance, scientific skill or navigation safety."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    try:
        result = smoke(args.base_url, os.environ.get("SOUTHERN_PASSAGE_API_TOKEN", ""))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0 if result["research_service_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
