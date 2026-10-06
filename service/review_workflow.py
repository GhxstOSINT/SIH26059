"""Gateway-attested research review ledger. Not a navigation clearance system."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

GATES = {
    "science": "science_reviewer",
    "maritime": "maritime_reviewer",
    "security": "security_reviewer",
    "operations": "operations_reviewer",
}
ALL_ROLES = frozenset({"analyst", "auditor", "release_authority", *GATES.values()})
SUBJECT_RE = re.compile(r"^[A-Za-z0-9._:@/-]{3,160}$")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def verify_gateway_assertion(token: str, *, secret: str, issuer: str, audience: str,
                             method: str, path: str, request_id: str,
                             now: int | None = None) -> dict[str, Any]:
    """Verify a short-lived, request-bound HMAC identity envelope from the SSO gateway.

    Format: base64url(canonical JSON payload).base64url(HMAC-SHA256(payload segment)).
    This is an internal gateway protocol, deliberately not an OIDC ID token or JWT.
    """
    if not secret or len(secret) < 32 or len(token) > 4096 or token.count(".") != 1:
        raise HTTPException(401, "A valid gateway identity assertion is required.")
    encoded, signature = token.split(".", 1)
    try:
        expected = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
        supplied = _b64decode(signature)
        payload = json.loads(_b64decode(encoded))
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(401, "A valid gateway identity assertion is required.") from None
    if not hmac.compare_digest(expected, supplied) or not isinstance(payload, dict):
        raise HTTPException(401, "A valid gateway identity assertion is required.")
    clock = int(time.time()) if now is None else now
    issued, expires = payload.get("iat"), payload.get("exp")
    subject, roles = payload.get("sub"), payload.get("roles")
    if (payload.get("iss") != issuer or payload.get("aud") != audience
            or payload.get("method") != method or payload.get("path") != path
            or payload.get("request_id") != request_id
            or type(issued) is not int or type(expires) is not int
            or issued > clock + 30 or issued < clock - 300
            or expires <= clock or expires > issued + 300
            or not isinstance(subject, str) or not SUBJECT_RE.fullmatch(subject)
            or not isinstance(roles, list) or not roles or len(roles) > 12
            or any(not isinstance(role, str) or role not in ALL_ROLES for role in roles)):
        raise HTTPException(401, "A valid gateway identity assertion is required.")
    return {"iss": issuer, "sub": subject, "roles": sorted(set(roles))}


def require_role(actor: dict[str, Any], role: str) -> None:
    if role not in actor["roles"]:
        raise HTTPException(403, f"The {role} role is required.")


def _connect(path: Path) -> sqlite3.Connection:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA foreign_keys=ON")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise sqlite3.DatabaseError("Unsupported review schema version.")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS review_cases (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                creator_issuer TEXT NOT NULL, creator_subject TEXT NOT NULL,
                analysis_key TEXT NOT NULL, packet_sha256 TEXT NOT NULL,
                packet_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_events (
                id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES review_cases(id),
                gate TEXT NOT NULL, decision TEXT NOT NULL,
                actor_issuer TEXT NOT NULL, actor_subject TEXT NOT NULL,
                actor_roles_json TEXT NOT NULL, rationale TEXT NOT NULL,
                evidence_json TEXT NOT NULL, request_id TEXT NOT NULL,
                created_at TEXT NOT NULL, previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL, UNIQUE(case_id, gate)
            );
            CREATE INDEX IF NOT EXISTS review_events_case_idx ON review_events(case_id, created_at);
        """)
        if version == 0:
            db.execute("PRAGMA user_version=1")
        return db
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(503, "Review record storage is unavailable.") from exc


@contextmanager
def review_db(path: Path):
    db = _connect(path)
    try:
        yield db
    finally:
        db.close()


def _event_rows(db: sqlite3.Connection, case_id: str) -> list[dict[str, Any]]:
    rows = db.execute("SELECT * FROM review_events WHERE case_id=? ORDER BY created_at, rowid", (case_id,)).fetchall()
    return [{**dict(row), "actor_roles": json.loads(row["actor_roles_json"]),
             "evidence": json.loads(row["evidence_json"])} for row in rows]


def _view(db: sqlite3.Connection, case_id: str, include_packet: bool = True) -> dict[str, Any]:
    row = db.execute("SELECT * FROM review_cases WHERE id=?", (case_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Review case not found.")
    try:
        packet = json.loads(row["packet_json"])
        packet_digest = hashlib.sha256(canonical({key: value for key, value in packet.items()
                                                   if key != "integrity"})).hexdigest()
        if packet_digest != row["packet_sha256"] or packet.get("integrity", {}).get("sha256") != packet_digest:
            raise ValueError("packet digest mismatch")
        events = _event_rows(db, case_id)
        previous_hash = row["packet_sha256"]
        for event in events:
            content = {key: event[key] for key in ("id", "case_id", "gate", "decision", "actor_issuer",
                                                    "actor_subject", "actor_roles", "rationale", "evidence",
                                                    "request_id", "created_at", "previous_hash")}
            if (event["previous_hash"] != previous_hash
                    or hashlib.sha256(canonical(content)).hexdigest() != event["event_hash"]):
                raise ValueError("event chain mismatch")
            previous_hash = event["event_hash"]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "Review record integrity check failed.") from exc
    decisions = {event["gate"]: event["decision"] for event in events}
    if "handoff" in decisions:
        status = "pilot_review_packaged"
    elif any(value in {"blocked", "changes_required"} for value in decisions.values()):
        status = "changes_required"
    elif all(decisions.get(gate) == "reviewed" for gate in GATES):
        status = "awaiting_release_handoff"
    else:
        status = "in_review"
    result = {"case_id": row["id"], "created_at": row["created_at"],
              "creator": {"iss": row["creator_issuer"], "sub": row["creator_subject"]},
              "route_label": packet.get("route", {}).get("route_id", "Route review"),
              "analysis_key": row["analysis_key"], "packet_sha256": row["packet_sha256"],
              "status": status, "gate_roles": GATES, "decisions": decisions,
              "events": [{key: value for key, value in event.items()
                          if key not in {"actor_roles_json", "evidence_json"}} for event in events],
              "navigation_clearance": False,
              "warning": "Review completion records process attestations only; it does not certify a route or authorize navigation."}
    if include_packet:
        result["packet"] = packet
    return result


def create_case(path: Path, packet: dict[str, Any], actor: dict[str, Any]) -> dict[str, Any]:
    require_role(actor, "analyst")
    if packet.get("review", {}).get("navigation_suitability") != "not_assessed":
        raise HTTPException(422, "Only non-clearance evidence packets may be filed.")
    digest = hashlib.sha256(canonical({key: value for key, value in packet.items()
                                       if key != "integrity"})).hexdigest()
    if packet.get("integrity", {}).get("sha256") != digest:
        raise HTTPException(422, "Evidence packet integrity check failed.")
    case_id = str(uuid.uuid4())
    record = (case_id, datetime.now(timezone.utc).isoformat(), actor["iss"], actor["sub"],
              packet["analysis_key"], packet["integrity"]["sha256"], canonical(packet).decode("utf-8"))
    with review_db(path) as db:
        db.execute("INSERT INTO review_cases VALUES (?,?,?,?,?,?,?)", record)
        return _view(db, case_id)


def list_cases(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    with review_db(path) as db:
        ids = db.execute("SELECT id FROM review_cases ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_view(db, row["id"], include_packet=False) for row in ids]


def get_case(path: Path, case_id: str) -> dict[str, Any]:
    with review_db(path) as db:
        return _view(db, case_id)


def append_decision(path: Path, case_id: str, *, gate: str, decision: str,
                    actor: dict[str, Any], rationale: str,
                    evidence: list[dict[str, str]], request_id: str) -> dict[str, Any]:
    role = GATES.get(gate) if gate != "handoff" else "release_authority"
    if role is None:
        raise HTTPException(422, "Unknown review gate.")
    require_role(actor, role)
    if gate == "handoff" and decision != "package":
        raise HTTPException(422, "The handoff decision must be package.")
    if gate != "handoff" and decision not in {"reviewed", "blocked", "changes_required"}:
        raise HTTPException(422, "Unsupported review decision.")
    if decision in {"reviewed", "package"} and not evidence:
        raise HTTPException(422, "Evidence references are required for an affirmative attestation.")
    with review_db(path) as db:
        try:
            db.execute("BEGIN IMMEDIATE")
            current = _view(db, case_id)
            if (current["creator"]["iss"], current["creator"]["sub"]) == (actor["iss"], actor["sub"]):
                raise HTTPException(403, "A case creator cannot attest their own case.")
            if any((event["actor_issuer"], event["actor_subject"]) == (actor["iss"], actor["sub"])
                   for event in current["events"]):
                raise HTTPException(403, "Each gate requires a distinct reviewer.")
            if gate in current["decisions"]:
                raise HTTPException(409, "This gate already has an immutable decision. Create a new case for revisions.")
            if current["status"] in {"changes_required", "pilot_review_packaged"}:
                raise HTTPException(409, "This case is closed to further review.")
            if gate == "handoff" and current["status"] != "awaiting_release_handoff":
                raise HTTPException(409, "All four independent gates must be reviewed before packaging.")
            previous = current["events"][-1]["event_hash"] if current["events"] else current["packet_sha256"]
            event = {"id": str(uuid.uuid4()), "case_id": case_id, "gate": gate,
                     "decision": decision, "actor_issuer": actor["iss"], "actor_subject": actor["sub"],
                     "actor_roles": actor["roles"], "rationale": rationale,
                     "evidence": evidence, "request_id": request_id,
                     "created_at": datetime.now(timezone.utc).isoformat(), "previous_hash": previous}
            digest = hashlib.sha256(canonical(event)).hexdigest()
            db.execute("INSERT INTO review_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (event["id"], case_id, gate, decision, actor["iss"], actor["sub"],
                        canonical(actor["roles"]).decode(), rationale, canonical(evidence).decode(),
                        request_id, event["created_at"], previous, digest))
            db.commit()
            return _view(db, case_id)
        except sqlite3.Error as exc:
            db.rollback()
            raise HTTPException(503, "Review record storage is unavailable.") from exc
