import hashlib
import sqlite3

import pytest

from ops.review_backup import backup_review_store, inspect_review_store, verify_backup
from service.review_workflow import _connect, canonical, create_case


def make_case(path):
    db = _connect(path)
    db.close()
    packet = {"schema_version": 1, "analysis_key": "a" * 64,
              "review": {"navigation_suitability": "not_assessed", "decision": None}}
    packet["integrity"] = {"sha256": hashlib.sha256(canonical(packet)).hexdigest()}
    return create_case(path, packet, {"iss": "test", "sub": "analyst", "roles": ["analyst"]})


def test_online_backup_can_be_verified_without_overwriting(tmp_path):
    source = tmp_path / "live.sqlite3"
    make_case(source)
    target = tmp_path / "backups" / "review-20261002.sqlite3"
    record = backup_review_store(source, target)
    assert record["case_count"] == 1
    assert record["event_count"] == 0
    assert verify_backup(target)["sha256"] == record["sha256"]
    with pytest.raises(FileExistsError):
        backup_review_store(source, target)


def test_backup_rejects_tampered_review_chain(tmp_path):
    source = tmp_path / "live.sqlite3"
    case = make_case(source)
    with sqlite3.connect(source) as db:
        db.execute("UPDATE review_cases SET packet_sha256=? WHERE id=?", ("0" * 64, case["case_id"]))
    with pytest.raises(ValueError, match="integrity"):
        inspect_review_store(source)
    target = tmp_path / "backup.sqlite3"
    with pytest.raises(ValueError, match="integrity"):
        backup_review_store(source, target)
    assert not target.exists()
