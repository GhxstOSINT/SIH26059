"""Create and verify a consistent backup of the local review ledger.

The backup is a recoverability aid, not an immutable audit destination.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
from contextlib import closing

from fastapi import HTTPException

from service.review_workflow import _view


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_review_store(path: Path) -> dict[str, int]:
    """Check SQLite structure, foreign keys, and every packet/event chain."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("Review database is missing or is a symbolic link")
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Unsupported review database schema")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("Review database foreign-key check failed")
        try:
            case_ids = [row["id"] for row in db.execute("SELECT id FROM review_cases ORDER BY id")]
            for case_id in case_ids:
                _view(db, case_id)
            event_count = db.execute("SELECT COUNT(*) FROM review_events").fetchone()[0]
        except (sqlite3.Error, HTTPException) as exc:
            raise ValueError("Review packet or event-chain integrity check failed") from exc
    return {"case_count": len(case_ids), "event_count": event_count}


def backup_review_store(source: Path, target: Path) -> dict[str, Any]:
    """Use SQLite's online backup API; never overwrite an existing backup."""
    if not source.is_file() or source.is_symlink():
        raise ValueError("Source review database is missing or is a symbolic link")
    if target.suffix != ".sqlite3" or source.resolve() == target.resolve():
        raise ValueError("Select a distinct .sqlite3 backup target")
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    if target.exists() or manifest_path.exists():
        raise FileExistsError("Backup target or manifest already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".review-backup-", suffix=".sqlite3.tmp",
                                         dir=target.parent, delete=False) as item:
            backup_temp = Path(item.name)
        source_uri = source.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(source_uri, uri=True, timeout=10)) as live:
            with closing(sqlite3.connect(backup_temp)) as backup:
                live.backup(backup, pages=128, sleep=0.1)
        counts = inspect_review_store(backup_temp)
        record = {"schema_version": 1, "scope": "research_review_backup",
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "database_file": target.name, "sha256": sha256_file(backup_temp),
                  **counts, "operational_decision_ready": False}
        with tempfile.NamedTemporaryFile(prefix=".review-backup-manifest-", suffix=".json.tmp",
                                         dir=target.parent, delete=False, mode="w", encoding="utf-8") as item:
            manifest_temp = Path(item.name)
            json.dump(record, item, indent=2)
            item.write("\n")
        # Neither destination existed at entry. Refuse a late collision too.
        if target.exists() or manifest_path.exists():
            raise FileExistsError("Backup target appeared during creation")
        os.replace(backup_temp, target)
        backup_temp = None
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
        return record
    finally:
        for temporary in (backup_temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def verify_backup(target: Path) -> dict[str, Any]:
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (record.get("schema_version") != 1 or record.get("database_file") != target.name
            or record.get("scope") != "research_review_backup"
            or record.get("operational_decision_ready") is not False
            or record.get("sha256") != sha256_file(target)):
        raise ValueError("Review backup manifest or file digest mismatch")
    counts = inspect_review_store(target)
    if any(record.get(key) != value for key, value in counts.items()):
        raise ValueError("Review backup case/event counts differ from the manifest")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Current review SQLite database")
    parser.add_argument("--output", type=Path, required=True, help="New .sqlite3 backup path")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.source is None:
        parser.error("--source is required unless --verify-only is set")
    try:
        record = verify_backup(args.output) if args.verify_only else backup_review_store(args.source, args.output)
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "reason": str(exc)}))
        return 1
    print(json.dumps({"verified": True, **record}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
