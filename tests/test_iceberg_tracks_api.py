import hashlib
import json
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient
import pytest

import service.main as service_main


client = TestClient(service_main.app)


def _archive(tmp_path: Path) -> tuple[Path, str]:
    archive = tmp_path / "consolidated_database_v8.0.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("A23a.csv", "date,ascat_1,ascat_2,ascat_3\n"
                        "2025001,-65,-55,0\n"
                        "2025005,-64,-54,0\n"
                        "2025008,-63,-53,1\n"
                        "2025010,-62,-52,0\n"
                        "2025011,-61,-51,0\n")
        output.writestr("B09B.csv", "date,ascat_1,ascat_2,ascat_3\n"
                        "2024366,-70,-40,0\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "dataset": "BYU/NIC Antarctic Iceberg Tracking Database",
        "version": "8.0",
        "complete": True,
        "archive": {"file": archive.name, "sha256": digest},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return archive, digest


def test_observed_iceberg_track_api_returns_only_measured_positions(tmp_path, monkeypatch):
    archive, digest = _archive(tmp_path)
    monkeypatch.setattr(service_main, "ICEBERG_TRACKS_PATH", archive)

    response = client.get("/api/v1/icebergs/observed-tracks",
                          params={"as_of_date": "2025-01-10", "window_days": 10})

    assert response.status_code == 200
    report = response.json()
    assert report["status"] == "historical_observations_only"
    assert report["dataset"]["archive_sha256"] == digest
    assert report["query"] == {
        "as_of_date": "2025-01-10", "window_days": 10, "window_start": "2025-01-01"
    }
    feature = next(item for item in report["features"] if item["id"] == "A23a")
    assert len(feature["geometry"]["coordinates"]) == 3
    assert feature["geometry"]["coordinates"][0] == pytest.approx([-55.0, -65.0])
    assert feature["geometry"]["coordinates"][1] == pytest.approx([-54.0, -64.0])
    assert feature["geometry"]["coordinates"][2] == pytest.approx([-52.0, -62.0])
    assert feature["properties"]["last_observed_date"] == "2025-01-10"
    assert feature["properties"]["age_days"] == 0
    assert feature["properties"]["source_measured_position_count"] == 3
    assert all(item["id"] != "B09B" for item in report["features"])
    assert "not" in report["warning"]


def test_observed_iceberg_track_api_withholds_unconfirmed_archive_period(tmp_path, monkeypatch):
    monkeypatch.setattr(service_main, "ICEBERG_TRACKS_PATH", tmp_path / "missing.zip")

    response = client.get("/api/v1/icebergs/observed-tracks",
                          params={"as_of_date": "2025-04-23"})

    assert response.status_code == 422
    assert "published BYU/NIC v8.0 coverage ends on 2025-04-22" in response.json()["detail"]


def test_observed_iceberg_track_api_rejects_archive_checksum_mismatch(tmp_path, monkeypatch):
    archive, _ = _archive(tmp_path)
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(service_main, "ICEBERG_TRACKS_PATH", archive)

    response = client.get("/api/v1/icebergs/observed-tracks",
                          params={"as_of_date": "2025-01-10"})

    assert response.status_code == 503
    assert "checksum does not match" in response.json()["detail"]


def test_historical_projection_uses_measured_points_and_matching_evidence(tmp_path, monkeypatch):
    archive, digest = _archive(tmp_path)
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps({
        "archive_sha256": digest,
        "task": "Observed-position-only next-report displacement error; not a fixed-time operational forecast",
        "test_target_period": {"start": "2025-01-01", "end": "2025-04-22"},
        "metrics": {"1-2d": {
            "constant_velocity": {"mean_error_km": 2.5, "count": 100},
            "persistence": {"mean_error_km": 3.1, "count": 100},
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(service_main, "ICEBERG_TRACKS_PATH", archive)
    monkeypatch.setattr(service_main, "ICEBERG_EVIDENCE_PATH", report_path)
    monkeypatch.setattr(service_main, "ICEBERG_EVIDENCE_SHA256",
                        hashlib.sha256(report_path.read_bytes()).hexdigest())
    response = client.get("/api/v1/research/icebergs/A23a/projection",
                          params={"as_of_date": "2025-01-11", "lead_days": 2})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "historical_research_projection"
    assert [item["date"] for item in result["observations"]] == ["2025-01-10", "2025-01-11"]
    assert result["target_date"] == "2025-01-13"
    assert result["projected_position"]["latitude"] > -61
    assert result["evidence"]["archive_sha256"] == digest
    assert "not fixed-time" in result["warning"]

    missing_history = client.get("/api/v1/research/icebergs/A23a/projection",
                                 params={"as_of_date": "2025-01-05"})
    assert missing_history.status_code == 200
    unsupported_date = client.get("/api/v1/research/icebergs/A23a/projection",
                                  params={"as_of_date": "2025-01-06"})
    assert unsupported_date.status_code == 422

    monkeypatch.setattr(service_main, "ICEBERG_EVIDENCE_SHA256", "0" * 64)
    altered_evidence = client.get("/api/v1/research/icebergs/A23a/projection",
                                  params={"as_of_date": "2025-01-11"})
    assert altered_evidence.status_code == 503
