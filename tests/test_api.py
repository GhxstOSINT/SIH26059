import json
import uuid

from fastapi.testclient import TestClient
import numpy as np
from pyproj import Transformer

from service.main import app
from service import main as service_main

client = TestClient(app)


def payload(**overrides):
    request = {
        "previous_day_sic": [[0.2, 0.4], [0.6, 0.8]],
        "two_days_prior_sic": [[0.2, 0.4], [0.6, 0.8]],
        "valid_mask": [[True, True], [True, True]],
        "grid": {"crs": "EPSG:3412", "transform": [-1000000, 25000, 0, 1000000, 0, -25000]},
        "day_before": "2022-12-30",
        "two_days_before": "2022-12-29",
        "target_day_of_year": 365,
        "source_product": "documented-test-product",
        "source_version": "test-v1",
    }
    request.update(overrides)
    return request


def test_health_and_readiness():
    assert client.get("/healthz").status_code == 200
    readiness = client.get("/readyz").json()
    assert readiness["ready"] is True
    assert readiness["mode"] == "research-only"


def test_glo12_archive_status_fails_closed_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(service_main, "GLO12_FORECAST_ROOT", tmp_path / "absent")
    response = client.get("/api/v1/forecast-runs/glo12/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["archive_state"] == "unavailable"
    assert payload["uncertainty_calibrated"] is False
    assert payload["operator_decision_ready"] is False


def test_research_transect_evidence_fails_closed_without_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(service_main, "GLO12_VERIFICATION_ROOT", tmp_path / "absent")
    response = client.get("/api/v1/research/transect-evidence")
    assert response.status_code == 200
    evidence = response.json()
    assert evidence["verified_case_count"] == 0
    assert evidence["uncertainty_calibrated"] is False
    assert evidence["navigation_clearance"] is False
    assert all(lead["route_mean_sic_interval"] is None for lead in evidence["leads"])


def test_initial_model_selection_is_verified_and_research_only():
    response = client.get("/api/v1/model/selection")
    assert response.status_code == 200
    selection = response.json()
    assert selection["verified"] is True
    assert selection["model_fingerprint"] == service_main.load_model()["model_fingerprint"]
    assert selection["candidate_decision"] == "retain_current_model"
    assert selection["forecast_horizon_days"] == 1
    assert selection["operational_decision_ready"] is False


def test_multiplatform_corridor_evaluation_is_frozen_and_inadequate(tmp_path, monkeypatch):
    response = client.get("/api/v1/evaluations/multiplatform-corridors")
    assert response.status_code == 200
    report = response.json()
    assert report["quality_gate"] == "insufficient_evidence"
    assert report["operational_decision_ready"] is False
    assert {platform["platform"] for platform in report["platforms"]} == {"S-NPP", "NOAA-21", "NOAA-20"}
    assert all(route["coverage"]["calendar_coverage_fraction"] < 0.5
               for platform in report["platforms"] for route in platform["routes"])
    altered = tmp_path / "altered-report.json"
    altered.write_bytes(service_main.MULTIPLATFORM_CORRIDOR_PATH.read_bytes() + b" ")
    monkeypatch.setattr(service_main, "MULTIPLATFORM_CORRIDOR_PATH", altered)
    assert client.get("/api/v1/evaluations/multiplatform-corridors").status_code == 503


def test_model_selection_fails_closed_on_runtime_swap(tmp_path, monkeypatch):
    other = tmp_path / "other-model.json"
    other.write_bytes(service_main.MODEL_PATH.read_bytes())
    monkeypatch.setattr(service_main, "MODEL_PATH", other)
    assert service_main.load_model() is None
    readiness = client.get("/readyz")
    assert readiness.status_code == 503
    assert readiness.json()["ready"] is False
    assert client.get("/api/v1/model/selection").status_code == 503


def test_ice_motion_evidence_is_frozen_research_only(tmp_path, monkeypatch):
    response = client.get("/api/v1/research/ice-motion")
    assert response.status_code == 200
    report = response.json()
    assert report["available"] is True
    assert report["decision_status"] == "research_only"
    assert report["fit_year"] == 2023
    assert report["evaluation_year"] == 2024
    assert report["evaluated_days"] == 354
    assert report["zero_coverage_days"] == 10
    assert 3.1 < report["mae"]["reduction_percent"] < 3.2
    assert "not a live forecast" in report["warning"]
    altered = tmp_path / "motion-report.json"
    altered.write_text('{"holdout_metrics": {"mae_reduction_percent": 99}}', encoding="utf-8")
    monkeypatch.setattr(service_main, "MOTION_EVIDENCE_PATH", altered)
    assert client.get("/api/v1/research/ice-motion").json()["available"] is False


def test_api_rejects_oversized_request_body_before_json_parsing():
    response = client.post(
        "/api/v1/route/exposure",
        content=b" " * (16 * 1024 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert "16 MiB" in response.json()["detail"]


def test_model_metadata_includes_evaluation_caveat():
    model = client.get("/api/v1/model").json()
    assert model["available"] is True
    assert len(model["model_fingerprint"]) == 64
    assert len(model["artifact_sha256"]) == 64
    assert model["reproducibility"]["source_manifest_sha256"]
    assert model["training_configuration"]["chronological_split_cutoff_date"] == "2011-05-27"
    assert model["validation"]["rmse"] < model["validation"]["persistence_rmse"]
    assert "NOAA/NSIDC G02202 v6" in model["validation_scope"]
    assert model["external_evaluation"]["dataset_id"] == "noaacwVIIRSnppiceconcSP06Daily"
    assert model["external_evaluation"]["model_fingerprint"] == model["model_fingerprint"]
    assert model["external_evaluation"]["dataset_complete"] is True
    assert model["external_evaluation"]["period"] == {"first_target_date": "2024-06-06", "last_target_date": "2024-08-31"}
    assert model["external_evaluation"]["spatial_evaluation"]["method"] == "native grid"
    assert model["external_evaluation"]["metrics"]["rmse"] < model["external_evaluation"]["metrics"]["persistence_rmse"]
    assert model["same_product_holdout"]["dataset_id"] == "G02202"
    assert model["same_product_holdout"]["period"] == {"first_target_date": "2017-01-03", "last_target_date": "2024-12-31"}
    assert "not area-weighted" in model["same_product_holdout"]["warning"]
    assert model["same_product_holdout"]["area_weighted"]["overall"]["rmse"] < model["same_product_holdout"]["area_weighted"]["overall"]["persistence_rmse"]
    assert model["same_product_holdout"]["area_weighted"]["by_latitude_band"]["north_of_60S"]["rmse"] > model["same_product_holdout"]["area_weighted"]["by_latitude_band"]["north_of_60S"]["persistence_rmse"]
    same_era_edge = model["same_product_holdout"]["area_weighted"]["ice_edge"]["overall"]
    assert same_era_edge["threshold_sic_fraction"] == 0.15
    assert same_era_edge["mean_daily_iiee_million_km2"] > same_era_edge["persistence_mean_daily_iiee_million_km2"]
    sensor_era = model["sensor_era_transfer"]
    assert sensor_era["dataset_id"] == "G02202"
    assert sensor_era["dataset_complete"] is True
    assert sensor_era["period"] == {"first_target_date": "2025-01-03", "last_target_date": "2025-12-31"}
    assert "AMSR2 input beginning 2025-01-01" in sensor_era["warning"]
    assert "Do not pool" in sensor_era["warning"]
    assert sensor_era["area_weighted"]["overall"]["rmse"] < sensor_era["area_weighted"]["overall"]["persistence_rmse"]
    sensor_edge = sensor_era["area_weighted"]["ice_edge"]["overall"]
    assert sensor_edge["valid_daily_fields"] == 363
    assert abs(sensor_edge["mean_daily_iiee_million_km2"] - sensor_edge["persistence_mean_daily_iiee_million_km2"]) < 0.00001
    rolling = model["rolling_origin_benchmark"]
    assert rolling["available"] is True
    assert rolling["dataset_id"] == "G02202"
    assert rolling["fold_count"] == 10
    assert rolling["validation_period"] == {"first_target_date": "1997-01-01", "last_target_date": "2016-12-31"}
    assert rolling["area_weighted"]["overall"]["rmse"] < rolling["area_weighted"]["overall"]["persistence_rmse"]
    assert rolling["monthly_climatology"]["grid_cell_samples"] > 0
    assert rolling["monthly_climatology"]["rmse"] > rolling["cell_pooled"]["rmse"]
    assert rolling["area_weighted"]["monthly_climatology"]["evaluated_area_km2_samples"] > 0
    assert rolling["area_weighted"]["monthly_climatology"]["rmse"] > rolling["area_weighted"]["overall"]["rmse"]
    assert "not scores for the currently packaged model" in rolling["model_artifact_binding"]
    assert "Research baseline only" in model["warning"]
    for key, platform, expected_samples in [
        ("snpp_2026_transfer", "S-NPP", 20740),
        ("noaa21_2026_transfer", "NOAA-21", 7934),
        ("noaa20_2026_transfer", "NOAA-20", 13388),
    ]:
        transfer = model[key]
        assert transfer["available"] is True
        assert transfer["model_fingerprint"] == model["model_fingerprint"]
        assert transfer["dataset_complete"] is True
        assert transfer["period"] == {"first_target_date": "2026-09-03", "last_target_date": "2026-09-21"}
        assert transfer["metrics"]["grid_cell_samples"] == expected_samples
        assert transfer["sample_scope"]["target_days"] == 19
        assert transfer["sample_scope"]["route_scale_evidence"] is False
        assert "correlated" in transfer["sample_scope"]["caveat"]
        assert transfer["platform"] == platform


def test_model_artifact_fingerprint_is_checked(tmp_path, monkeypatch):
    artifact = json.loads(service_main.MODEL_PATH.read_text(encoding="utf-8"))
    artifact["coefficients"][0] += 0.01
    path = tmp_path / "tampered-model.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(service_main, "MODEL_PATH", path)
    assert service_main.load_model() is None


def test_model_artifact_digest_binds_provenance_metadata(tmp_path, monkeypatch):
    artifact = json.loads(service_main.MODEL_PATH.read_text(encoding="utf-8"))
    coefficient_identity = artifact["model_fingerprint"]
    artifact["validation"]["rmse"] += 0.01
    assert artifact["model_fingerprint"] == coefficient_identity
    path = tmp_path / "tampered-evidence-model.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(service_main, "MODEL_PATH", path)
    assert service_main.load_model() is None


def test_malformed_evaluation_is_unavailable_instead_of_breaking_model_metadata(tmp_path):
    model = service_main.load_model()
    assert model is not None
    malformed = tmp_path / "malformed-evaluation.json"
    malformed.write_text(json.dumps({
        "model_id": model["model_id"],
        "model_fingerprint": model["model_fingerprint"],
        "source_manifest": {"complete": True, "dataset_id": "test"},
        "period": {"first_target_date": "not-a-date", "last_target_date": "2026-09-14"},
        "overall": {"grid_cell_samples": 5},
    }), encoding="utf-8")
    assert service_main.load_evaluation(malformed, model) is None
    malformed.write_text(json.dumps({
        "model_id": model["model_id"],
        "model_fingerprint": model["model_fingerprint"],
        "source_manifest": {"complete": True, "dataset_id": "test"},
        "period": {"first_target_date": "2026-09-14", "last_target_date": "2026-09-13"},
        "overall": {"grid_cell_samples": 5},
    }), encoding="utf-8")
    assert service_main.load_evaluation(malformed, model) is None


def test_api_token_is_enforced_when_configured(monkeypatch):
    monkeypatch.setattr(service_main, "API_AUTH_TOKEN", "test-secret")
    assert client.get("/api/v1/model").status_code == 401
    assert client.get("/api/v1/model", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/api/v1/model", headers={"Authorization": "Bearer test-secret"})
    assert response.status_code == 200
    assert len(response.headers["x-request-id"]) == 32


def test_api_request_id_and_structured_audit_event_are_body_and_secret_free(monkeypatch):
    events = []
    monkeypatch.setattr(service_main, "API_AUTH_TOKEN", "test-secret")
    monkeypatch.setattr(service_main.AUDIT_LOGGER, "info", events.append)
    response = client.get("/api/v1/model", headers={
        "Authorization": "Bearer test-secret", "X-Request-ID": "pilot-42",
    })

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "pilot-42"
    assert len(events) == 1
    event = json.loads(events[0])
    assert event == {"event": "api_request", "request_id": "pilot-42",
                     "method": "GET", "path": "/api/v1/model",
                     "status_code": 200, "duration_ms": event["duration_ms"]}
    assert "test-secret" not in events[0]
    assert "Authorization" not in events[0]


def test_invalid_request_id_is_replaced_before_logging(monkeypatch):
    events = []
    monkeypatch.setattr(service_main, "API_AUTH_TOKEN", "test-secret")
    monkeypatch.setattr(service_main.AUDIT_LOGGER, "info", events.append)
    response = client.get("/api/v1/model", headers={
        "Authorization": "Bearer test-secret", "X-Request-ID": "not valid",
    })

    assert response.status_code == 200
    assert len(response.headers["x-request-id"]) == 32
    assert json.loads(events[0])["request_id"] == response.headers["x-request-id"]


def test_authenticated_cross_origin_cors_preflight_allows_approved_cookie_client(monkeypatch):
    monkeypatch.setattr(service_main, "API_AUTH_TOKEN", "test-secret")
    response = client.options("/api/v1/model", headers={
        "Origin": "http://localhost:4173",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:4173"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_production_configuration_fails_closed():
    validate = service_main.validate_runtime_configuration
    import pytest

    with pytest.raises(RuntimeError, match="at least 32 characters"):
        validate("production", "short", ["https://ministry.example"])
    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        validate("production", "x" * 40, ["*"])
    with pytest.raises(RuntimeError, match="HTTPS"):
        validate("production", "x" * 40, ["http://ministry.example"])
    with pytest.raises(RuntimeError, match="APP_ENV"):
        validate("prod", "x" * 40, ["https://ministry.example"])
    validate("production", "x" * 40, ["https://ministry.example"])


def test_grid_schema_rejects_unrecognized_crs_singular_transform_and_non_boolean_mask():
    import pytest
    from pydantic import ValidationError

    base = payload()
    base["grid"] = {"crs": "this-is-not-a-crs", "transform": [-1, 1, 0, 1, 0, -1]}
    with pytest.raises(ValidationError):
        service_main.GridForecastRequest.model_validate(base)
    base = payload(grid={"crs": "EPSG:3412", "transform": [-1, 1, 2, 1, 2, 4]})
    with pytest.raises(ValidationError):
        service_main.GridForecastRequest.model_validate(base)
    base = payload(valid_mask=[["yes", True], [False, True]])
    with pytest.raises(ValidationError):
        service_main.GridForecastRequest.model_validate(base)


def test_grid_schema_bounds_array_dimensions_before_forecast():
    import pytest
    from pydantic import ValidationError

    too_many_rows = [[0.1] for _ in range(501)]
    with pytest.raises(ValidationError):
        service_main.GridForecastRequest.model_validate(payload(
            previous_day_sic=too_many_rows,
            two_days_prior_sic=too_many_rows,
            valid_mask=[[True] for _ in range(501)],
        ))


def test_evaluation_is_not_attached_to_a_different_coefficient_set():
    artifact = service_main.load_model()
    assert artifact is not None
    changed = dict(artifact)
    changed["coefficients"] = list(artifact["coefficients"])
    changed["coefficients"][0] += 0.01
    assert service_main.load_external_evaluation(changed) is None


def test_observation_endpoints_serve_dated_georeferenced_source_data(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    import xarray as xr

    data_dir = tmp_path / "viirs"
    data_dir.mkdir()
    values = np.array([[[[0.2, np.nan, 0.8], [0.1, 0.4, 0.9]]]], dtype=np.float32)
    xr.Dataset({"IceConc": (("time", "altitude", "rows", "cols"), values)},
               coords={"time": [np.datetime64("2024-06-01T12:00:00")], "altitude": [0.0],
                       "rows": [1000.0, 0.0], "cols": [0.0, 1000.0, 2000.0]}).to_netcdf(data_dir / "sample.nc")
    source_digest = hashlib.sha256((data_dir / "sample.nc").read_bytes()).hexdigest()
    (data_dir / "manifest.json").write_text(json.dumps({
        "dataset_id": "test-viirs", "dataset_title": "Test SIC product", "complete": True,
        "accessed_at_utc": "2024-06-02T00:00:00Z", "selection": {"crs": "EPSG:3976"},
        "files": [{"file": "sample.nc", "sha256": source_digest}],
    }), encoding="utf-8")
    monkeypatch.setattr(service_main, "OBSERVATIONS_PATH", data_dir)
    monkeypatch.setattr(service_main, "SUPPLEMENTARY_OBSERVATIONS_PATHS", [])

    catalog = client.get("/api/v1/observations/sic").json()
    assert catalog["observation_count"] == 1
    assert catalog["download_complete"] is True
    assert catalog["freshness"]["status"] in {"recent", "delayed", "stale", "future_timestamp"}
    assert catalog["freshness"]["newest_timestamp"] == "2024-06-01T12:00:00Z"
    assert catalog["freshness"]["classification_basis"] == "demonstration_thresholds_not_approved_operational_limits"
    assert "not verify source timeliness" in catalog["freshness"]["note"]
    response = client.get("/api/v1/observations/sic/2024-06-01")
    assert response.status_code == 200
    observation = response.json()
    assert observation["grid"]["crs"] == "EPSG:3976"
    assert observation["grid"]["orientation"] == "west-to-east columns; north-to-south rows"
    assert observation["sic_fraction"][0][1] is None
    np.testing.assert_allclose([observation["sic_fraction"][0][0], observation["sic_fraction"][0][2]], [0.2, 0.8])
    assert observation["source_provenance"]["source_sha256"] == source_digest
    assert client.get("/api/v1/observations/sic/2024-06-02").status_code == 404
    with (data_dir / "sample.nc").open("ab") as altered:
        altered.write(b"tampered")
    assert client.get("/api/v1/observations/sic").status_code == 503


def test_supplementary_observation_catalog_merges_periods_and_preserves_per_file_provenance(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    import xarray as xr

    directories = [tmp_path / "primary", tmp_path / "supplement"]
    for directory, day, concentration, label in zip(
            directories, ["2024-06-01T12:00:00", "2026-09-01T11:34:50"], [0.25, 0.75], ["primary", "supplement"]):
        directory.mkdir()
        source = directory / f"{label}.nc"
        values = np.full((1, 1, 2, 2), concentration, dtype=np.float32)
        xr.Dataset({"IceConc": (("time", "altitude", "rows", "cols"), values)},
                   coords={"time": [np.datetime64(day)], "altitude": [0.0],
                           "rows": [1000.0, 0.0], "cols": [0.0, 1000.0]}).to_netcdf(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        (directory / "manifest.json").write_text(json.dumps({
            "dataset_id": "noaacwVIIRSn21iceconcSP06Daily" if label == "supplement" else label,
            "dataset_title": f"Test {label}", "complete": True,
            "accessed_at_utc": "2026-09-29T00:00:00Z", "selection": {"crs": "EPSG:3976"},
            "files": [{"file": source.name, "sha256": digest}],
        }), encoding="utf-8")

    monkeypatch.setattr(service_main, "OBSERVATIONS_PATH", directories[0])
    monkeypatch.setattr(service_main, "SUPPLEMENTARY_OBSERVATIONS_PATHS", [directories[1]])
    catalog = client.get("/api/v1/observations/sic").json()
    assert catalog["observation_count"] == 2
    assert [item["dataset_id"] for item in catalog["datasets"]] == ["primary", "noaacwVIIRSn21iceconcSP06Daily"]
    assert [item["date"] for item in catalog["observations"]] == ["2024-06-01", "2026-09-01"]
    observation = client.get("/api/v1/observations/sic/2026-09-01").json()
    assert observation["source_provenance"]["dataset_id"] == "noaacwVIIRSn21iceconcSP06Daily"
    assert observation["source_provenance"]["platform"] == "NOAA-21"
    assert observation["sic_fraction"][0][0] == 0.75


def test_observation_freshness_status_is_explicit_and_not_a_quality_claim():
    from datetime import datetime, timezone

    record = (None, None, "2026-09-25T00:00:00Z", None, None)
    now = datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc)
    summary = service_main.summarize_observation_freshness({"2026-09-25": record}, now)
    assert summary["status"] == "delayed"
    assert summary["age_hours"] == 96.0
    assert summary["thresholds_hours"] == {"recent_max": 48, "delayed_max": 168}
    assert "does not verify" in summary["note"]

    future = (None, None, "2026-09-30T00:00:00Z", None, None)
    assert service_main.summarize_observation_freshness({"2026-09-30": future}, now)["status"] == "future_timestamp"


def test_upstream_source_monitor_is_distinct_from_local_observation_status(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    report_path = tmp_path / "source-status.json"
    report_path.write_text(json.dumps({
        "schema_version": 1,
        "checked_at": now.isoformat().replace("+00:00", "Z"),
        "status": "stale",
        "classification_basis": "demonstration_thresholds_not_approved_operational_limits",
        "thresholds_hours": {"recent_max": 48, "delayed_max": 168},
        "products": [{"dataset_id": "noaacwVIIRSn20iceconcSP06Daily", "platform": "NOAA-20",
                      "status": "stale", "latest_source_timestamp": "2026-09-21T12:20:18Z",
                      "age_hours": 200}],
        "warning": "Upstream metadata only.",
    }), encoding="utf-8")
    monkeypatch.setattr(service_main, "VIIRS_SOURCE_STATUS_PATH", report_path)
    result = client.get("/api/v1/observations/sic/source-status")
    assert result.status_code == 200
    body = result.json()
    assert body["available"] is True
    assert body["monitor_status"] == "current_check"
    assert body["source_status"] == "stale"
    assert body["products"][0]["latest_source_timestamp"] == "2026-09-21T12:20:18Z"

    monkeypatch.setattr(service_main, "VIIRS_SOURCE_STATUS_PATH", tmp_path / "missing.json")
    unavailable = client.get("/api/v1/observations/sic/source-status").json()
    assert unavailable["available"] is False
    assert unavailable["monitor_status"] == "unconfigured"


def test_route_exposure_samples_scene_and_reports_partial_coverage_without_safety_claim(monkeypatch):
    from pyproj import CRS, Transformer

    center_x, center_y, spacing = -2_283_720.0, 1_318_506.0, 1000.0
    transformer = Transformer.from_crs(CRS.from_epsg(3976), CRS.from_epsg(4326), always_xy=True)
    first = transformer.transform(center_x - 900.0, center_y)
    second = transformer.transform(center_x + 900.0, center_y)
    scene = {
        "date": "2024-06-01", "timestamp": "2024-06-01T12:00:00Z",
        "grid": {"crs": "EPSG:3976", "shape": [3, 3],
                 "x_first_center_m": center_x - spacing, "x_last_center_m": center_x + spacing,
                 "y_first_center_m": center_y + spacing, "y_last_center_m": center_y - spacing,
                 "pixel_spacing_x_m": spacing, "pixel_spacing_y_m": spacing,
                 "orientation": "west-to-east columns; north-to-south rows"},
        "sic_fraction": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "source_provenance": {"provider": "test", "source_sha256": "a" * 64},
    }
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    response = client.post("/api/v1/route/exposure", json={
        "route_id": "peninsula-test",
        "observation_date": "2024-06-01",
        "analysis_half_width_km": 0,
        "geometry": {"type": "LineString", "coordinates": [first, second]},
    })
    assert response.status_code == 200
    report = response.json()
    assert report["analysis_type"] == "observed_sic_route_centerline_screen"
    assert report["navigation_suitability"] == "not_assessed"
    assert report["data_coverage_status"] == "complete"
    assert report["sampling"]["corridor_width_assessed"] is False
    assert len(report["centerline_pixel_coordinates"]) == report["sampling"]["total_sample_count"]
    assert all(len(point) == 2 for point in report["centerline_pixel_coordinates"])
    assert 0.4 <= report["sic_summary_fraction"]["mean"] <= 0.6
    assert all(item["data_coverage_fraction"] == 1 for item in report["profile_5km"])
    assert report["source_provenance"]["source_sha256"] == "a" * 64
    assert "not a vessel-specific navigation corridor or forecast" in report["warning"]


def test_route_exposure_reports_context_band_without_safety_assessment(monkeypatch):
    from pyproj import CRS, Transformer

    center_x, center_y, spacing = -2_283_720.0, 1_318_506.0, 1000.0
    transformer = Transformer.from_crs(CRS.from_epsg(3976), CRS.from_epsg(4326), always_xy=True)
    first = transformer.transform(center_x - 900.0, center_y)
    second = transformer.transform(center_x + 900.0, center_y)
    scene = {
        "date": "2024-06-01", "timestamp": "2024-06-01T12:00:00Z",
        "grid": {"crs": "EPSG:3976", "shape": [3, 3],
                 "x_first_center_m": center_x - spacing, "x_last_center_m": center_x + spacing,
                 "y_first_center_m": center_y + spacing, "y_last_center_m": center_y - spacing,
                 "pixel_spacing_x_m": spacing, "pixel_spacing_y_m": spacing,
                 "orientation": "west-to-east columns; north-to-south rows"},
        "sic_fraction": [[0.8, 0.8, 0.8], [0.1, 0.1, 0.1], [0.7, 0.7, 0.7]],
        "source_provenance": {"provider": "test", "source_sha256": "c" * 64},
    }
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    response = client.post("/api/v1/route/exposure", json={
        "route_id": "context-band-test", "observation_date": "2024-06-01",
        "analysis_half_width_km": 1.0,
        "geometry": {"type": "LineString", "coordinates": [first, second]},
    })
    assert response.status_code == 200
    report = response.json()
    assert report["analysis_type"] == "observed_sic_route_centerline_and_context_band_screen"
    assert report["sic_summary_fraction"]["mean"] < report["context_band_sic_summary_fraction"]["mean"]
    assert report["sampling"]["corridor_width_assessed"] is True
    assert report["sampling"]["context_band_total_width_km"] == 2.0
    assert report["sampling"]["context_band_valid_sample_fraction"] == 1.0
    assert report["sampling"]["context_band_navigation_corridor"] is False
    assert len(report["context_band_pixel_polygon"]) > 4
    assert all(item["context_band_maximum_sic_fraction"] > report["sic_summary_fraction"]["mean"]
               for item in report["profile_5km"])
    assert report["navigation_suitability"] == "not_assessed"


def test_route_context_band_width_is_bounded():
    base = {"route_id": "bounded", "observation_date": "2024-06-01",
            "geometry": {"type": "LineString", "coordinates": [[0, -60], [0.01, -60]]}}
    assert client.post("/api/v1/route/exposure", json={**base, "analysis_half_width_km": 20.1}).status_code == 422
    assert client.post("/api/v1/route/exposure", json={**base, "analysis_half_width_km": -0.1}).status_code == 422


def test_route_exposure_handles_out_of_scene_routes_and_invalid_geojson(monkeypatch):
    scene = {"date": "2024-06-01", "timestamp": "2024-06-01T12:00:00Z",
             "grid": {"crs": "EPSG:3976", "shape": [1, 1], "x_first_center_m": -2_283_720.0,
                      "x_last_center_m": -2_283_720.0, "y_first_center_m": 1_318_506.0,
                      "y_last_center_m": 1_318_506.0, "pixel_spacing_x_m": 1000.0,
                      "pixel_spacing_y_m": 1000.0},
             "sic_fraction": [[0.5]], "source_provenance": {"source_sha256": "b" * 64}}
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    response = client.post("/api/v1/route/exposure", json={
        "route_id": "outside", "observation_date": "2024-06-01",
        "geometry": {"type": "LineString", "coordinates": [[-30.0, -60.0], [-29.99, -60.0]]},
    })
    assert response.status_code == 200
    assert response.json()["data_coverage_status"] == "no_valid_data"
    assert response.json()["sic_summary_fraction"] is None
    assert response.json()["profile_5km"][0]["data_coverage_fraction"] == 0
    invalid = client.post("/api/v1/route/exposure", json={
        "route_id": "invalid", "observation_date": "2024-06-01",
        "geometry": {"type": "LineString", "coordinates": [[190.0, -60.0], [191.0, -60.0]]},
    })
    assert invalid.status_code == 422


def test_route_exposure_rejects_routes_over_analysis_limit_before_sampling(monkeypatch):
    scene = {"date": "2024-06-01", "timestamp": "2024-06-01T12:00:00Z",
             "grid": {"crs": "EPSG:3976", "shape": [1, 1], "x_first_center_m": -2_283_720.0,
                      "x_last_center_m": -2_283_720.0, "y_first_center_m": 1_318_506.0,
                      "y_last_center_m": 1_318_506.0, "pixel_spacing_x_m": 1000.0,
                      "pixel_spacing_y_m": 1000.0},
             "sic_fraction": [[0.5]], "source_provenance": {"source_sha256": "b" * 64}}
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    response = client.post("/api/v1/route/exposure", json={
        "route_id": "too-long", "observation_date": "2024-06-01",
        "geometry": {"type": "LineString", "coordinates": [[-70.0, -60.0], [-60.0, -60.0], [-50.0, -60.0]]},
    })
    assert response.status_code == 422
    assert "1,000 km" in response.json()["detail"]


def _synthetic_route_scene(blocked_barrier=False):
    spacing = 1000.0
    to_grid = Transformer.from_crs("EPSG:4326", "EPSG:3976", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:3976", "EPSG:4326", always_xy=True)
    center_x, center_y = to_grid.transform(-60.0, -66.0)
    grid = {"crs": "EPSG:3976", "shape": [9, 9],
            "x_first_center_m": center_x, "x_last_center_m": center_x + 8 * spacing,
            "y_first_center_m": center_y + 4 * spacing, "y_last_center_m": center_y - 4 * spacing,
            "pixel_spacing_x_m": spacing, "pixel_spacing_y_m": spacing,
            "orientation": "west-to-east columns; north-to-south rows"}
    field = np.full((9, 9), 0.1, dtype=np.float32)
    field[1:9, 4] = 0.9
    if blocked_barrier:
        field[:, 4] = np.nan
    def point(row, col):
        return to_wgs84.transform(center_x + col * spacing,
                                  center_y + 4 * spacing - row * spacing)
    origin = point(4, 1)
    destination = point(4, 7)
    scene = {"date": "2024-06-04", "timestamp": "2024-06-04T20:47:35Z",
             "grid": grid, "sic_fraction": field.tolist(),
             "source_provenance": {"provider": "test", "source_sha256": "a" * 64}}
    return scene, origin, destination


def test_route_candidates_return_observation_constrained_objectives(monkeypatch):
    scene, origin, destination = _synthetic_route_scene()
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    response = client.post("/api/v1/route/candidates", json={
        "route_id": "observed-grid-test", "observation_date": "2024-06-04",
        "origin": {"type": "Point", "coordinates": origin},
        "destination": {"type": "Point", "coordinates": destination},
    })
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["analysis_type"] == "observed_sic_grid_route_candidates"
    assert report["navigation_suitability"] == "not_assessed"
    assert report["fuel_or_transit_time_estimates"] is None
    assert report["research_time_assumptions"]["status"] == "research_only_not_eta_or_navigation_advice"
    assert [candidate["candidate_id"] for candidate in report["candidates"]] == [
        "shortest_grid_distance", "distance_ice_tradeoff", "lower_observed_ice"]
    shortest, _, lower_ice = report["candidates"]
    assert shortest["metrics"]["distance_weighted_mean_sic_fraction"] > lower_ice["metrics"]["distance_weighted_mean_sic_fraction"]
    assert shortest["metrics"]["grid_distance_km"] < lower_ice["metrics"]["grid_distance_km"]
    assert shortest["research_transit_sensitivity"]["status"] == "unavailable"
    assert lower_ice["research_transit_sensitivity"]["status"] == "hypothetical"
    assert lower_ice["research_transit_sensitivity"]["fast_hours"] < lower_ice["research_transit_sensitivity"]["slow_hours"]
    assert all(candidate["geometry"]["type"] == "LineString" for candidate in report["candidates"])
    assert report["fixed_path_counterfactuals"]["navigation_suitability"] == "not_assessed"
    assert len(report["fixed_path_counterfactuals"]["scenarios"]) == 3
    assert "not calibrated to a vessel" in report["warning"]


def test_route_candidates_research_speeds_are_bounded_and_change_time(monkeypatch):
    scene, origin, destination = _synthetic_route_scene()
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    request = {"route_id": "research", "observation_date": "2024-06-04",
               "origin": {"type": "Point", "coordinates": origin},
               "destination": {"type": "Point", "coordinates": destination},
               "research_open_water_knots": 10, "research_ice_affected_knots": 3}
    baseline = client.post("/api/v1/route/candidates", json=request)
    assert baseline.status_code == 200
    faster = client.post("/api/v1/route/candidates", json={**request,
        "research_open_water_knots": 15, "research_ice_affected_knots": 5})
    assert faster.status_code == 200
    assert (faster.json()["candidates"][2]["research_transit_sensitivity"]["central_hours"]
            < baseline.json()["candidates"][2]["research_transit_sensitivity"]["central_hours"])
    invalid = client.post("/api/v1/route/candidates", json={**request,
        "research_open_water_knots": 2, "research_ice_affected_knots": 3})
    assert invalid.status_code == 422


def test_route_candidates_fail_closed_for_missing_data_barrier_and_unobserved_endpoint(monkeypatch):
    scene, origin, destination = _synthetic_route_scene(blocked_barrier=True)
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    request = {"route_id": "blocked", "observation_date": "2024-06-04",
               "origin": {"type": "Point", "coordinates": origin},
               "destination": {"type": "Point", "coordinates": destination}}
    blocked = client.post("/api/v1/route/candidates", json=request)
    assert blocked.status_code == 422
    assert "No continuous valid-data path" in blocked.json()["detail"]
    scene["sic_fraction"][4][1] = None
    endpoint_missing = client.post("/api/v1/route/candidates", json=request)
    assert endpoint_missing.status_code == 422
    assert "origin maps to a cell without valid observed SIC" in endpoint_missing.json()["detail"]


def test_forecast_returns_bounded_fractions_and_provenance():
    response = client.post("/api/v1/forecast/sic", json=payload())
    assert response.status_code == 200
    result = response.json()
    assert str(uuid.UUID(result["run_id"])) == result["run_id"]
    assert result["grid_shape"] == [2, 2]
    assert result["grid"]["crs"] == "EPSG:3412"
    assert result["model_fingerprint"] == service_main.load_model()["model_fingerprint"]
    assert result["artifact_sha256"] == service_main.load_model()["artifact_sha256"]
    assert result["valid_cell_count"] == 4
    assert result["source_provenance"]["product"] == "documented-test-product"
    assert all(0 <= value <= 1 for row in result["sic_fraction"] for value in row)
    assert "caller-supplied" in result["source_provenance"]["note"]
    assert result["decision_status"] == "research_only"
    evidence = result["model_evidence"]
    assert evidence["artifact_sha256"] == result["artifact_sha256"]
    assert evidence["training_data"]["dataset_id"] == "G02202"
    assert evidence["training_data"]["doi"] == "10.7265/b18j-z797"
    assert evidence["training_data"]["manifest_complete"] is True
    assert evidence["holdout_scope"] == service_main.load_model()["validation_scope"]
    assert evidence["holdout"]["rmse_sic_fraction"] == service_main.load_model()["validation"]["rmse"]
    assert evidence["holdout"]["persistence_rmse_sic_fraction"] == service_main.load_model()["validation"]["persistence_rmse"]
    assert evidence["holdout"]["relative_rmse_reduction_vs_persistence"] < 0.01
    assert any("No calibrated uncertainty" in item for item in evidence["limitations"])
    assert evidence["regional_cross_product_transfer_check"]["dataset_id"] == "noaacwVIIRSnppiceconcSP06Daily"
    assert evidence["regional_cross_product_transfer_check"]["rmse_sic_fraction"] < evidence["regional_cross_product_transfer_check"]["persistence_rmse_sic_fraction"]


def test_forecast_rejects_wrong_target_date_and_bad_sic():
    assert client.post("/api/v1/forecast/sic", json=payload(target_day_of_year=364)).status_code == 422
    bad_grid = [[0.2, 0.4], [0.6, 1.1]]
    assert client.post("/api/v1/forecast/sic", json=payload(previous_day_sic=bad_grid)).status_code == 422


def test_forecast_rejects_nonconsecutive_input_dates():
    request = payload(two_days_before="2022-12-28", target_day_of_year=365)
    assert client.post("/api/v1/forecast/sic", json=request).status_code == 422


def test_forecast_masks_invalid_cells_instead_of_reporting_predictions():
    request = payload(valid_mask=[[True, False], [False, True]])
    response = client.post("/api/v1/forecast/sic", json=request)
    assert response.status_code == 200
    assert response.json()["sic_fraction"][0][1] is None
    assert response.json()["sic_fraction"][1][0] is None
    assert response.json()["valid_cell_count"] == 2


def test_forecast_rejects_invalid_grid_reference():
    request = payload(grid={"crs": "EPSG:3412", "transform": [0, 0, 0, 0, 0, 0]})
    assert client.post("/api/v1/forecast/sic", json=request).status_code == 422


def _observed_scene(day, values, *, dataset_id="test-viirs", platform="S-NPP", x_shift=0.0):
    return {
        "date": day,
        "timestamp": f"{day}T00:00:00Z",
        "sic_fraction": values,
        "grid": {"shape": [2, 2], "crs": "EPSG:3976",
                 "x_first_center_m": 100.0 + x_shift, "x_last_center_m": 110.0 + x_shift,
                 "y_first_center_m": 210.0, "y_last_center_m": 200.0,
                 "pixel_spacing_x_m": 10.0, "pixel_spacing_y_m": 10.0},
        "source_provenance": {"dataset_id": dataset_id, "platform": platform,
                              "product": "Test VIIRS daily SIC", "source_file": f"{day}.nc",
                              "source_sha256": (day[0] * 64)},
    }


def test_observed_one_day_forecast_binds_two_consecutive_files_to_result(monkeypatch):
    scenes = {
        "2024-08-30": _observed_scene("2024-08-30", [[0.3, 0.2], [0.4, 0.1]]),
        "2024-08-31": _observed_scene("2024-08-31", [[0.25, None], [0.35, 0.12]]),
    }
    monkeypatch.setattr(service_main, "get_sic_observation", lambda day: scenes[day.isoformat()])
    response = client.get("/api/v1/observations/sic/2024-08-31/forecast")
    assert response.status_code == 200
    report = response.json()
    assert report["initialization_date"] == "2024-08-31"
    assert report["target_date"] == "2024-09-01"
    assert report["valid_cell_count"] == 3
    assert report["input_coverage"]["common_coverage_fraction"] == 0.75
    assert report["input_coverage"]["threshold_status"] == "demo_only_not_operationally_approved"
    assert report["sic_fraction"][0][1] is None
    assert report["decision_status"] == "research_only"
    assert [item["date"] for item in report["source_provenance"]["input_observations"]] == ["2024-08-30", "2024-08-31"]
    assert all(len(item["source_sha256"]) == 64 for item in report["source_provenance"]["input_observations"])


def test_observed_one_day_forecast_requires_consecutive_same_grid_same_product(monkeypatch):
    from fastapi import HTTPException

    current = _observed_scene("2024-08-31", [[0.25, 0.2], [0.35, 0.12]])
    def missing_previous(day):
        if day.isoformat() != "2024-08-31":
            raise HTTPException(status_code=404, detail="no scene")
        return current
    monkeypatch.setattr(service_main, "get_sic_observation", missing_previous)
    no_previous = client.get("/api/v1/observations/sic/2024-08-31/forecast")
    assert no_previous.status_code == 422
    assert "consecutive VIIRS observation" in no_previous.json()["detail"]

    previous = _observed_scene("2024-08-30", [[0.3, 0.2], [0.4, 0.1]], dataset_id="other")
    monkeypatch.setattr(service_main, "get_sic_observation",
                        lambda day: current if day.isoformat() == "2024-08-31" else previous)
    different_product = client.get("/api/v1/observations/sic/2024-08-31/forecast")
    assert different_product.status_code == 422

    previous["source_provenance"]["dataset_id"] = "test-viirs"
    previous["grid"]["x_first_center_m"] += 100
    previous["grid"]["x_last_center_m"] += 100
    different_grid = client.get("/api/v1/observations/sic/2024-08-31/forecast")
    assert different_grid.status_code == 422


def test_observed_one_day_forecast_fails_closed_when_common_scene_coverage_is_sparse(monkeypatch):
    from fastapi import HTTPException

    scenes = {
        "2024-08-30": _observed_scene("2024-08-30", [[0.3, 0.2], [0.4, 0.1]]),
        "2024-08-31": _observed_scene("2024-08-31", [[None, None], [None, None]]),
    }
    def lookup(day):
        if day.isoformat() not in scenes:
            raise HTTPException(status_code=404, detail="no scene")
        return scenes[day.isoformat()]
    monkeypatch.setattr(service_main, "get_sic_observation", lookup)
    response = client.get("/api/v1/observations/sic/2024-08-31/forecast")
    assert response.status_code == 422
    assert "Only 0 of 4 grid cells" in response.json()["detail"]
    assert "not an approved operational threshold" in response.json()["detail"]
