import hashlib
import json

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pyproj import Transformer

import service.main as service_main


client = TestClient(service_main.app)


def test_bundled_portal_serves_only_allowlisted_assets():
    redirect = client.get("/portal", follow_redirects=False)
    assert redirect.status_code == 308
    assert redirect.headers["location"] == "/portal/"
    page = client.get("/portal/")
    assert page.status_code == 200
    assert "Southern Passage" in page.text
    assert 'href="marine.css"' in page.text
    assert 'src="hero-iceberg.png"' in page.text
    assert 'src="marine-interactions.js"' in page.text
    assert page.headers["cache-control"] == "no-store"
    assert client.get("/portal/app.js").status_code == 200
    assert client.get("/portal/review-workflow.js").status_code == 200
    assert client.get("/portal/product.css").status_code == 200
    assert client.get("/portal/experience.js").status_code == 200
    assert client.get("/portal/cinematic.css").status_code == 200
    assert client.get("/portal/orbital.css").status_code == 200
    marine_css = client.get("/portal/marine.css")
    assert marine_css.status_code == 200
    assert marine_css.headers["content-type"].startswith("text/css")
    marine_script = client.get("/portal/marine-interactions.js")
    assert marine_script.status_code == 200
    assert marine_script.headers["content-type"].startswith("text/javascript")
    image = client.get("/portal/polar-orbit.png")
    satellite = client.get("/portal/orbital-satellite.png")
    assert satellite.status_code == 200
    assert satellite.headers["content-type"] == "image/png"
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/png")
    for asset_name in ("antarctic-sea.png", "hero-iceberg.png", "indian-research-vessel.png"):
        asset = client.get(f"/portal/{asset_name}")
        assert asset.status_code == 200
        assert asset.headers["content-type"] == "image/png"
    assert client.get("/portal/main.py").status_code == 404
    assert client.get("/portal/%2e%2e%2fmodels%2fsic_g02202_1989_2016.json").status_code == 404


def test_portal_capabilities_and_protected_openapi_contract():
    capabilities = client.get("/api/v1/integration/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["navigation_clearance_available"] is False
    schema = client.get("/api/v1/integration/openapi.json")
    assert schema.status_code == 200
    assert "/api/v1/reviews/route" in schema.json()["paths"]


def test_route_review_packet_contains_source_provenance_and_verifiable_digest(monkeypatch):
    center_x, center_y, spacing = -2_283_720.0, 1_318_506.0, 1000.0
    to_geo = Transformer.from_crs("EPSG:3976", "EPSG:4326", always_xy=True)
    first = to_geo.transform(center_x - 900, center_y)
    second = to_geo.transform(center_x + 900, center_y)
    scene = {
        "date": "2024-06-01", "timestamp": "2024-06-01T12:00:00Z",
        "grid": {"crs": "EPSG:3976", "shape": [3, 3],
                 "x_first_center_m": center_x - spacing, "x_last_center_m": center_x + spacing,
                 "y_first_center_m": center_y + spacing, "y_last_center_m": center_y - spacing,
                 "pixel_spacing_x_m": spacing, "pixel_spacing_y_m": spacing},
        "sic_fraction": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "source_provenance": {"provider": "NOAA test scene", "source_sha256": "d" * 64},
    }
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: scene)
    response = client.post("/api/v1/reviews/route", headers={"X-Request-ID": "case-52"}, json={
        "route_id": "case-route", "observation_date": "2024-06-01",
        "analysis_half_width_km": 0,
        "geometry": {"type": "LineString", "coordinates": [first, second]},
    })
    assert response.status_code == 200
    packet = response.json()
    assert packet["schema_version"] == 1
    assert packet["request_id"] == "case-52"
    assert packet["observation"]["source_provenance"]["source_sha256"] == "d" * 64
    assert packet["review"]["status"] == "awaiting_human_review"
    assert packet["review"]["decision"] is None
    assert packet["review"]["navigation_suitability"] == "not_assessed"
    assert packet["analysis"]["evidence_gate"]["decision_status"] == "abstain"
    assert packet["analysis"]["ice_edge_fragility"]["analysis_type"] == "observed_ice_edge_threshold_sensitivity"
    assert packet["analysis"]["vessel_research_screen"] is None
    assert len(packet["analysis_key"]) == 64
    canonical = json.dumps({key: value for key, value in packet.items() if key != "integrity"},
                           sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert packet["integrity"]["sha256"] == hashlib.sha256(canonical).hexdigest()
    geo = client.post("/api/v1/reviews/route.geojson", json={
        "route_id": "case-route", "observation_date": "2024-06-01",
        "analysis_half_width_km": 0,
        "geometry": {"type": "LineString", "coordinates": [first, second]},
    })
    assert geo.status_code == 200
    assert geo.json()["type"] == "FeatureCollection"
    assert geo.json()["features"][0]["geometry"]["type"] == "LineString"
    assert geo.json()["features"][0]["properties"]["navigation_clearance"] is False
    assert geo.json()["features"][0]["properties"]["decision_status"] == "abstain"
    assert geo.json()["southern_passage_metadata"]["standards_note"].endswith("not an OGC API-EDR conformance claim")
    bundle = client.post("/api/v1/reviews/route/bundle", json={
        "route_id": "case-route", "observation_date": "2024-06-01",
        "analysis_half_width_km": 0,
        "geometry": {"type": "LineString", "coordinates": [first, second]},
    })
    assert bundle.status_code == 200
    pair = bundle.json()
    assert pair["geojson"]["features"][0]["properties"]["packet_sha256"] == pair["packet"]["integrity"]["sha256"]
    assert pair["geojson"]["features"][0]["id"] == pair["packet"]["packet_id"]


def test_entered_vessel_profile_is_screened_but_never_certified(monkeypatch):
    center_x, center_y = -2_283_720.0, 1_318_506.0
    to_geo = Transformer.from_crs("EPSG:3976", "EPSG:4326", always_xy=True)
    first = to_geo.transform(center_x - 900, center_y)
    second = to_geo.transform(center_x + 900, center_y)
    monkeypatch.setattr(service_main, "get_sic_observation", lambda _day: {
        "date": "2024-06-01", "timestamp": "2024-06-01T12:00:00Z",
        "grid": {"crs": "EPSG:3976", "shape": [3, 3],
                 "x_first_center_m": center_x - 1000, "y_first_center_m": center_y + 1000,
                 "pixel_spacing_x_m": 1000, "pixel_spacing_y_m": 1000},
        "sic_fraction": [[0.2] * 3] * 3,
        "source_provenance": {"provider": "test", "source_sha256": "f" * 64},
    })
    payload = {"route_id": "research-vessel", "observation_date": "2024-06-01",
               "analysis_half_width_km": 0,
               "geometry": {"type": "LineString", "coordinates": [first, second]},
               "research_vessel_profile": {"profile_label": "Unverified research profile",
                                           "max_observed_sic_fraction": 0.3,
                                           "minimum_observation_coverage_fraction": 0.9}}
    response = client.post("/api/v1/reviews/route", json=payload)
    assert response.status_code == 200
    screen = response.json()["analysis"]["vessel_research_screen"]
    assert screen["navigation_clearance"] is False
    assert screen["profile_verification"] == "user_supplied_unverified"
    assert response.json()["route"]["research_vessel_profile"]["profile_label"] == "Unverified research profile"


def test_integration_status_reports_missing_components_without_false_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(service_main, "load_model", lambda: None)
    monkeypatch.setattr(service_main, "viirs_observation_index",
                        lambda: (_ for _ in ()).throw(HTTPException(503, "No mounted observations.")))
    monkeypatch.setattr(service_main, "load_viirs_source_status",
                        lambda: {"available": False, "monitor_status": "unconfigured"})
    monkeypatch.setattr(service_main, "ice_motion_evidence", lambda: {"available": False})
    monkeypatch.setattr(service_main, "active_viirs_status",
                        lambda: {"available_for_research": False, "status": "unavailable"})
    monkeypatch.setattr(service_main, "corridor_evaluation_status",
                        lambda: {"available": False, "quality_gate": "unavailable"})
    monkeypatch.setattr(service_main, "ICEBERG_TRACKS_PATH", tmp_path / "missing.zip")
    response = client.get("/api/v1/integration/status")
    assert response.status_code == 200
    status = response.json()
    assert status["operational_decision_ready"] is False
    assert all(component["available"] is False for component in status["research_components"].values())
