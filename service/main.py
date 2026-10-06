"""Small API boundary for Southern Passage research forecasts.

Put this service behind ministry identity, network, and audit controls before
exposing it beyond a trusted development environment.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import re
import secrets
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator
from pyproj import CRS, Geod, Transformer
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from ml.identity import model_artifact_sha256, model_fingerprint
from ml.verify_initial_model import verify_initial_model
from ml.summarize_research_transect import summarize as summarize_research_transect
from service.body_limit import ApiBodySizeLimitMiddleware
from service.decision_support import evidence_gate, fixed_path_counterfactuals, ice_edge_fragility, vessel_screen
from service.glo12_status import forecast_archive_status
from service.iceberg_tracks import observed_tracks_geojson, historical_short_iceberg_projection
from service.route_planner import route_candidates, research_transit_sensitivity
from service.review_workflow import GATES, append_decision, create_case, get_case, list_cases, verify_gateway_assertion

ROOT = Path(__file__).resolve().parents[1]
PORTAL_ROOT = ROOT / "portal" if (ROOT / "portal" / "index.html").is_file() else ROOT
PORTAL_ASSETS = frozenset({"index.html", "app.js", "review-workflow.js", "experience.js", "marine-interactions.js", "styles.css", "refinement.css", "product.css", "cinematic.css", "orbital.css", "marine.css", "polar-orbit.png", "orbital-satellite.png", "antarctic-sea.png", "hero-iceberg.png", "indian-research-vessel.png"})
MODEL_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_MODEL", ROOT / "models" / "sic_g02202_1989_2016.json"))
MODEL_SELECTION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_MODEL_SELECTION", ROOT / "config" / "initial-model.json"))
EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_EVALUATION", ROOT / "outputs" / "polarwatch-viirs-g02202-1989-2016.json"))
HOLDOUT_EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_HOLDOUT_EVALUATION", ROOT / "outputs" / "g02202-2017-2024-holdout.json"))
SENSOR_ERA_EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_SENSOR_ERA_EVALUATION", ROOT / "outputs" / "g02202-2025-era-transfer.json"))
ROLLING_EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_ROLLING_EVALUATION", ROOT / "outputs" / "g02202-1989-2016-rolling-climatology.json"))
MOTION_EVIDENCE_PATH = ROOT / "outputs" / "nsidc0116-advection-2024-2023fit-linked.json"
MOTION_EVIDENCE_SHA256 = "24f4f07aafe9dd533fef6bf968c8dd53a1d0cccec9af637dc0ee5cb431eb0713"
SNPP_2026_EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_SNPP_2026_EVALUATION", ROOT / "outputs" / "polarwatch-viirs-2026-sep01-21-snpp-transfer.json"))
NOAA21_2026_EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_NOAA21_2026_EVALUATION", ROOT / "outputs" / "polarwatch-viirs-2026-sep01-21-noaa21-transfer.json"))
NOAA20_2026_EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_NOAA20_2026_EVALUATION", ROOT / "outputs" / "polarwatch-viirs-2026-sep01-21-noaa20-transfer.json"))
OBSERVATIONS_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_VIIRS_DATA", ROOT / "work" / "datasets" / "polarwatch-viirs"))
SUPPLEMENTARY_OBSERVATIONS_PATHS = [Path(value) for value in os.environ.get("SOUTHERN_PASSAGE_VIIRS_SUPPLEMENT_DATA", "").split(os.pathsep) if value]
VIIRS_SOURCE_STATUS_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_VIIRS_SOURCE_STATUS",
                                               ROOT / "work" / "monitoring" / "polarwatch-source-status.json"))
ACTIVE_VIIRS_ROOT = Path(os.environ.get("SOUTHERN_PASSAGE_ACTIVE_VIIRS_ROOT", ROOT / "work" / "active-viirs"))
ACTIVE_VIIRS_POLICY = Path(os.environ.get("SOUTHERN_PASSAGE_ACTIVE_VIIRS_POLICY", ROOT / "config" / "viirs-research-policy.example.json"))
CORRIDOR_EVALUATION_PATH = Path(os.environ.get("SOUTHERN_PASSAGE_CORRIDOR_EVALUATION", ROOT / "outputs" / "viirs-2024-frozen-corridor-evaluation.json"))
MULTIPLATFORM_CORRIDOR_PATH = ROOT / "outputs" / "viirs-2024-multiplatform-frozen-corridors.json"
MULTIPLATFORM_CORRIDOR_SHA256 = "19abdcc0c26658efd5cfd8cbf1686a1c5dfc6be18d485143d7e13a23e5ce38b9"
ICEBERG_TRACKS_PATH = Path(os.environ.get(
    "SOUTHERN_PASSAGE_ICEBERG_TRACKS",
    ROOT / "work" / "datasets" / "byu-iceberg-v8" / "consolidated_database_v8.0.zip",
))
ICEBERG_EVIDENCE_PATH = ROOT / "outputs" / "byu-iceberg-v8-drift-holdout.json"
ICEBERG_EVIDENCE_SHA256 = "4ca3563b346ae0af3c62df4dcdd825595cf43c4bdd8ee1ccb32ba6d8b5825d57"
GLO12_FORECAST_ROOT = Path(os.environ.get("SOUTHERN_PASSAGE_GLO12_FORECAST_ROOT", ROOT / "work" / "datasets" / "copernicus-glo12-forecast"))
GLO12_VERIFICATION_ROOT = Path(os.environ.get("SOUTHERN_PASSAGE_GLO12_VERIFICATION_ROOT", ROOT / "work" / "verification"))
# Demo-only display gate, not an operational data-quality or voyage-safety threshold.
OBSERVED_FORECAST_MIN_COMMON_COVERAGE = 0.10
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
API_AUTH_TOKEN = os.getenv("SOUTHERN_PASSAGE_API_TOKEN", "")
REVIEW_WORKFLOW_ENABLED = os.getenv("SOUTHERN_PASSAGE_REVIEW_ENABLED", "false").lower() in {"1", "true", "yes"}
REVIEW_ASSERTION_SECRET_FILE = os.getenv("SOUTHERN_PASSAGE_REVIEW_ASSERTION_SECRET_FILE", "")
if REVIEW_ASSERTION_SECRET_FILE and os.getenv("SOUTHERN_PASSAGE_REVIEW_ASSERTION_SECRET"):
    raise RuntimeError("Choose either a review assertion secret file or an environment value, not both.")
REVIEW_ASSERTION_SECRET = (Path(REVIEW_ASSERTION_SECRET_FILE).read_text(encoding="utf-8").strip()
                           if REVIEW_ASSERTION_SECRET_FILE else os.getenv("SOUTHERN_PASSAGE_REVIEW_ASSERTION_SECRET", ""))
REVIEW_ASSERTION_ISSUER = os.getenv("SOUTHERN_PASSAGE_REVIEW_ASSERTION_ISSUER", "")
REVIEW_ASSERTION_AUDIENCE = os.getenv("SOUTHERN_PASSAGE_REVIEW_ASSERTION_AUDIENCE", "southern-passage-review")
REVIEW_DB_PATH = Path(os.getenv("SOUTHERN_PASSAGE_REVIEW_DB", ROOT / "work" / "review" / "reviews.sqlite3"))
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:4173,http://127.0.0.1:4173").split(",") if origin.strip()]
AUDIT_LOGGER = logging.getLogger("southern_passage.api_audit")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
AUDIT_LOGGER.setLevel(logging.INFO)
AUDIT_LOGGER.propagate = False
if not AUDIT_LOGGER.handlers:
    _audit_handler = logging.StreamHandler()
    _audit_handler.setFormatter(logging.Formatter("%(message)s"))
    AUDIT_LOGGER.addHandler(_audit_handler)

def validate_runtime_configuration(environment: str, token: str, cors_origins: list[str]) -> None:
    if environment not in {"development", "test", "production"}:
        raise RuntimeError("APP_ENV must be one of: development, test, production.")
    if environment == "production":
        if len(token) < 32:
            raise RuntimeError("Production requires SOUTHERN_PASSAGE_API_TOKEN with at least 32 characters; provide secrets through the ministry secret manager.")
        if (not cors_origins or "*" in cors_origins
                or any(not origin.startswith("https://") or "localhost" in origin or "127.0.0.1" in origin for origin in cors_origins)):
            raise RuntimeError("Production requires exact HTTPS CORS_ORIGINS without wildcard or localhost origins.")


validate_runtime_configuration(APP_ENV, API_AUTH_TOKEN, CORS_ORIGINS)
if REVIEW_WORKFLOW_ENABLED and (len(REVIEW_ASSERTION_SECRET) < 32 or not REVIEW_ASSERTION_ISSUER):
    raise RuntimeError("Review workflow requires a 32+ character gateway assertion secret and an explicit issuer.")

app = FastAPI(title="Southern Passage Research API", version="0.1.0",
              description="Research-only Antarctic SIC baseline endpoints; not for navigation.",
              docs_url=None if APP_ENV == "production" else "/docs",
              redoc_url=None if APP_ENV == "production" else "/redoc",
              openapi_url=None if APP_ENV == "production" else "/openapi.json")
# SIC arrays are separately capped at 250,000 cells; this also bounds JSON
# overhead and malformed/chunked writes before request parsing allocates memory.
app.add_middleware(ApiBodySizeLimitMiddleware, max_bytes=16 * 1024 * 1024)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])


@app.get("/portal", include_in_schema=False)
def portal_redirect() -> RedirectResponse:
    return RedirectResponse(url="/portal/", status_code=308)


@app.get("/portal/", include_in_schema=False)
def portal_index() -> FileResponse:
    """Serve only the bundled portal shell; the identity gateway protects this path."""
    return FileResponse(PORTAL_ROOT / "index.html", headers={"Cache-Control": "no-store",
                         "X-Content-Type-Options": "nosniff"})


@app.get("/portal/{asset_name}", include_in_schema=False)
def portal_asset(asset_name: str) -> FileResponse:
    if asset_name not in PORTAL_ASSETS:
        raise HTTPException(status_code=404, detail="Portal asset not found.")
    return FileResponse(PORTAL_ROOT / asset_name, headers={"Cache-Control": "no-cache",
                         "X-Content-Type-Options": "nosniff"})


@app.middleware("http")
async def protect_api_routes(request, call_next):
    """Enforce the interim service credential and emit body-free audit events."""
    is_api = request.url.path.startswith("/api/")
    candidate = request.headers.get("x-request-id", "")
    request_id = candidate if REQUEST_ID_RE.fullmatch(candidate) else secrets.token_hex(16)
    request.state.request_id = request_id
    started = time.perf_counter()
    response = None
    if (request.method == "OPTIONS" and request.headers.get("origin")
            and request.headers.get("access-control-request-method")):
        # Browser CORS preflight carries no user credential. The outer identity
        # gateway must still enforce its own policy; the actual API request is
        # authenticated here after CORS has authorized the exact origin.
        pass
    elif is_api and (APP_ENV == "production" or API_AUTH_TOKEN):
        supplied = request.headers.get("authorization", "")
        scheme, _, credential = supplied.partition(" ")
        if scheme.lower() != "bearer" or not credential or not secrets.compare_digest(credential.encode("utf-8"), API_AUTH_TOKEN.encode("utf-8")):
            response = JSONResponse(
                {"detail": "A valid bearer credential is required."}, status_code=401,
                headers={"WWW-Authenticate": "Bearer"})
    if response is None:
        try:
            response = await call_next(request)
        except Exception:
            if is_api:
                log_api_request(request, request_id, 500, started)
            raise
    status_code = response.status_code
    if is_api:
        log_api_request(request, request_id, status_code, started)
    response.headers["X-Request-ID"] = request_id
    return response


def log_api_request(request, request_id: str, status_code: int, started: float) -> None:
    """Emit allowlisted metadata only; never log credentials or request content."""
    event = {"event": "api_request", "request_id": request_id,
             "method": request.method, "path": request.url.path,
             "status_code": status_code,
             "duration_ms": round((time.perf_counter() - started) * 1000, 3)}
    AUDIT_LOGGER.info(json.dumps(event, separators=(",", ":"), sort_keys=True))


SICRow = Annotated[list[float], Field(min_length=1, max_length=500)]
SICGrid = Annotated[list[SICRow], Field(min_length=1, max_length=500)]
MaskRow = Annotated[list[StrictBool], Field(min_length=1, max_length=500)]
ValidityMask = Annotated[list[MaskRow], Field(min_length=1, max_length=500)]


class GridReference(BaseModel):
    crs: str = Field(min_length=3, max_length=2000)
    # GDAL affine order: x-origin, x-pixel, x-row, y-origin, y-column, y-pixel.
    transform: list[float] = Field(min_length=6, max_length=6)

    @field_validator("transform")
    @classmethod
    def finite_nonzero_pixel_scale(cls, value: list[float]) -> list[float]:
        determinant = value[1] * value[5] - value[2] * value[4]
        if not all(math.isfinite(v) for v in value) or not math.isfinite(determinant) or abs(determinant) < 1e-12:
            raise ValueError("Grid transform must be finite and have a non-zero pixel-area determinant")
        return value

    @field_validator("crs")
    @classmethod
    def supported_crs_definition(cls, value: str) -> str:
        try:
            CRS.from_user_input(value)
        except Exception as exc:
            raise ValueError("Grid CRS must be a recognized authority code or CRS definition") from exc
        return value


class GridForecastRequest(BaseModel):
    previous_day_sic: SICGrid
    two_days_prior_sic: SICGrid
    valid_mask: ValidityMask
    grid: GridReference
    target_day_of_year: int = Field(ge=1, le=366)
    day_before: date
    two_days_before: date
    source_product: str = Field(min_length=3, max_length=160)
    source_version: str = Field(min_length=1, max_length=80)


class GeoJSONLineString(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["LineString"]
    coordinates: list[tuple[float, float]] = Field(min_length=2, max_length=500)

    @model_validator(mode="after")
    def valid_wgs84_coordinates(self) -> "GeoJSONLineString":
        previous = None
        for longitude, latitude in self.coordinates:
            if not math.isfinite(longitude) or not math.isfinite(latitude):
                raise ValueError("Route coordinates must be finite WGS84 longitude/latitude pairs.")
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError("Route coordinates must use WGS84 longitude [-180,180] and latitude [-90,90].")
            if previous == (longitude, latitude):
                raise ValueError("Consecutive route vertices must not be identical.")
            previous = (longitude, latitude)
        return self


class VesselResearchProfile(BaseModel):
    """Caller-entered exploratory limits, never authenticated PWOM constraints."""

    model_config = {"extra": "forbid"}
    profile_label: str = Field(min_length=1, max_length=120)
    max_observed_sic_fraction: float = Field(ge=0, le=1)
    minimum_observation_coverage_fraction: float = Field(ge=0, le=1)


class RouteExposureRequest(BaseModel):
    route_id: str = Field(min_length=1, max_length=120)
    observation_date: date
    geometry: GeoJSONLineString
    analysis_half_width_km: float = Field(default=5.0, ge=0.0, le=20.0)
    research_vessel_profile: VesselResearchProfile | None = None


class GeoJSONPoint(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["Point"]
    coordinates: tuple[float, float]

    @field_validator("coordinates")
    @classmethod
    def valid_wgs84_point(cls, value: tuple[float, float]) -> tuple[float, float]:
        longitude, latitude = value
        if (not math.isfinite(longitude) or not math.isfinite(latitude)
                or not -180 <= longitude <= 180 or not -90 <= latitude <= 90):
            raise ValueError("Route endpoints must be finite WGS84 longitude/latitude pairs.")
        return value


class RouteCandidatesRequest(BaseModel):
    model_config = {"extra": "forbid"}
    route_id: str = Field(min_length=1, max_length=120)
    observation_date: date
    origin: GeoJSONPoint
    destination: GeoJSONPoint
    research_open_water_knots: float = Field(default=10.0, gt=0, le=25, allow_inf_nan=False)
    research_ice_affected_knots: float = Field(default=3.0, gt=0, le=12, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_research_speeds(self) -> "RouteCandidatesRequest":
        if self.research_ice_affected_knots > self.research_open_water_knots:
            raise ValueError("Assumed ice-affected speed cannot exceed open-water speed")
        return self


def viirs_observation_index() -> tuple[dict[str, Any], dict[str, tuple[Path, int, str, dict[str, Any], dict[str, Any]]]]:
    """Index checksummed observation files without loading their SIC raster values."""
    global _OBSERVATION_CACHE_SIGNATURE, _OBSERVATION_CACHE_VALUE
    if not OBSERVATIONS_PATH.is_dir():
        raise HTTPException(status_code=503, detail="NOAA VIIRS observation data are not mounted on this service.")
    dataset_paths = [OBSERVATIONS_PATH]
    dataset_paths.extend(SUPPLEMENTARY_OBSERVATIONS_PATHS)
    manifests: list[tuple[Path, dict[str, Any]]] = []
    listed_files: list[tuple[Path, str, int, dict[str, Any], dict[str, Any]]] = []
    try:
        for dataset_path in dataset_paths:
            manifest_path = dataset_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
                raise ValueError("VIIRS acquisition manifest is malformed or missing its file inventory")
            manifests.append((manifest_path, manifest))
            for item in manifest["files"]:
                if not isinstance(item, dict):
                    raise ValueError("Invalid file inventory record")
                name, expected = item.get("file"), item.get("sha256")
                if (not isinstance(name, str) or Path(name).name != name or not name.endswith(".nc")
                        or not isinstance(expected, str) or len(expected) != 64
                        or any(char not in "0123456789abcdef" for char in expected.lower())):
                    raise ValueError("Every NetCDF inventory record must include a safe filename and SHA-256 digest")
                path = dataset_path / name
                if not path.is_file():
                    if manifest.get("complete"):
                        raise ValueError(f"Completed VIIRS dataset is missing {name}")
                    continue
                listed_files.append((path, expected.lower(), path.stat().st_size, manifest, item))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="VIIRS acquisition manifest or file inventory is invalid.") from exc
    if not listed_files:
        raise HTTPException(status_code=503, detail="No checksummed VIIRS files are available in the mounted data directory.")
    signature = (tuple((path.stat().st_mtime_ns, tuple((file.name, size, file.stat().st_mtime_ns, digest)
                       for file, digest, size, _, _ in listed_files if file.parent == path.parent))
                       for path, _ in manifests),)
    if _OBSERVATION_CACHE_SIGNATURE == signature and _OBSERVATION_CACHE_VALUE is not None:
        return _OBSERVATION_CACHE_VALUE
    records: dict[str, tuple[Path, int, str, dict[str, Any], dict[str, Any]]] = {}
    try:
        for path, expected_sha256, _, manifest, item in listed_files:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if not secrets.compare_digest(digest.hexdigest(), expected_sha256):
                raise ValueError(f"SHA-256 mismatch for {path.name}")
            with xr.open_dataset(path, decode_cf=True) as dataset:
                if "time" not in dataset.coords or "IceConc" not in dataset.data_vars:
                    continue
                for index, value in enumerate(dataset.time.values):
                    if np.isnat(value):
                        continue
                    timestamp = np.datetime_as_string(value, unit="s") + "Z"
                    current = records.get(timestamp[:10])
                    if current is None or timestamp > current[2]:
                        records[timestamp[:10]] = (path, index, timestamp, manifest, item)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Mounted VIIRS files could not be indexed as NetCDF observations.") from exc
    if not records:
        raise HTTPException(status_code=503, detail="No dated VIIRS observations are available in the mounted data directory.")
    _OBSERVATION_CACHE_SIGNATURE = signature
    catalog = dict(manifests[0][1])
    catalog["datasets"] = [{"dataset_id": manifest.get("dataset_id"),
                            "dataset_title": manifest.get("dataset_title"),
                            "platform": manifest.get("platform") or {"noaacwVIIRSnppiceconcSP06Daily": "S-NPP", "noaacwVIIRSn21iceconcSP06Daily": "NOAA-21", "noaacwVIIRSn20iceconcSP06Daily": "NOAA-20"}.get(manifest.get("dataset_id")),
                            "download_complete": bool(manifest.get("complete", False)),
                            "selection": manifest.get("selection")}
                           for _, manifest in manifests]
    _OBSERVATION_CACHE_VALUE = (catalog, records)
    return _OBSERVATION_CACHE_VALUE


def summarize_observation_freshness(records: dict[str, tuple[Any, ...]], now: datetime | None = None) -> dict[str, Any]:
    """Describe catalog recency only; this is not a source-quality or safety assessment."""
    newest = max((record[2] for record in records.values()), default=None)
    if newest is None:
        return {"status": "unavailable", "newest_timestamp": None, "age_hours": None,
                "note": "No dated observations are available."}
    observed_at = datetime.fromisoformat(newest.replace("Z", "+00:00"))
    current_time = now or datetime.now(timezone.utc)
    age_hours = (current_time - observed_at).total_seconds() / 3600
    if age_hours < -6:
        status = "future_timestamp"
    elif age_hours <= 48:
        status = "recent"
    elif age_hours <= 168:
        status = "delayed"
    else:
        status = "stale"
    return {"status": status, "newest_timestamp": newest,
            "age_hours": round(max(0.0, age_hours), 1),
            "classification_basis": "demonstration_thresholds_not_approved_operational_limits",
            "thresholds_hours": {"recent_max": 48, "delayed_max": 168},
            "note": "Recency is based only on the newest mounted observation timestamp. Display thresholds are illustrative, not ministry-approved service limits; recency does not verify source timeliness, quality, completeness, or operational fitness."}


def load_viirs_source_status(path: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Read an operator-refreshed NOAA metadata snapshot, separately from mounted-data age."""
    source_path = path or VIIRS_SOURCE_STATUS_PATH
    try:
        report = json.loads(source_path.read_text(encoding="utf-8"))
        if (not isinstance(report, dict) or report.get("schema_version") != 1
                or not isinstance(report.get("products"), list)):
            raise ValueError("invalid source-monitor schema")
        checked_at = datetime.fromisoformat(str(report.get("checked_at", "")).replace("Z", "+00:00"))
        if checked_at.tzinfo is None:
            raise ValueError("source-monitor timestamp must include timezone")
        current_time = now or datetime.now(timezone.utc)
        age_hours = (current_time - checked_at.astimezone(timezone.utc)).total_seconds() / 3600
        if age_hours < -1:
            monitor_status = "future_timestamp"
        elif age_hours > 24:
            monitor_status = "stale_check"
        else:
            monitor_status = "current_check"
        return {"available": True, "monitor_status": monitor_status,
                "checked_at": report["checked_at"], "report_age_hours": round(max(0.0, age_hours), 1),
                "source_status": report.get("status", "unknown"),
                "classification_basis": report.get("classification_basis"),
                "thresholds_hours": report.get("thresholds_hours"), "products": report["products"],
                "warning": report.get("warning", "Upstream metadata only; not operational fitness.")}
    except FileNotFoundError:
        return {"available": False, "monitor_status": "unconfigured", "checked_at": None,
                "products": [],
                "warning": "No operator-refreshed upstream metadata snapshot is mounted; local data status cannot confirm source timeliness."}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {"available": False, "monitor_status": "invalid_report", "checked_at": None,
                "products": [],
                "warning": "The upstream metadata snapshot is missing required fields or is unreadable."}


_OBSERVATION_CACHE_SIGNATURE: tuple[Any, ...] | None = None
_OBSERVATION_CACHE_VALUE: tuple[dict[str, Any], dict[str, tuple[Path, int, str, dict[str, Any], dict[str, Any]]]] | None = None


def load_model() -> dict[str, Any] | None:
    try:
        verify_initial_model(root=ROOT, manifest_path=MODEL_SELECTION_PATH,
                             runtime_model_path=MODEL_PATH)
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if not isinstance(model, dict):
            return None
        coeff = model.get("coefficients")
        try:
            valid_coefficients = isinstance(coeff, list) and len(coeff) == 5 and all(math.isfinite(float(v)) for v in coeff)
        except (TypeError, ValueError):
            valid_coefficients = False
        if (not model.get("model_id") or not valid_coefficients
                or model.get("model_fingerprint") != model_fingerprint(model)
                or model.get("artifact_sha256") != model_artifact_sha256(model)):
            return None
        return model
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def forecast_model_evidence(model: dict[str, Any]) -> dict[str, Any]:
    """Attach the artifact's actual evaluation context without upgrading its claims."""
    training = model.get("training") if isinstance(model.get("training"), dict) else {}
    source_manifest = model.get("source_manifest") if isinstance(model.get("source_manifest"), dict) else {}
    validation = model.get("validation") if isinstance(model.get("validation"), dict) else {}
    persistence_rmse = validation.get("persistence_rmse")
    rmse = validation.get("rmse")
    comparison = None
    if (isinstance(persistence_rmse, (int, float)) and not isinstance(persistence_rmse, bool)
            and isinstance(rmse, (int, float)) and not isinstance(rmse, bool)
            and math.isfinite(persistence_rmse) and persistence_rmse > 0
            and math.isfinite(rmse)):
        comparison = (persistence_rmse - rmse) / persistence_rmse
    external = load_evaluation(EVALUATION_PATH, model)
    regional_transfer_check = None
    if external:
        metrics = external.get("metrics") or {}
        regional_transfer_check = {
            "dataset_id": external.get("dataset_id"),
            "period": external.get("period"),
            "spatial_evaluation": external.get("spatial_evaluation"),
            "grid_cell_samples": metrics.get("grid_cell_samples"),
            "rmse_sic_fraction": metrics.get("rmse"),
            "persistence_rmse_sic_fraction": metrics.get("persistence_rmse"),
            "warning": external.get("warning"),
        }
    return {
        "status": "research_only",
        "artifact_sha256": model.get("artifact_sha256"),
        "training_period": {
            "start": training.get("selected_period_start"),
            "end": training.get("selected_period_end"),
        },
        "training_data": {
            "dataset_id": source_manifest.get("dataset_id"),
            "dataset": source_manifest.get("dataset"),
            "doi": source_manifest.get("doi"),
            "manifest_complete": source_manifest.get("complete"),
        },
        "holdout_scope": model.get("validation_scope"),
        "holdout": {
            "grid_cell_samples": validation.get("grid_cell_samples"),
            "mae_sic_fraction": validation.get("mae"),
            "rmse_sic_fraction": rmse,
            "persistence_rmse_sic_fraction": persistence_rmse,
            "relative_rmse_reduction_vs_persistence": comparison,
        },
        "regional_cross_product_transfer_check": regional_transfer_check,
        "limitations": [
            "Same-product retrospective holdout is not independent cross-product or operational forecast validation.",
            "No calibrated uncertainty interval is produced.",
            "No navigability, vessel-safety, or route-clearance assessment is produced.",
        ],
    }


def load_evaluation(path: Path, model: dict[str, Any]) -> dict[str, Any] | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        source_manifest = report.get("source_manifest") or {}
        period = report.get("period")
        overall = report.get("overall")
        if (report.get("model_id") != model.get("model_id")
                or report.get("model_fingerprint") != model_fingerprint(model)
                or not isinstance(source_manifest, dict)
                or not isinstance(period, dict)
                or not isinstance(overall, dict)):
            return None
        first_target = date.fromisoformat(period["first_target_date"])
        last_target = date.fromisoformat(period["last_target_date"])
        target_days = (last_target - first_target).days + 1
        sample_count = overall.get("grid_cell_samples")
        if target_days < 1 or not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
            return None
        return {"available": True,
                "model_fingerprint": report.get("model_fingerprint"),
                "dataset_id": source_manifest.get("dataset_id"),
                "dataset_complete": bool(source_manifest.get("complete", False)),
                "period": period,
                "spatial_evaluation": report.get("spatial_evaluation"),
                "metrics": overall,
                "area_weighted": report.get("area_weighted"),
                "metric_weighting": report.get("metric_weighting"),
                "by_target_month": report.get("by_target_month"),
                "platform": source_manifest.get("platform") or {
                    "noaacwVIIRSnppiceconcSP06Daily": "S-NPP",
                    "noaacwVIIRSn21iceconcSP06Daily": "NOAA-21",
                    "noaacwVIIRSn20iceconcSP06Daily": "NOAA-20",
                }.get(source_manifest.get("dataset_id")),
                "selection": source_manifest.get("selection"),
                "sample_scope": {
                    "target_days": target_days,
                    "valid_grid_cell_samples": sample_count,
                    "route_scale_evidence": False,
                    "caveat": "Small regional diagnostic only; grid-cell samples are spatially and temporally correlated and are not independent trials. Not route-scale forecast validation.",
                },
                "warning": report.get("warning")}
    except (OSError, json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError):
        return None


def load_external_evaluation(model: dict[str, Any]) -> dict[str, Any] | None:
    return load_evaluation(EVALUATION_PATH, model)


def load_rolling_evaluation() -> dict[str, Any] | None:
    """Load archive-level rolling-fold metrics, explicitly not bound to one model artifact."""
    try:
        report = json.loads(ROLLING_EVALUATION_PATH.read_text(encoding="utf-8"))
        source_manifest = report.get("source_manifest") or {}
        overall = report.get("overall")
        monthly_climatology = report.get("monthly_climatology")
        area_weighted = report.get("area_weighted")
        folds = report.get("folds")
        if (report.get("dataset_id") != "G02202" or not source_manifest.get("complete")
                or not isinstance(overall, dict) or not isinstance(area_weighted, dict)
                or not isinstance(area_weighted.get("overall"), dict)
                or not isinstance(monthly_climatology, dict)
                or not isinstance(area_weighted.get("monthly_climatology"), dict)
                or not isinstance(folds, list) or len(folds) != report.get("fold_count")
                or not folds):
            return None
        return {
            "available": True,
            "dataset_id": "G02202",
            "source_manifest_sha256": report.get("source_manifest_sha256"),
            "validation_period": {
                "first_target_date": folds[0].get("validation_period", {}).get("start"),
                "last_target_date": folds[-1].get("validation_period", {}).get("end"),
            },
            "fold_count": report.get("fold_count"),
            "window_configuration": report.get("window_configuration"),
            "cell_pooled": overall,
            "monthly_climatology": monthly_climatology,
            "area_weighted": area_weighted,
            "model_artifact_binding": "Each fold fits its own earlier-only coefficients. These metrics are an archive-level model-development benchmark, not scores for the currently packaged model artifact.",
            "warning": report.get("warning"),
        }
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return None


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "southern-passage-research-api"}


@app.get("/readyz")
def readiness() -> Any:
    model = load_model()
    payload = {"ready": model is not None, "model_loaded": model is not None,
               "mode": "research-only" if model else "no-model-artifact"}
    return payload if model else JSONResponse(status_code=503, content=payload)


@app.get("/api/v1/forecast-runs/glo12/status")
def glo12_forecast_status() -> dict[str, Any]:
    """Expose only integrity-verified original runs; never imply route readiness."""
    return forecast_archive_status(GLO12_FORECAST_ROOT)


@app.get("/api/v1/research/transect-evidence")
def research_transect_evidence() -> dict[str, Any]:
    """Recheck source-bound reports; never expose a route-safety conclusion."""
    return summarize_research_transect(GLO12_VERIFICATION_ROOT)


@app.get("/api/v1/integration/capabilities")
def integration_capabilities() -> dict[str, Any]:
    """Stable machine-readable contract for a portal gateway or review system."""
    return {
        "schema_version": 1,
        "service": "southern-passage",
        "api_version": "v1",
        "authentication": "gateway_service_bearer_in_production",
        "request_correlation_header": "X-Request-ID",
        "available_workflows": [
            {"id": "glo12_forecast_archive_status", "method": "GET",
             "path": "/api/v1/forecast-runs/glo12/status",
             "output": "integrity-verified original bulletin availability; no skill or safety clearance"},
            {"id": "research_transect_evidence", "method": "GET",
             "path": "/api/v1/research/transect-evidence",
             "output": "date-blocked SIC research diagnostics; no route or navigation clearance"},
            {"id": "observed_sic", "method": "GET", "path": "/api/v1/observations/sic",
             "output": "dated manifest-verified observation catalog"},
            {"id": "route_review_packet", "method": "POST", "path": "/api/v1/reviews/route",
             "output": "versioned JSON observation assessment packet"},
            {"id": "route_review_bundle", "method": "POST", "path": "/api/v1/reviews/route/bundle",
             "output": "paired unsigned JSON packet and GeoJSON map feature"},
            {"id": "model_evidence", "method": "GET", "path": "/api/v1/model",
             "output": "model lineage and research evaluation"},
            {"id": "model_selection", "method": "GET", "path": "/api/v1/model/selection",
             "output": "verified initial research-model selection"},
            {"id": "multiplatform_corridor_evaluation", "method": "GET",
             "path": "/api/v1/evaluations/multiplatform-corridors",
             "output": "frozen three-platform VIIRS corridor coverage and research metrics"},
            {"id": "component_status", "method": "GET", "path": "/api/v1/integration/status",
             "output": "availability and provenance status"},
            {"id": "review_workflow", "method": "GET", "path": "/api/v1/review-workflow/capabilities",
             "output": "gateway SSO and independent review-gate configuration"},
        ],
        "openapi_path": "/api/v1/integration/openapi.json",
        "decision_scope": "research_observation_context_only",
        "navigation_clearance_available": False,
        "portal_responsibilities": ["OIDC identity and role mapping", "actor-attributed gateway audit",
                                     "authoritative record retention", "operational decision workflow"],
    }


@app.get("/api/v1/integration/openapi.json", include_in_schema=False)
def integration_openapi() -> dict[str, Any]:
    """Expose the protected schema even when public docs are disabled."""
    return app.openapi()


def active_viirs_status() -> dict[str, Any]:
    """A previously promoted archive still fails closed when it becomes stale."""
    try:
        from ml.refresh_viirs import active_status, validate_policy
        policy = validate_policy(json.loads(ACTIVE_VIIRS_POLICY.read_text(encoding="utf-8")))
        return active_status(ACTIVE_VIIRS_ROOT, policy, now=datetime.now(timezone.utc))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"available_for_research": False, "operational_decision_ready": False,
                "status": "unavailable", "reason": str(exc)}


def corridor_evaluation_status() -> dict[str, Any]:
    try:
        payload = json.loads(CORRIDOR_EVALUATION_PATH.read_text(encoding="utf-8"))
        if payload.get("scope") != "research_only":
            raise ValueError("Invalid corridor evaluation scope")
        return {"available": True, "quality_gate": payload.get("quality_gate"),
                "report_sha256": hashlib.sha256(CORRIDOR_EVALUATION_PATH.read_bytes()).hexdigest()}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"available": False, "quality_gate": "unavailable", "reason": type(exc).__name__}


def load_multiplatform_corridor_evaluation() -> dict[str, Any] | None:
    try:
        raw = MULTIPLATFORM_CORRIDOR_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() != MULTIPLATFORM_CORRIDOR_SHA256:
            return None
        report = json.loads(raw)
        model = load_model()
        if (model is None or report.get("scope") != "research_only"
                or report.get("quality_gate") != "insufficient_evidence"
                or report.get("operational_decision_ready") is not False
                or report.get("model_fingerprint") != model["model_fingerprint"]
                or report.get("model_artifact_sha256") != model["artifact_sha256"]
                or len(report.get("platforms", [])) != 3):
            return None
        return report
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


@app.get("/api/v1/observations/active/status")
def get_active_observation_status() -> dict[str, Any]:
    return active_viirs_status()


@app.get("/api/v1/evaluations/frozen-corridors")
def get_frozen_corridor_evaluation() -> dict[str, Any]:
    try:
        payload = json.loads(CORRIDOR_EVALUATION_PATH.read_text(encoding="utf-8"))
        if payload.get("scope") != "research_only" or payload.get("operational_decision_ready") is not False:
            raise ValueError("Invalid evaluation scope")
        return payload
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Frozen corridor evaluation unavailable or invalid.") from exc


@app.get("/api/v1/evaluations/multiplatform-corridors")
def get_multiplatform_corridor_evaluation() -> dict[str, Any]:
    report = load_multiplatform_corridor_evaluation()
    if report is None:
        raise HTTPException(status_code=503, detail="Frozen multiplatform evaluation unavailable or invalid.")
    return report


@app.get("/api/v1/integration/status")
def integration_status() -> dict[str, Any]:
    """Report component availability without implying navigation fitness."""
    model = load_model()
    try:
        _, records = viirs_observation_index()
        observations = {"available": True, "count": len(records),
                        "freshness": summarize_observation_freshness(records)}
    except HTTPException as exc:
        observations = {"available": False, "count": 0,
                        "reason": exc.detail if isinstance(exc.detail, str) else "Observation catalog unavailable."}
    monitor = load_viirs_source_status()
    active_feed = active_viirs_status()
    corridor_evaluation = corridor_evaluation_status()
    multiplatform_corridors = load_multiplatform_corridor_evaluation()
    motion = ice_motion_evidence()
    try:
        manifest_path = ICEBERG_TRACKS_PATH.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = manifest.get("archive", {})
        iceberg_available = (manifest.get("complete") is True
                             and item.get("file") == ICEBERG_TRACKS_PATH.name
                             and ICEBERG_TRACKS_PATH.is_file()
                             and ICEBERG_TRACKS_PATH.stat().st_size == item.get("bytes")
                             and hashlib.sha256(ICEBERG_TRACKS_PATH.read_bytes()).hexdigest() == item.get("sha256"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        iceberg_available = False
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "service_available": True,
        "research_components": {
            "sic_model": {"available": model is not None,
                          "fingerprint": model.get("model_fingerprint") if model else None},
            "mounted_observations": observations,
            "upstream_source_monitor": {"available": monitor["available"],
                                        "status": monitor["monitor_status"],
                                        "checked_at": monitor.get("checked_at")},
            "gated_active_observations": {"available": active_feed["available_for_research"],
                                          "status": active_feed["status"]},
            "frozen_corridor_evaluation": corridor_evaluation,
            "multiplatform_corridor_evaluation": {
                "available": multiplatform_corridors is not None,
                "quality_gate": multiplatform_corridors.get("quality_gate") if multiplatform_corridors else "unavailable",
            },
            "ice_motion_evidence": {"available": motion["available"],
                                    "report_sha256": motion.get("report_sha256")},
            "historical_iceberg_archive": {"available": iceberg_available,
                                            "published_coverage_end": "2025-04-22"},
        },
        "operational_decision_ready": False,
        "reason": "Live source availability, vessel constraints, authoritative hazards, and route-outcome validation are not established.",
    }


def authenticated_review_actor(request: Request) -> dict[str, Any]:
    if not REVIEW_WORKFLOW_ENABLED:
        raise HTTPException(503, "The SSO review workflow is not configured.")
    return verify_gateway_assertion(
        request.headers.get("x-sp-actor-assertion", ""),
        secret=REVIEW_ASSERTION_SECRET, issuer=REVIEW_ASSERTION_ISSUER,
        audience=REVIEW_ASSERTION_AUDIENCE, method=request.method,
        path=request.url.path, request_id=request.state.request_id,
    )


@app.get("/api/v1/review-workflow/capabilities")
def review_workflow_capabilities() -> dict[str, Any]:
    return {"schema_version": 1, "enabled": REVIEW_WORKFLOW_ENABLED,
            "identity_boundary": "ministry_oidc_gateway_signed_internal_assertion",
            "gates": GATES, "creator_role": "analyst", "handoff_role": "release_authority",
            "independent_signers_required": True, "navigation_clearance_available": False}


@app.get("/api/v1/review-workflow/me")
def review_workflow_actor(request: Request) -> dict[str, Any]:
    return authenticated_review_actor(request)


@app.get("/api/v1/model")
def model_metadata() -> dict[str, Any]:
    model = load_model()
    if model is None:
        return {"available": False, "message": "No trained model artifact. Load quality-controlled data and run ml/train_sic.py."}
    return {"available": True, "model_id": model.get("model_id"), "target": model.get("target"),
            "model_fingerprint": model.get("model_fingerprint"),
            "artifact_sha256": model.get("artifact_sha256"),
            "training": model.get("training"), "validation": model.get("validation"),
            "training_configuration": model.get("training_configuration"),
            "reproducibility": model.get("reproducibility"),
            "validation_scope": model.get("validation_scope"),
            "same_product_holdout": load_evaluation(HOLDOUT_EVALUATION_PATH, model),
            "sensor_era_transfer": load_evaluation(SENSOR_ERA_EVALUATION_PATH, model),
            "snpp_2026_transfer": load_evaluation(SNPP_2026_EVALUATION_PATH, model),
            "noaa21_2026_transfer": load_evaluation(NOAA21_2026_EVALUATION_PATH, model),
            "noaa20_2026_transfer": load_evaluation(NOAA20_2026_EVALUATION_PATH, model),
            "external_evaluation": load_external_evaluation(model),
            "rolling_origin_benchmark": load_rolling_evaluation(),
            "warning": model.get("warning")}


@app.get("/api/v1/model/selection")
def model_selection() -> dict[str, Any]:
    """Expose the pinned model decision without implying operational clearance."""
    try:
        return verify_initial_model(root=ROOT, manifest_path=MODEL_SELECTION_PATH,
                                    runtime_model_path=MODEL_PATH)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Initial model selection is unavailable or invalid.") from exc


@app.get("/api/v1/research/ice-motion")
def ice_motion_evidence() -> dict[str, Any]:
    """Expose only the reviewed, frozen cross-year research result, not its cell-level data."""
    unavailable = {"available": False, "decision_status": "research_only",
                   "message": "Verified ice-motion evidence is unavailable; do not substitute a live forecast."}
    try:
        raw = MOTION_EVIDENCE_PATH.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != MOTION_EVIDENCE_SHA256:
            return unavailable
        report = json.loads(raw)
        source = report["coefficient_source_report"]
        dates = report["date_range"]
        metrics = report["holdout_metrics"]
        if (report.get("decision_status") != "research_only_not_navigation_or_operational_forecast"
                or source.get("source_last_issue") != "2023-12-30"
                or source.get("sha256") != "85dfadd6e091d5c9fc4f77be0a93832cef9b0e981ae716588b01a13bb2e59611"
                or dates.get("start_issue") != "2024-01-02"
                or dates.get("end_issue") != "2024-12-30"
                or abs(float(report["selected_lambda"]) - 0.7) > 1e-9
                or metrics["evaluated_days"] != 354
                or metrics["n_cell_days"] != 5417459):
            return unavailable
        return {"available": True, "decision_status": "research_only",
                "experiment": "2023-fitted sea-ice motion blend, frozen on 2024",
                "fit_year": 2023, "evaluation_year": 2024,
                "evaluated_days": metrics["evaluated_days"],
                "zero_coverage_days": 10,
                "correlated_cell_days": metrics["n_cell_days"],
                "mae": {"persistence": metrics["persistence_mae"],
                        "motion_blend": metrics["blend_mae"],
                        "reduction_percent": metrics["mae_reduction_percent"]},
                "rmse": {"persistence": metrics["persistence_rmse"],
                         "motion_blend": metrics["blend_rmse"],
                         "reduction_percent": metrics["rmse_reduction_percent"]},
                "report_sha256": digest,
                "report_file": MOTION_EVIDENCE_PATH.name,
                "warning": "Retrospective, same-product, correlated-cell diagnostic. Motion availability timing is unverified; not a live forecast, route validation, or navigation advice."}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return unavailable


@app.get("/api/v1/observations/sic")
def list_sic_observations() -> dict[str, Any]:
    manifest, records = viirs_observation_index()
    dates = sorted(records)
    freshness = summarize_observation_freshness(records)
    return {"available": True, "dataset_id": manifest.get("dataset_id", "noaacwVIIRSnppiceconcSP06Daily"),
            "dataset_title": manifest.get("dataset_title"), "download_complete": bool(manifest.get("complete", False)),
            "selection": manifest.get("selection"),
            "observation_count": len(dates),
            "datasets": manifest.get("datasets", []),
            "observations": [{"date": day, "timestamp": records[day][2]} for day in dates],
            "freshness": freshness,
            "warning": "Observed near-real-time satellite SIC for this subset only; observation dates are sparse. Not a forecast or navigation guidance."}


@app.get("/api/v1/observations/sic/source-status")
def viirs_source_status() -> dict[str, Any]:
    """Return the latest separately checked NOAA catalog metadata snapshot, if mounted."""
    return load_viirs_source_status()


@app.get("/api/v1/icebergs/observed-tracks")
def list_observed_iceberg_tracks(as_of_date: date,
                                  window_days: int = Query(default=30, ge=1, le=90)) -> dict[str, Any]:
    """Return source-measured large-iceberg track snippets within published coverage."""
    return observed_tracks_geojson(ICEBERG_TRACKS_PATH, as_of_date, window_days)


@app.get("/api/v1/research/icebergs/{iceberg_id}/projection")
def project_historical_iceberg(iceberg_id: str, as_of_date: date,
                               lead_days: int = Query(default=1, ge=1, le=2)) -> dict[str, Any]:
    """Bounded, historical short-horizon research baseline for known icebergs."""
    return historical_short_iceberg_projection(
        ICEBERG_TRACKS_PATH, iceberg_id, as_of_date, lead_days,
        ICEBERG_EVIDENCE_PATH, ICEBERG_EVIDENCE_SHA256,
    )


@app.get("/api/v1/observations/sic/{observation_date}")
def get_sic_observation(observation_date: date) -> dict[str, Any]:
    manifest, records = viirs_observation_index()
    record = records.get(observation_date.isoformat())
    if record is None:
        raise HTTPException(status_code=404, detail="No VIIRS observation exists for this date in the mounted regional subset.")
    path, index, timestamp, manifest, file_record = record
    try:
        with xr.open_dataset(path, decode_cf=True) as dataset:
            field = np.asarray(dataset["IceConc"].isel(time=index).squeeze().values, dtype=np.float32)
            x = np.asarray(dataset["cols"].values, dtype=np.float64)
            y = np.asarray(dataset["rows"].values, dtype=np.float64)
            if field.ndim != 2 or field.shape != (len(y), len(x)):
                raise ValueError("Unexpected SIC grid dimensions or coordinate lengths")
            if field.size > 250_000:
                raise HTTPException(status_code=413, detail="Observation grid exceeds the 250,000-cell response limit.")
            if len(x) > 1 and x[0] > x[-1]:
                x, field = x[::-1], field[:, ::-1]
            if len(y) > 1 and y[0] < y[-1]:
                y, field = y[::-1], field[::-1, :]
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="The selected VIIRS grid could not be read safely.") from exc
    dx = float(np.median(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
    dy = float(np.median(np.abs(np.diff(y)))) if len(y) > 1 else 0.0
    if not math.isfinite(dx + dy) or dx <= 0 or dy <= 0 or not np.allclose(np.diff(x), dx, rtol=0.01, atol=0.01) or not np.allclose(np.abs(np.diff(y)), dy, rtol=0.01, atol=0.01):
        raise HTTPException(status_code=422, detail="The selected observation is not on a supported regular rectilinear grid.")
    field[(~np.isfinite(field)) | (field < 0) | (field > 1)] = np.nan
    finite = np.isfinite(field)
    values = np.where(finite, field, None).tolist()
    selection = manifest.get("selection") or {}
    return {"date": observation_date.isoformat(), "timestamp": timestamp,
            "grid": {"crs": selection.get("crs", "EPSG:3976"), "shape": list(field.shape),
                     "x_first_center_m": float(x[0]), "x_last_center_m": float(x[-1]),
                     "y_first_center_m": float(y[0]), "y_last_center_m": float(y[-1]),
                     "pixel_spacing_x_m": dx, "pixel_spacing_y_m": dy,
                     "orientation": "west-to-east columns; north-to-south rows"},
            "sic_fraction": values,
            "source_provenance": {"provider": "NOAA CoastWatch/PolarWatch", "dataset_id": manifest.get("dataset_id"),
                                  "product": manifest.get("dataset_title"),
                                  "platform": manifest.get("platform") or {"noaacwVIIRSnppiceconcSP06Daily": "S-NPP", "noaacwVIIRSn21iceconcSP06Daily": "NOAA-21", "noaacwVIIRSn20iceconcSP06Daily": "NOAA-20"}.get(manifest.get("dataset_id")),
                                  "accessed_at_utc": manifest.get("accessed_at_utc"),
                                  "source_file": path.name, "source_sha256": file_record.get("sha256"),
                                  "crs": selection.get("crs", "EPSG:3976"),
                                  "note": "Observed satellite sea-ice concentration; sparse near-real-time subset, not forecast."}}


def geodesic_line_samples(coordinates: list[tuple[float, float]], spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample a WGS84 GeoJSON LineString at bounded, near-uniform geodesic intervals."""
    geod = Geod(ellps="WGS84")
    longitudes = [coordinates[0][0]]
    latitudes = [coordinates[0][1]]
    distances = [0.0]
    bearings = []
    total_m = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coordinates, coordinates[1:]):
        azimuth, _, length_m = geod.inv(lon1, lat1, lon2, lat2)
        if not bearings:
            bearings.append(azimuth)
        if not math.isfinite(length_m) or length_m <= 0:
            raise HTTPException(status_code=422, detail="Route must have positive finite geodesic length.")
        next_total_m = total_m + length_m
        # Enforce the request work limit before allocating any segment samples.
        if next_total_m > 1_000_000:
            raise HTTPException(status_code=422, detail="Route line exceeds the 1,000 km analysis limit.")
        steps = max(1, math.ceil(length_m / spacing_m))
        for step in range(1, steps + 1):
            if step == steps:
                longitude, latitude = lon2, lat2
            else:
                longitude, latitude, _ = geod.fwd(lon1, lat1, azimuth, length_m * step / steps)
            longitudes.append(longitude)
            latitudes.append(latitude)
            distances.append(total_m + length_m * step / steps)
            bearings.append(azimuth)
        total_m = next_total_m
    return (np.asarray(longitudes, dtype=np.float64), np.asarray(latitudes, dtype=np.float64),
            np.asarray(distances, dtype=np.float64), np.asarray(bearings, dtype=np.float64), total_m)


def geodesic_context_band(longitudes: np.ndarray, latitudes: np.ndarray, bearings: np.ndarray,
                          half_width_m: float, spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create perpendicular geodesic samples across a context band, not a safety corridor."""
    if half_width_m <= 0:
        offsets = np.array([0.0], dtype=np.float64)
    else:
        intervals = max(2, math.ceil(2 * half_width_m / spacing_m))
        if intervals % 2:
            intervals += 1
        offsets = np.linspace(-half_width_m, half_width_m, intervals + 1, dtype=np.float64)
    along_count = len(longitudes)
    if along_count * len(offsets) > 250_000:
        raise HTTPException(status_code=413, detail="Route context-band analysis exceeds the 250,000-point sampling limit; shorten the route or reduce the band width.")
    lon_grid = np.repeat(longitudes[:, None], len(offsets), axis=1)
    lat_grid = np.repeat(latitudes[:, None], len(offsets), axis=1)
    if len(offsets) > 1:
        bearing_grid = np.repeat(bearings[:, None], len(offsets), axis=1)
        offset_grid = np.broadcast_to(offsets[None, :], bearing_grid.shape)
        side_bearing = np.where(offset_grid >= 0, bearing_grid + 90.0, bearing_grid - 90.0)
        geod = Geod(ellps="WGS84")
        lon_grid, lat_grid, _ = geod.fwd(lon_grid, lat_grid, side_bearing, np.abs(offset_grid))
    return lon_grid, lat_grid, offsets


def sample_projected_grid(longitudes: np.ndarray, latitudes: np.ndarray, field: np.ndarray,
                          grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        to_grid = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(grid["crs"]), always_xy=True)
        eastings, northings = to_grid.transform(longitudes, latitudes)
        pixel_x = (np.asarray(eastings, dtype=np.float64) - float(grid["x_first_center_m"])) / float(grid["pixel_spacing_x_m"])
        pixel_y = (float(grid["y_first_center_m"]) - np.asarray(northings, dtype=np.float64)) / float(grid["pixel_spacing_y_m"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The observation grid CRS could not be used for route sampling.") from exc
    finite = np.isfinite(pixel_x) & np.isfinite(pixel_y)
    cols = np.full(pixel_x.shape, -1, dtype=np.int64)
    rows = np.full(pixel_y.shape, -1, dtype=np.int64)
    cols[finite] = np.rint(pixel_x[finite]).astype(np.int64)
    rows[finite] = np.rint(pixel_y[finite]).astype(np.int64)
    inside = finite & (rows >= 0) & (rows < field.shape[0]) & (cols >= 0) & (cols < field.shape[1])
    sampled = np.full(pixel_x.shape, np.nan, dtype=np.float32)
    sampled[inside] = field[rows[inside], cols[inside]]
    valid = np.isfinite(sampled) & (sampled >= 0) & (sampled <= 1)
    return pixel_x, pixel_y, np.where(valid, sampled, np.nan)


def route_endpoint_cell(point: tuple[float, float], grid: dict[str, Any],
                        shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[float, float], float]:
    """Map a WGS84 endpoint to its nearest observed pixel center."""
    try:
        forward = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(grid["crs"]), always_xy=True)
        inverse = Transformer.from_crs(CRS.from_user_input(grid["crs"]), CRS.from_epsg(4326), always_xy=True)
        x, y = forward.transform(*point)
        col = int(np.rint((x - float(grid["x_first_center_m"])) / float(grid["pixel_spacing_x_m"])))
        row = int(np.rint((float(grid["y_first_center_m"]) - y) / float(grid["pixel_spacing_y_m"])))
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            raise HTTPException(status_code=422, detail="An endpoint falls outside the selected observation grid.")
        center_x = float(grid["x_first_center_m"]) + col * float(grid["pixel_spacing_x_m"])
        center_y = float(grid["y_first_center_m"]) - row * float(grid["pixel_spacing_y_m"])
        center_lon, center_lat = inverse.transform(center_x, center_y)
        _, _, snap_m = Geod(ellps="WGS84").inv(point[0], point[1], center_lon, center_lat)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The observation grid CRS could not be used for route planning.") from exc
    return (row, col), (float(center_lon), float(center_lat)), float(snap_m / 1000.0)


@app.post("/api/v1/route/candidates")
def plan_observed_route_candidates(request: RouteCandidatesRequest) -> dict[str, Any]:
    """Compare grid paths against observed SIC, without claiming navigability."""
    scene = get_sic_observation(request.observation_date)
    field = np.asarray(scene["sic_fraction"], dtype=np.float32)
    grid = scene["grid"]
    if field.ndim != 2 or field.size > 250_000:
        raise HTTPException(status_code=413, detail="The observation grid exceeds the route-planning work limit.")
    origin_cell, snapped_origin, origin_snap_km = route_endpoint_cell(
        request.origin.coordinates, grid, field.shape)
    destination_cell, snapped_destination, destination_snap_km = route_endpoint_cell(
        request.destination.coordinates, grid, field.shape)
    valid = np.isfinite(field) & (field >= 0) & (field <= 1)
    try:
        candidates = route_candidates(
            field, valid, origin_cell, destination_cell,
            float(grid["pixel_spacing_x_m"]), float(grid["pixel_spacing_y_m"]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    counterfactual = fixed_path_counterfactuals(
        field, candidates, float(grid["pixel_spacing_x_m"]), float(grid["pixel_spacing_y_m"]))
    to_wgs84 = Transformer.from_crs(CRS.from_user_input(grid["crs"]), CRS.from_epsg(4326), always_xy=True)
    output_candidates = []
    for candidate in candidates:
        cells = candidate.pop("display_cells")
        path_cells = candidate.pop("path_cells")
        candidate["research_transit_sensitivity"] = research_transit_sensitivity(
            field, path_cells, float(grid["pixel_spacing_x_m"]), float(grid["pixel_spacing_y_m"]),
            request.research_open_water_knots, request.research_ice_affected_knots)
        projected_x = [float(grid["x_first_center_m"]) + col * float(grid["pixel_spacing_x_m"])
                       for _, col in cells]
        projected_y = [float(grid["y_first_center_m"]) - row * float(grid["pixel_spacing_y_m"])
                       for row, _ in cells]
        route_lons, route_lats = to_wgs84.transform(projected_x, projected_y)
        candidate["geometry"] = {"type": "LineString",
                                  "coordinates": [[float(lon), float(lat)]
                                                  for lon, lat in zip(route_lons, route_lats)]}
        candidate["grid_pixel_coordinates"] = [[int(col), int(row)] for row, col in cells]
        candidate["start_grid_cell"] = list(origin_cell)
        candidate["end_grid_cell"] = list(destination_cell)
        output_candidates.append(candidate)
    return {
        "analysis_type": "observed_sic_grid_route_candidates",
        "route_id": request.route_id,
        "observation_date": scene["date"],
        "observation_timestamp": scene["timestamp"],
        "input_data": "selected single-date satellite SIC observation; no forecast field",
        "endpoints": {
            "requested_origin": {"type": "Point", "coordinates": list(request.origin.coordinates)},
            "requested_destination": {"type": "Point", "coordinates": list(request.destination.coordinates)},
            "grid_center_origin": {"type": "Point", "coordinates": list(snapped_origin), "snap_distance_km": origin_snap_km},
            "grid_center_destination": {"type": "Point", "coordinates": list(snapped_destination), "snap_distance_km": destination_snap_km},
            "endpoint_mapping": "nearest grid-cell center; both selected cells require valid observed SIC",
        },
        "candidates": output_candidates,
        "fixed_path_counterfactuals": counterfactual,
        "navigation_suitability": "not_assessed",
        "fuel_or_transit_time_estimates": None,
        "research_time_assumptions": {
            "open_water_knots": request.research_open_water_knots,
            "ice_affected_knots": request.research_ice_affected_knots,
            "speed_relation": "linear interpolation between user-visible assumed speeds using edge-midpoint observed SIC",
            "speed_sensitivity": "80% to 120% of both assumed speeds; not a calibrated uncertainty or confidence interval",
            "high_sic_abstention_fraction": 0.7,
            "fuel": "not estimated without a documented vessel-specific burn curve",
            "status": "research_only_not_eta_or_navigation_advice",
        },
        "grid": grid,
        "source_provenance": scene["source_provenance"],
        "warning": ("These are alternative paths through the selected observed SIC raster, not forecast routes. "
                    "The concentration weights are illustrative, not calibrated to a vessel. No iceberg, "
                    "wind/current, bathymetry, thickness, ship performance, regulation, fuel, verified transit-time, "
                    "clearance, passability, or safety assessment is included. Do not use for navigation."),
    }


@app.post("/api/v1/route/exposure")
def analyze_route_exposure(request: RouteExposureRequest) -> dict[str, Any]:
    """Describe observed SIC sampled along a user route; no navigability is inferred."""
    scene = get_sic_observation(request.observation_date)
    grid = scene["grid"]
    field = np.asarray(scene["sic_fraction"], dtype=np.float32)
    dx, dy = float(grid["pixel_spacing_x_m"]), float(grid["pixel_spacing_y_m"])
    spacing_m = min(dx, dy) / 2.0
    lons, lats, route_distance_m, bearings, total_distance_m = geodesic_line_samples(
        request.geometry.coordinates, spacing_m)
    pixel_x, pixel_y, sampled = sample_projected_grid(lons, lats, field, grid)
    route_trace = [[float(x), float(y)] if math.isfinite(float(x)) and math.isfinite(float(y)) else None
                   for x, y in zip(pixel_x, pixel_y)]
    valid = np.isfinite(sampled) & (sampled >= 0) & (sampled <= 1)
    values = sampled[valid].astype(np.float64)
    coverage = float(np.count_nonzero(valid) / len(valid))
    coverage_status = "no_valid_data" if values.size == 0 else "complete" if coverage >= 0.999 else "partial"
    total_km = total_distance_m / 1000.0
    summary = None
    concentration_share = None
    profile = []
    band_lons, band_lats, band_offsets = geodesic_context_band(
        lons, lats, bearings, request.analysis_half_width_km * 1000.0, spacing_m)
    band_pixel_x, band_pixel_y, band_sampled = sample_projected_grid(band_lons, band_lats, field, grid)
    band_valid = np.isfinite(band_sampled) & (band_sampled >= 0) & (band_sampled <= 1)
    band_values = band_sampled[band_valid].astype(np.float64)
    band_coverage = float(np.count_nonzero(band_valid) / band_valid.size)
    band_summary = None
    if band_values.size:
        band_summary = {"mean": float(np.mean(band_values)), "median": float(np.median(band_values)),
                        "p95": float(np.percentile(band_values, 95)), "maximum": float(np.max(band_values))}
    left = [[float(x), float(y)] if math.isfinite(float(x)) and math.isfinite(float(y)) else None
            for x, y in zip(band_pixel_x[:, 0], band_pixel_y[:, 0])]
    right = [[float(x), float(y)] if math.isfinite(float(x)) and math.isfinite(float(y)) else None
             for x, y in zip(band_pixel_x[:, -1], band_pixel_y[:, -1])]
    band_polygon = left + list(reversed(right)) + ([left[0]] if left and left[0] is not None else [])
    if any(point is None for point in band_polygon):
        band_polygon = []
    if values.size:
        summary = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                   "p95": float(np.percentile(values, 95)), "maximum": float(np.max(values))}
        concentration_share = {
            "0_to_0.15": float(np.mean(values <= 0.15)),
            "over_0.15_to_0.5": float(np.mean((values > 0.15) & (values <= 0.5))),
            "over_0.5_to_1": float(np.mean(values > 0.5)),
        }
    bins = np.floor(route_distance_m / 5000.0).astype(np.int64)
    for bin_index in np.unique(bins):
        in_bin = bins == bin_index
        local = sampled[in_bin & valid].astype(np.float64)
        start_km, end_km = bin_index * 5.0, min((bin_index + 1) * 5.0, total_km)
        profile.append({"start_km": float(start_km), "end_km": float(end_km),
                        "valid_sample_count": int(len(local)), "total_sample_count": int(np.count_nonzero(in_bin)),
                        "data_coverage_fraction": float(len(local) / np.count_nonzero(in_bin)),
                        "mean_sic_fraction": float(local.mean()) if len(local) else None,
                        "maximum_sic_fraction": float(local.max()) if len(local) else None})
        band_bin = in_bin[:, None] & np.ones((1, len(band_offsets)), dtype=bool)
        band_local_mask = band_valid[band_bin]
        band_local = band_sampled[band_bin][band_local_mask].astype(np.float64)
        profile[-1].update({"context_band_valid_sample_count": int(len(band_local)),
                            "context_band_total_sample_count": int(np.count_nonzero(band_bin)),
                            "context_band_coverage_fraction": float(len(band_local) / np.count_nonzero(band_bin)),
                            "context_band_mean_sic_fraction": float(band_local.mean()) if len(band_local) else None,
                            "context_band_maximum_sic_fraction": float(band_local.max()) if len(band_local) else None})
    result = {
        "analysis_type": ("observed_sic_route_centerline_and_context_band_screen"
                           if request.analysis_half_width_km > 0 else "observed_sic_route_centerline_screen"),
        "route_id": request.route_id,
        "observation_date": scene["date"],
        "observation_timestamp": scene["timestamp"],
        "route_length_km": total_km,
        "navigation_suitability": "not_assessed",
        "data_coverage_status": coverage_status,
        "sampling": {"method": "WGS84 geodesic samples projected to the observation grid; nearest pixel centre",
                     "sample_spacing_m": spacing_m, "valid_sample_count": int(values.size),
                     "total_sample_count": int(len(valid)), "valid_sample_fraction": coverage,
                     "approximate_covered_centerline_km": total_km * coverage,
                     "context_band_half_width_km": request.analysis_half_width_km,
                     "context_band_total_width_km": request.analysis_half_width_km * 2,
                     "context_band_cross_track_offsets_m": band_offsets.tolist(),
                     "context_band_sample_spacing_m": spacing_m,
                     "context_band_valid_sample_count": int(np.count_nonzero(band_valid)),
                     "context_band_total_sample_count": int(band_valid.size),
                     "context_band_valid_sample_fraction": band_coverage,
                     "context_band_data_coverage_status": "no_valid_data" if band_values.size == 0 else "complete" if band_coverage >= 0.999 else "partial",
                     "context_band_navigation_corridor": False,
                     "corridor_width_assessed": request.analysis_half_width_km > 0},
        "centerline_pixel_coordinates": route_trace,
        "context_band_pixel_polygon": band_polygon if request.analysis_half_width_km > 0 else [],
        "context_band_sic_summary_fraction": band_summary,
        "sic_summary_fraction": summary,
        "sic_concentration_share": concentration_share,
        "profile_5km": profile,
        "grid": grid,
        "source_provenance": scene["source_provenance"],
        "warning": "Observed satellite SIC on the selected date only. The optional swath is a fixed-width data-context band, not a vessel-specific navigation corridor or forecast. No wind, currents, bathymetry, vessel capability, regulations, or navigability/safety assessment is included.",
    }
    result["evidence_gate"] = evidence_gate(result)
    result["ice_edge_fragility"] = ice_edge_fragility(profile)
    result["vessel_research_screen"] = (vessel_screen(
        result, max_sic=request.research_vessel_profile.max_observed_sic_fraction,
        min_coverage=request.research_vessel_profile.minimum_observation_coverage_fraction)
        if request.research_vessel_profile else None)
    return result


@app.post("/api/v1/reviews/route")
def create_route_review_packet(route: RouteExposureRequest, http_request: Request) -> dict[str, Any]:
    """Create a portable, unsigned evidence packet for a human review workflow."""
    result = analyze_route_exposure(route)
    source = result["source_provenance"]
    input_snapshot = route.model_dump(mode="json")
    analysis_key = hashlib.sha256(json.dumps({
        "input": input_snapshot,
        "source_file_sha256": source.get("source_sha256"),
        "source_timestamp": result["observation_timestamp"],
        "analysis_type": result["analysis_type"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    packet = {
        "schema_version": 1,
        "packet_id": str(uuid.uuid4()),
        "analysis_key": analysis_key,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": http_request.state.request_id,
        "route": input_snapshot,
        "observation": {"date": result["observation_date"],
                        "timestamp": result["observation_timestamp"],
                        "source_provenance": source},
        "analysis": {"type": result["analysis_type"],
                     "length_km": result["route_length_km"],
                     "coverage_status": result["data_coverage_status"],
                     "sampling": result["sampling"],
                     "centerline_sic_fraction": result["sic_summary_fraction"],
                     "context_band_sic_fraction": result["context_band_sic_summary_fraction"],
                     "profile_5km": result["profile_5km"],
                     "evidence_gate": result["evidence_gate"],
                     "ice_edge_fragility": result["ice_edge_fragility"],
                     "vessel_research_screen": result["vessel_research_screen"]},
        "review": {"status": "awaiting_human_review",
                   "navigation_suitability": "not_assessed",
                   "decision": None,
                   "missing_evidence": ["vessel capability", "wind and ocean forecasts",
                                        "bathymetry and chart constraints", "current iceberg census",
                                        "approved operational thresholds"]},
        "integrity": {"type": "sha256_unsigned", "scope": "packet_without_integrity"},
    }
    packet["integrity"]["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in packet.items() if key != "integrity"},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return packet


def geojson_from_review_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Build a GIS feature from the exact packet it identifies."""
    analysis = packet["analysis"]
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": packet["packet_id"],
            "geometry": packet["route"]["geometry"],
            "properties": {
                "route_id": packet["route"]["route_id"],
                "observation_date": packet["observation"]["date"],
                "observation_timestamp": packet["observation"]["timestamp"],
                "source_sha256": packet["observation"]["source_provenance"].get("source_sha256"),
                "packet_sha256": packet["integrity"]["sha256"],
                "packet_id": packet["packet_id"],
                "decision_status": analysis["evidence_gate"]["decision_status"],
                "observation_display_status": analysis["evidence_gate"]["observation_display_status"],
                "fragile_segment_count": analysis["ice_edge_fragility"]["fragile_segment_count"],
                "vessel_research_screen": analysis["vessel_research_screen"],
                "navigation_clearance": False,
            },
        }],
        "southern_passage_metadata": {
            "schema_version": 1,
            "packet_integrity": "SHA-256 of the associated unsigned JSON packet; this export is not independently signed",
            "full_packet_endpoint": "/api/v1/reviews/route",
            "standards_note": "RFC 7946 GeoJSON; not an OGC API-EDR conformance claim",
        },
    }


@app.post("/api/v1/reviews/route/bundle")
def export_route_review_bundle(route: RouteExposureRequest, http_request: Request) -> dict[str, Any]:
    """Return JSON and GeoJSON together with one shared packet identifier/digest."""
    packet = create_route_review_packet(route, http_request)
    return {"packet": packet, "geojson": geojson_from_review_packet(packet)}


@app.post("/api/v1/reviews/route.geojson")
def export_route_review_geojson(route: RouteExposureRequest, http_request: Request) -> dict[str, Any]:
    """Standalone GeoJSON; use bundle to pair it with the identical JSON packet."""
    return geojson_from_review_packet(create_route_review_packet(route, http_request))


class EvidenceReference(BaseModel):
    model_config = {"extra": "forbid"}
    record_id: str = Field(min_length=3, max_length=120, pattern=r"^[A-Za-z0-9._:/-]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReviewDecisionRequest(BaseModel):
    model_config = {"extra": "forbid"}
    gate: Literal["science", "maritime", "security", "operations", "handoff"]
    decision: Literal["reviewed", "blocked", "changes_required", "package"]
    rationale: str = Field(min_length=30, max_length=2000)
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=10)


class CreateReviewCaseRequest(BaseModel):
    model_config = {"extra": "forbid"}
    route: RouteExposureRequest
    expected_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@app.post("/api/v1/review-workflow/cases", status_code=201)
def file_review_case(submission: CreateReviewCaseRequest, http_request: Request) -> dict[str, Any]:
    actor = authenticated_review_actor(http_request)
    packet = create_route_review_packet(submission.route, http_request)
    if packet.get("observation", {}).get("source_provenance", {}).get("source_sha256") != submission.expected_source_sha256:
        raise HTTPException(409, "The source observation changed. Review the route again before filing.")
    return create_case(REVIEW_DB_PATH, packet, actor)


@app.get("/api/v1/review-workflow/cases")
def review_case_list(http_request: Request, limit: int = Query(50, ge=1, le=100)) -> dict[str, Any]:
    authenticated_review_actor(http_request)
    return {"items": list_cases(REVIEW_DB_PATH, limit=limit), "limit": limit}


@app.get("/api/v1/review-workflow/cases/{case_id}")
def review_case_detail(case_id: uuid.UUID, http_request: Request) -> dict[str, Any]:
    authenticated_review_actor(http_request)
    return get_case(REVIEW_DB_PATH, str(case_id))


@app.post("/api/v1/review-workflow/cases/{case_id}/decisions")
def record_review_decision(case_id: uuid.UUID, decision: ReviewDecisionRequest,
                           http_request: Request) -> dict[str, Any]:
    actor = authenticated_review_actor(http_request)
    return append_decision(REVIEW_DB_PATH, str(case_id), gate=decision.gate,
                           decision=decision.decision, actor=actor,
                           rationale=decision.rationale,
                           evidence=[item.model_dump() for item in decision.evidence],
                           request_id=http_request.state.request_id)


@app.post("/api/v1/forecast/sic")
def forecast_sic(request: GridForecastRequest) -> dict[str, Any]:
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model artifact is installed.")
    a, b = request.previous_day_sic, request.two_days_prior_sic
    if request.day_before - request.two_days_before != timedelta(days=1):
        raise HTTPException(status_code=422, detail="Input observation dates must be consecutive calendar days.")
    forecast_date = request.day_before + timedelta(days=1)
    if forecast_date.timetuple().tm_yday != request.target_day_of_year:
        raise HTTPException(status_code=422, detail="target_day_of_year must match the day after the latest input date.")
    if not a or not b or not a[0] or len(a) != len(b) or any(len(row) != len(a[0]) for row in a + b):
        raise HTTPException(status_code=422, detail="Input grids must be non-empty, rectangular, and have matching shapes.")
    if len(request.valid_mask) != len(a) or any(len(row) != len(a[0]) for row in request.valid_mask):
        raise HTTPException(status_code=422, detail="valid_mask must match the SIC grid shape.")
    if len(a) * len(a[0]) > 250_000:
        raise HTTPException(status_code=413, detail="Grid exceeds the 250,000-cell request limit.")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for grid in (a, b) for row in grid for v in row):
        raise HTTPException(status_code=422, detail="SIC values must be finite fractions from 0 to 1; missing cells need a separate mask.")
    coeff = model.get("coefficients", [])
    if len(coeff) != 5:
        raise HTTPException(status_code=500, detail="Model artifact has an incompatible coefficient set.")
    phase = 2 * math.pi * (request.target_day_of_year - 1) / 365.2425
    c0, c1, c2, c3, c4 = coeff
    previous = np.asarray(a, dtype=np.float64)
    prior = np.asarray(b, dtype=np.float64)
    valid = np.asarray(request.valid_mask, dtype=np.bool_)
    predicted = np.clip(c0 + c1 * previous + c2 * prior + c3 * math.sin(phase) + c4 * math.cos(phase), 0.0, 1.0)
    predicted[~valid] = np.nan
    result = np.where(np.isfinite(predicted), predicted, None).tolist()
    return {"run_id": str(uuid.uuid4()),
            "model_id": model.get("model_id"), "model_fingerprint": model_fingerprint(model),
            "artifact_sha256": model.get("artifact_sha256"),
            "decision_status": "research_only",
            "model_evidence": forecast_model_evidence(model),
            "target_date": forecast_date.isoformat(),
            "target_day_of_year": request.target_day_of_year,
            "source_provenance": {"product": request.source_product, "version": request.source_version,
                                  "two_days_before": request.two_days_before.isoformat(),
                                  "day_before": request.day_before.isoformat(),
                                  "note": "Source identifiers and dates are caller-supplied; upstream authenticity/quality is not independently verified by this research API."},
            "grid_shape": [len(result), len(result[0])], "valid_cell_count": int(np.count_nonzero(valid)),
            "grid": request.grid.model_dump(),
            "valid_mask": request.valid_mask, "sic_fraction": result,
            "generated_at": datetime.now(timezone.utc).isoformat(), "warning": model.get("warning")}


@app.get("/api/v1/observations/sic/{observation_date}/forecast")
def forecast_from_mounted_observations(observation_date: date) -> dict[str, Any]:
    """Forecast one day ahead only from two consecutive mounted, checksummed scenes."""
    previous_date = observation_date - timedelta(days=1)
    current = get_sic_observation(observation_date)
    try:
        previous = get_sic_observation(previous_date)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=422, detail=f"A consecutive VIIRS observation for {previous_date.isoformat()} is required for this one-day research outlook.") from exc
        raise
    if (current.get("date") != observation_date.isoformat()
            or previous.get("date") != previous_date.isoformat()):
        raise HTTPException(status_code=503, detail="Observation service returned an unexpected date for the one-day forecast inputs.")

    current_grid, previous_grid = current["grid"], previous["grid"]
    current_source, previous_source = current["source_provenance"], previous["source_provenance"]
    if (current_source.get("dataset_id") != previous_source.get("dataset_id")
            or current_source.get("platform") != previous_source.get("platform")):
        raise HTTPException(status_code=422, detail="Consecutive observations must come from the same VIIRS product and platform.")
    for key in ("shape", "crs", "pixel_spacing_x_m", "pixel_spacing_y_m",
                "x_first_center_m", "x_last_center_m", "y_first_center_m", "y_last_center_m"):
        left, right = current_grid.get(key), previous_grid.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            aligned = math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
        else:
            aligned = left == right
        if not aligned:
            raise HTTPException(status_code=422, detail="Consecutive observations must share an identical grid and CRS.")

    current_values, previous_values = current["sic_fraction"], previous["sic_fraction"]
    valid_mask = [[a is not None and b is not None for a, b in zip(row_a, row_b)]
                  for row_a, row_b in zip(current_values, previous_values)]
    valid_count = sum(sum(row) for row in valid_mask)
    total_count = len(valid_mask) * (len(valid_mask[0]) if valid_mask else 0)
    coverage_fraction = valid_count / total_count if total_count else 0.0
    if coverage_fraction < OBSERVED_FORECAST_MIN_COMMON_COVERAGE:
        raise HTTPException(status_code=422, detail=(
            f"Only {valid_count} of {total_count} grid cells are valid in both consecutive scenes "
            f"({coverage_fraction:.2%}); the research display requires at least "
            f"{OBSERVED_FORECAST_MIN_COMMON_COVERAGE:.0%} common coverage. This demo rendering floor "
            "is not an approved operational threshold and does not imply spatially uniform coverage."
        ))
    current_filled = [[a if a is not None else 0.0 for a in row] for row in current_values]
    previous_filled = [[a if a is not None else 0.0 for a in row] for row in previous_values]
    dx, dy = current_grid["pixel_spacing_x_m"], current_grid["pixel_spacing_y_m"]
    transform = [current_grid["x_first_center_m"] - dx / 2, dx, 0.0,
                 current_grid["y_first_center_m"] + dy / 2, 0.0, -dy]
    request = GridForecastRequest.model_validate({
        "previous_day_sic": current_filled,
        "two_days_prior_sic": previous_filled,
        "valid_mask": valid_mask,
        "grid": {"crs": current_grid["crs"], "transform": transform},
        "target_day_of_year": (observation_date + timedelta(days=1)).timetuple().tm_yday,
        "day_before": observation_date,
        "two_days_before": previous_date,
        "source_product": current_source.get("product") or current_source.get("dataset_id") or "VIIRS",
        "source_version": "manifest-backed observation subset",
    })
    report = forecast_sic(request)
    report["initialization_date"] = observation_date.isoformat()
    report["input_coverage"] = {
        "valid_common_cells": valid_count,
        "total_grid_cells": total_count,
        "common_coverage_fraction": coverage_fraction,
        "display_floor_fraction": OBSERVED_FORECAST_MIN_COMMON_COVERAGE,
        "threshold_status": "demo_only_not_operationally_approved",
        "spatial_uniformity_assessed": False,
    }
    report["source_provenance"] = {
        "provider": "NOAA CoastWatch/PolarWatch",
        "dataset_id": current_source.get("dataset_id"),
        "platform": current_source.get("platform"),
        "input_observations": [
            {"date": item["date"], "timestamp": item["timestamp"],
             "source_file": item["source_provenance"].get("source_file"),
             "source_sha256": item["source_provenance"].get("source_sha256")}
            for item in (previous, current)
        ],
        "note": "Both input files passed the mounted-manifest SHA-256 check. This is a one-day lag-regression research outlook from two consecutive regional VIIRS scenes, not an operational forecast or navigation recommendation.",
    }
    return report
