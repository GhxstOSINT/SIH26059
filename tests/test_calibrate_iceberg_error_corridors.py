import csv
import hashlib
import json
from datetime import date, timedelta
import zipfile

from pyproj import Geod

from ml.calibrate_iceberg_error_corridors import evaluate, summarize_corridor


def _make_archive(tmp_path):
    archive = tmp_path / "consolidated_database_v8.0.zip"
    geod = Geod(ellps="WGS84")
    rows = ["date,ascat_1,ascat_2,ascat_3"]
    lon, lat = 30.0, -67.0
    bearing = 45.0
    first = date(2009, 1, 1)
    last = date(2018, 12, 31)
    day = first
    while day <= last:
        rows.append(f"{day.year}{day.timetuple().tm_yday:03d},{lat:.10f},{lon:.10f},0")
        lon, lat, back_azimuth = geod.fwd(lon, lat, bearing, 100)
        bearing = (back_azimuth + 180.0) % 360.0
        day += timedelta(days=1)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tracks/a1.csv", "\n".join(rows) + "\n")
    manifest = {"complete": True, "archive": {
        "file": archive.name, "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return archive


def test_corridor_quantiles_are_frozen_and_coverage_is_measured(tmp_path):
    archive = _make_archive(tmp_path)
    report = evaluate(archive, date(2013, 12, 31), date(2014, 1, 1),
                      date(2016, 12, 31), date(2017, 1, 1), date(2018, 12, 31))

    assert report["training_examples"] > 0
    assert report["calibration_examples"] > 0
    assert report["holdout_examples"] > 0
    velocity = report["by_next_report_lead"]["1-2d"]["constant_velocity"]
    assert velocity["status"] == "scored_next_report_distribution"
    assert velocity["holdout_empirical_coverage"]["90_percent_radius"] == 1.0
    assert velocity["calibrated_radius_km"]["90"] < 0.001


def test_corridor_summary_reports_missing_calibration_or_holdout_samples():
    assert summarize_corridor([], [1.0])["status"] == "insufficient_samples"
    assert summarize_corridor([1.0], [])["holdout_count"] == 0
