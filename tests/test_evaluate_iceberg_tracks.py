import csv
from datetime import date, timedelta
import zipfile

from pyproj import Geod

from ml.evaluate_iceberg_tracks import evaluate, parse_doy, read_observed_tracks


def _doy(day):
    return f"{day.year}{day.timetuple().tm_yday:03d}"


def test_parse_doy_rejects_invalid_day():
    assert parse_doy("2020060") == date(2020, 2, 29)
    try:
        parse_doy("2021366")
    except ValueError:
        pass
    else:
        raise AssertionError("non-leap year day 366 must be rejected")


def test_reader_uses_only_flagged_sensor_positions_and_ignores_interpolation(tmp_path):
    archive = tmp_path / "tracks.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tracks/a1.csv",
                        "date,qscat_1,qscat_2,qscat_3\n"
                        "2020001,-65,179,0\n"
                        "2020002,-65,-179,1\n")
    tracks, counts = read_observed_tracks(archive)
    assert counts["files"] == 1
    assert counts["track_days"] == 1
    assert len(tracks["a1"]) == 1
    assert tracks["a1"][0][2] == 179


def test_reader_accepts_legacy_source_columns_without_interpolation_marker(tmp_path):
    archive = tmp_path / "legacy.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tracks/old.csv", "date,ers_1,ers_2\n1992001,-66,30\n")
    tracks, _ = read_observed_tracks(archive)
    assert tracks["old"][0][0:2] == (date(1992, 1, 1), -66.0)
    assert abs(tracks["old"][0][2] - 30.0) < 1e-12


def test_fitted_drift_baseline_is_chronological_and_scores_observed_targets(tmp_path):
    archive = tmp_path / "tracks.zip"
    geod = Geod(ellps="WGS84")
    rows = ["date,ascat_1,ascat_2,ascat_3"]
    lon, lat = 10.0, -70.0
    start = date(2017, 1, 1)
    for offset in range(12):
        rows.append(f"{_doy(start + timedelta(days=offset))},{lat:.9f},{lon:.9f},0")
        lon, lat, _ = geod.fwd(lon, lat, 0, 10000)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tracks/b15.csv", "\n".join(rows) + "\n")

    report = evaluate(archive, date(2017, 1, 5), date(2017, 1, 6), date(2017, 1, 12))
    assert report["training_examples"] == 3
    assert report["test_examples"] == 7
    assert 0.95 <= report["fitted_velocity_damping"] <= 1.05
    metrics = report["metrics"]["1-2d"]
    assert metrics["constant_velocity"]["count"] == 7
    assert metrics["constant_velocity"]["mean_error_km"] < 1e-6
    assert metrics["persistence"]["mean_error_km"] > 5
