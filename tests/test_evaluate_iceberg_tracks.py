import csv
from datetime import date, timedelta
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Geod

from ml.evaluate_iceberg_tracks import (
    evaluate,
    parse_doy,
    read_observed_tracks,
    read_published_tracks,
    restrict_tracks_to_published_coverage,
)


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


def test_published_coverage_excludes_positions_outside_byu_page_dates():
    tracks = {"a1": [
        (date(1977, 12, 31), -66.0, 30.0),
        (date(1978, 1, 1), -66.0, 31.0),
        (date(2025, 4, 22), -66.0, 32.0),
        (date(2025, 4, 23), -66.0, 33.0),
    ], "b2": [(date(1977, 1, 1), -67.0, 34.0)]}

    published, counts = restrict_tracks_to_published_coverage(tracks)

    assert published == {"a1": tracks["a1"][1:3]}
    assert counts == {
        "source_track_days": 5,
        "published_track_days": 2,
        "excluded_before_start_track_days": 2,
        "excluded_after_end_track_days": 1,
        "source_icebergs": 2,
        "published_icebergs": 1,
    }


def test_published_reader_filters_raw_zip_to_declared_interval(tmp_path):
    archive = tmp_path / "tracks.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tracks/a1.csv",
                        "date,ascat_1,ascat_2,ascat_3\n"
                        "1977365,-65,30,0\n"
                        "1978001,-65,31,0\n"
                        "2025112,-65,32,0\n"
                        "2025113,-65,33,0\n")

    tracks, counts = read_published_tracks(archive)

    assert [day for day, _, _ in tracks["a1"]] == [date(1978, 1, 1), date(2025, 4, 22)]
    assert counts["source_track_days"] == 4
    assert counts["track_days"] == 2
    assert counts["published_coverage_excluded_before_start"] == 1
    assert counts["published_coverage_excluded_after_end"] == 1


def test_parquet_reader_checks_published_coverage_metadata(tmp_path):
    source_counts = {
        "files": 2,
        "rows": 4,
        "observed_sensor_positions": 4,
        "track_days": 4,
        "icebergs": 2,
    }
    coverage = {
        "start": "1978-01-01",
        "end": "2025-04-22",
        "source_track_days": 4,
        "published_track_days": 2,
        "excluded_before_start_track_days": 1,
        "excluded_after_end_track_days": 1,
        "source_icebergs": 2,
        "published_icebergs": 1,
    }
    table = pa.table({
        "iceberg_id": pa.array(["a1", "a1"], type=pa.string()),
        "observation_date": pa.array([date(1978, 1, 1), date(2025, 4, 22)], type=pa.date32()),
        "latitude": pa.array([-65.0, -64.0], type=pa.float64()),
        "longitude": pa.array([30.0, 31.0], type=pa.float64()),
    }).replace_schema_metadata({
        b"dataset_id": json.dumps("BYU/NIC Antarctic Iceberg Tracking Database").encode(),
        b"dataset_version": json.dumps("8.0").encode(),
        b"source_archive_sha256": json.dumps("a" * 64).encode(),
        b"source_counts": json.dumps(source_counts).encode(),
        b"published_coverage": json.dumps(coverage).encode(),
    })
    parquet_path = tmp_path / "tracks.parquet"
    pq.write_table(table, parquet_path)

    tracks, counts = read_published_tracks(parquet_path)

    assert [point[0] for point in tracks["a1"]] == [date(1978, 1, 1), date(2025, 4, 22)]
    assert counts["track_days"] == 2
    assert counts["source_track_days"] == 4
    assert counts["published_coverage_excluded_before_start"] == 1
    assert counts["published_coverage_excluded_after_end"] == 1
    assert counts["source_archive_sha256"] == "a" * 64


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
