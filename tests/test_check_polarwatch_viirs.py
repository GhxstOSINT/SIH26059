import datetime as dt
import json

import pytest

from ml.check_polarwatch_viirs import (
    MAX_METADATA_BYTES,
    check_sources,
    fetch_latest_timestamp,
    parse_coverage_end,
    write_report_atomic,
)


def payload(timestamp="2026-09-21T12:00:00Z"):
    return {"table": {"rows": [["attribute", "NC_GLOBAL", "time_coverage_end", "String", timestamp]]}}


def test_parse_coverage_end_requires_a_timezone_and_valid_iso_timestamp():
    assert parse_coverage_end(payload()) == "2026-09-21T12:00:00Z"
    with pytest.raises(ValueError, match="timezone"):
        parse_coverage_end(payload("2026-09-21T12:00:00"))
    with pytest.raises(ValueError, match="valid ISO"):
        parse_coverage_end(payload("not-a-date"))
    with pytest.raises(ValueError, match="did not contain"):
        parse_coverage_end({"table": {"rows": []}})


def test_fetch_latest_metadata_is_allowlisted_and_size_bounded():
    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit):
            assert limit == MAX_METADATA_BYTES + 1
            return self.body

    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["agent"] = request.get_header("User-agent")
        return Response(json.dumps(payload()).encode())

    assert fetch_latest_timestamp("noaacwVIIRSn20iceconcSP06Daily", timeout=3, opener=opener) == "2026-09-21T12:00:00Z"
    assert "/info/noaacwVIIRSn20iceconcSP06Daily/index.json" in seen["url"]
    assert seen["timeout"] == 3
    assert "SouthernPassage" in seen["agent"]
    with pytest.raises(ValueError, match="allowlisted"):
        fetch_latest_timestamp("https://attacker.invalid/metadata", opener=opener)
    with pytest.raises(ValueError, match="2 MiB"):
        fetch_latest_timestamp("noaacwVIIRSn20iceconcSP06Daily",
                               opener=lambda *_a, **_kw: Response(b" " * (MAX_METADATA_BYTES + 1)))


def test_check_sources_keeps_per_platform_failures_and_classifies_staleness():
    now = dt.datetime(2026, 9, 30, tzinfo=dt.timezone.utc)

    def fetcher(dataset_id):
        if "n20" in dataset_id:
            raise TimeoutError("offline")
        return "2026-09-21T12:00:00Z"

    report = check_sources(now=now, fetcher=fetcher)
    assert report["status"] == "partial_or_invalid"
    products = {item["platform"]: item for item in report["products"]}
    assert products["S-NPP"]["status"] == "stale"
    assert products["NOAA-21"]["status"] == "stale"
    assert products["NOAA-20"]["status"] == "unavailable"
    assert products["NOAA-20"]["error"] == "TimeoutError"
    assert report["classification_basis"] == "demonstration_thresholds_not_approved_operational_limits"


def test_write_report_is_atomic_and_readable(tmp_path):
    path = tmp_path / "monitor" / "status.json"
    write_report_atomic(path, {"schema_version": 1, "status": "stale"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"schema_version": 1, "status": "stale"}
    assert list(path.parent.iterdir()) == [path]
