import hashlib
import datetime as dt
import urllib.error

import numpy as np
import xarray as xr

from ml import fetch_g02202
from ml.fetch_polarwatch_viirs import download_to_temp, query_url, trim_netcdf


class Response:
    def __init__(self, body=b"", headers=None):
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


def test_fetch_resumes_partial_and_records_checksum(tmp_path, monkeypatch):
    content = b"antarctic-sic"
    filename = "sic_pss25_20100101-20101231_v06r00.nc"
    part = tmp_path / f"{filename}.part"
    part.write_bytes(content[:5])

    def fake_urlopen(request, timeout):
        if request.get_method() == "HEAD":
            return Response(headers={"Content-Length": str(len(content))})
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        return Response(content[start:end + 1], {"Content-Range": f"bytes {start}-{end}/{len(content)}"})

    monkeypatch.setattr(fetch_g02202.urllib.request, "urlopen", fake_urlopen)
    result = fetch_g02202.fetch(2010, tmp_path)
    assert (tmp_path / filename).read_bytes() == content
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert not part.exists()


def test_fetch_verifies_existing_complete_file_without_replacing(tmp_path, monkeypatch):
    content = b"verified-data"
    filename = "sic_pss25_20100101-20101231_v06r00.nc"
    target = tmp_path / filename
    target.write_bytes(content)

    def fake_urlopen(request, timeout):
        assert request.get_method() == "HEAD"
        return Response(headers={"Content-Length": str(len(content))})

    monkeypatch.setattr(fetch_g02202.urllib.request, "urlopen", fake_urlopen)
    result = fetch_g02202.fetch(2010, tmp_path)
    assert result["verified_existing_file"] is True
    assert target.read_bytes() == content


def test_polarwatch_subset_builds_georeferenced_time_and_region_query():
    url, selection = query_url(dt.date(2024, 6, 1), dt.date(2024, 8, 31), -60, -66, 100)
    assert "noaacwVIIRSnppiceconcSP06Daily.nc?IceConc" in url
    assert "?IceConc%5B%282024-06-01T00%3A00%3A00Z%29" in url
    assert selection["crs"] == "EPSG:3976"
    assert selection["projected_bounds_m"]["x_max"] > selection["projected_bounds_m"]["x_min"]


def test_polarwatch_chunk_trimming_removes_neighbor_dates(tmp_path):
    source, target = tmp_path / "raw.nc", tmp_path / "trimmed.nc"
    dates = np.array(["2024-05-28T07:00", "2024-06-01T12:00", "2024-06-02T12:00", "2024-06-03T12:00"], dtype="datetime64[m]")
    xr.Dataset({"IceConc": (("time", "rows", "cols"), np.ones((4, 2, 2), dtype=np.float32))},
               coords={"time": dates, "rows": [1, 0], "cols": [0, 1]}).to_netcdf(source)
    summary = trim_netcdf(source, target, dt.date(2024, 6, 1), dt.date(2024, 6, 3), True)
    with xr.open_dataset(target) as dataset:
        assert dataset.sizes["time"] == 3
        assert str(dataset.time.values[0])[:10] == "2024-06-01"
    assert summary["time_count"] == 3


def test_polarwatch_query_supports_only_allowlisted_south_polar_viirs_platforms():
    start, end = dt.date(2026, 9, 1), dt.date(2026, 9, 14)
    url, selection = query_url(start, end, -60, -66, 100,
                               dataset_id="noaacwVIIRSn21iceconcSP06Daily")
    assert "noaacwVIIRSn21iceconcSP06Daily.nc?IceConc" in url
    assert selection["crs"] == "EPSG:3976"

    noaa20_url, _ = query_url(start, end, -60, -66, 100,
                              dataset_id="noaacwVIIRSn20iceconcSP06Daily")
    assert "noaacwVIIRSn20iceconcSP06Daily.nc?IceConc" in noaa20_url

    import pytest
    with pytest.raises(ValueError, match="supported NOAA Antarctic VIIRS"):
        query_url(start, end, -60, -66, 100, dataset_id="https://attacker.invalid/dataset")


def test_polarwatch_resume_accepts_same_requested_noaa21_dataset(tmp_path, monkeypatch):
    import json
    import pytest
    from ml import fetch_polarwatch_viirs

    selection = query_url(dt.date(2026, 9, 1), dt.date(2026, 9, 3), -60, -66, 100,
                          "noaacwVIIRSn21iceconcSP06Daily")[1]
    selection["chunk_days"] = 14
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset_id": "noaacwVIIRSn21iceconcSP06Daily", "selection": selection, "files": []
    }), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["fetch_polarwatch_viirs", str(tmp_path), "--start", "2026-09-01",
                                     "--end", "2026-09-03", "--dataset-id",
                                     "noaacwVIIRSn21iceconcSP06Daily"])
    def reached_download(*_args):
        raise RuntimeError("matching manifest accepted")

    monkeypatch.setattr(fetch_polarwatch_viirs, "download_to_temp", reached_download)
    # No network fixture is needed: reaching the request path proves the matching manifest was accepted.
    with pytest.raises(RuntimeError, match="matching manifest accepted"):
        fetch_polarwatch_viirs.main()


def test_polarwatch_fetch_retries_transient_proxy_errors(tmp_path, monkeypatch):
    body = b"CDF\x05" + bytes(2048)
    calls = 0

    class StreamResponse(Response):
        status = 200

        def read(self, size=-1):
            data, self.body = self.body[:size], self.body[size:]
            return data

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("https://example.test", 502, "Proxy Error", {}, None)
        return StreamResponse(body)

    monkeypatch.setattr("ml.fetch_polarwatch_viirs.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("ml.fetch_polarwatch_viirs.time.sleep", lambda _: None)
    target = tmp_path / "download.part"
    size, checksum = download_to_temp("https://example.test", str(target))
    assert calls == 2
    assert size == len(body)
    assert checksum == hashlib.sha256(body).hexdigest()
