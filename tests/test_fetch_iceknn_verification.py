import hashlib
import io
import json

import pytest

from ml import fetch_iceknn_verification as fetcher


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_fetch_writes_checksum_manifest_and_reuses_verified_file(tmp_path, monkeypatch):
    content = b"published model verification sample"
    monkeypatch.setattr(fetcher, "FILE_SIZE", len(content))
    monkeypatch.setattr(fetcher, "FILE_MD5", hashlib.md5(content, usedforsecurity=False).hexdigest())
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse(content)

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", fake_urlopen)
    first = fetcher.acquire(tmp_path)
    second = fetcher.acquire(tmp_path)

    assert first["sha256"] == hashlib.sha256(content).hexdigest()
    assert second["md5"] == first["md5"]
    assert len(calls) == 1
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["license"] == "CC-BY-4.0"


def test_fetch_refuses_corrupt_existing_archive_without_replacing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "FILE_SIZE", 5)
    monkeypatch.setattr(fetcher, "FILE_MD5", hashlib.md5(b"valid", usedforsecurity=False).hexdigest())
    target = tmp_path / fetcher.FILE_NAME
    target.write_bytes(b"bad!!")
    with pytest.raises(ValueError, match="Published verification failed"):
        fetcher.acquire(tmp_path)
    assert target.read_bytes() == b"bad!!"
