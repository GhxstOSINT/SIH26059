import hashlib

import pytest

from ml.train_sic import verify_g02202_manifest


def make_manifest(path, complete=True):
    content = path.read_bytes()
    return {
        "dataset_id": "G02202",
        "complete": complete,
        "files": [{"file": path.name, "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}],
    }


def test_g02202_manifest_requires_complete_inventory_and_matching_hash(tmp_path):
    path = tmp_path / "sic_pss25_20100101-20101231_v06r00.nc"
    path.write_bytes(b"verified test source")
    manifest = make_manifest(path)
    verify_g02202_manifest(tmp_path, [path], manifest)

    path.write_bytes(b"altered source bytes")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_g02202_manifest(tmp_path, [path], manifest)
    with pytest.raises(ValueError, match="complete annual-file inventory"):
        verify_g02202_manifest(tmp_path, [path], make_manifest(path, complete=False))


def test_g02202_manifest_rejects_unlisted_files(tmp_path):
    listed = tmp_path / "sic_pss25_20100101-20101231_v06r00.nc"
    extra = tmp_path / "unlisted.nc"
    listed.write_bytes(b"a")
    extra.write_bytes(b"b")
    with pytest.raises(ValueError, match="do not match"):
        verify_g02202_manifest(tmp_path, [listed, extra], make_manifest(listed))
