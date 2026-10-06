import hashlib
import json

import numpy as np
import xarray as xr

from ml.identity import model_artifact_sha256, model_fingerprint
from ml.train_sic import grids, inventory, main


def test_documented_cdr_quality_bits_mask_no_input_invalid_mask_and_temporal_fill(tmp_path):
    data = np.full((3, 2, 3), 0.5, dtype=np.float32)
    qa = np.zeros((3, 2, 3), dtype=np.uint8)
    qa[0] = np.array([[0, 8, 16], [64, 1, 32]], dtype=np.uint8)
    path = tmp_path / "cdr.nc"
    xr.Dataset({
        "cdr_seaice_conc": (("time", "y", "x"), data),
        "cdr_seaice_conc_qa_flag": (("time", "y", "x"), qa),
    }, coords={"time": np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-01-04"))}).to_netcdf(path)

    daily = list(grids(inventory([str(path)])))
    first = daily[0][1]
    assert np.isnan(first[0, 1])  # no input brightness-temperature data
    assert np.isnan(first[0, 2])  # invalid ice/ocean mask
    assert np.isnan(first[1, 0])  # temporal interpolation may use future observations
    assert first[1, 1] == 0.5  # accepted weather-filter flag
    assert first[1, 2] == 0.5  # contemporaneous spatial interpolation


def test_training_artifact_records_split_source_and_runtime_reproducibility(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    days = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-05-06"))
    base = np.linspace(0.0, 1.0, 144, dtype=np.float32).reshape(12, 12)
    fields = np.stack([np.clip(base + 0.01 * np.sin(index / 7.0), 0.0, 1.0) for index in range(len(days))])
    source = dataset_dir / "sic_pss25_20100101-20100505_v06r00.nc"
    xr.Dataset({"cdr_seaice_conc": (("time", "y", "x"), fields)},
               coords={"time": days}).to_netcdf(source)
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "dataset_id": "G02202", "complete": True,
        "files": [{"file": source.name, "bytes": source.stat().st_size,
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }), encoding="utf-8")
    output = tmp_path / "model.json"
    monkeypatch.setattr("sys.argv", ["train_sic.py", str(dataset_dir), "--validation-fraction", "0.3",
                                    "--output", str(output)])
    assert main() == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["model_fingerprint"] == model_fingerprint(artifact)
    assert artifact["artifact_sha256"] == model_artifact_sha256(artifact)
    assert artifact["training_configuration"]["validation_fraction_requested"] == 0.3
    assert artifact["training_configuration"]["chronological_split_cutoff_date"] == artifact["training"]["validation_start"]
    assert artifact["reproducibility"]["source_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert len(artifact["reproducibility"]["trainer_sha256"]) == 64
    assert artifact["reproducibility"]["runtime"]["numpy"] == np.__version__


def test_combined_verified_archives_use_explicit_annual_holdout(tmp_path, monkeypatch):
    directories = [tmp_path / "early", tmp_path / "late"]
    for directory, start, end in ((directories[0], "2010-01-01", "2010-04-01"),
                                  (directories[1], "2010-04-01", "2010-07-01")):
        directory.mkdir()
        days = np.arange(np.datetime64(start), np.datetime64(end))
        fields = np.stack([np.full((12, 12), 0.2 + index / 1000, dtype=np.float32)
                           for index in range(len(days))])
        source = directory / f"sic_{start}_{end}.nc"
        dataset = xr.Dataset({"cdr_seaice_conc": (("time", "y", "x"), fields),
                              "crs": ((), 0, {"crs_wkt": "test-identical-crs"})},
                             coords={"time": days, "x": np.arange(12), "y": np.arange(12)})
        dataset.to_netcdf(source)
        (directory / "manifest.json").write_text(json.dumps({
            "dataset_id": "G02202", "hemisphere": "Southern", "complete": True,
            "files": [{"file": source.name, "bytes": source.stat().st_size,
                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        }), encoding="utf-8")
    output = tmp_path / "combined.json"
    monkeypatch.setattr("sys.argv", ["train_sic.py", str(directories[0]),
                                    "--additional-data-dir", str(directories[1]),
                                    "--validation-start", "2010-06-01", "--output", str(output)])
    assert main() == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["training"]["validation_start"] == "2010-06-01"
    assert artifact["training"]["files"] == 2
    assert len(artifact["reproducibility"]["source_manifest_sha256s"]) == 2
    assert artifact["artifact_sha256"] == model_artifact_sha256(artifact)
