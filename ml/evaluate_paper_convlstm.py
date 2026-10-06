"""Transfer-check the published Antarctic ConvLSTM on independent G02202 inputs.

This is a research evaluation only. The paper's exact preprocessing is not
available in the bundled artifact, so this script makes its approximation
explicit and never changes the production forecast service.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "work/datasets/convlstm-paper/ConvLSTM"
MODEL_PATH = ROOT / "models/paper-convlstm/ConvLSTM.params"
TEST_PATH = PAPER_DIR / "testing_set.nc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_architecture():
    # Use the original model definition bundled with the attributed checkpoint.
    source = PAPER_DIR / "gitcode/convlstm.py"
    spec = importlib.util.spec_from_file_location("paper_convlstm", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    class PaperModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(module.ConvLSTM(
                input_dim=6, hidden_dim=[8, 8, 4, 2, 1], kernel_size=(5, 5),
                num_layers=5, batch_first=True, bias=True, return_all_layers=False,
            ))

        def forward(self, x):
            layers, _states = self.net(x)
            return layers[-1]

    net = PaperModel()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def daily_training_fields(data_dir: Path):
    """Read G02202 1989-2016, QA-mask, and average 4x4 to the 100-km grid."""
    paths = sorted(data_dir.rglob("*.nc"))
    if not paths:
        raise ValueError(f"No NetCDF files under {data_dir}")
    fields, dates = [], []
    for path in paths:
        with xr.open_dataset(path) as ds:
            sic = ds["cdr_seaice_conc"].astype("float32") / 100.0
            qa = ds["cdr_seaice_conc_qa_flag"]
            sic = sic.where((qa & np.uint8(8 | 16 | 64)) == 0)
            coarse = sic.coarsen(y=4, x=4, boundary="exact").mean(skipna=True)
            values = np.asarray(coarse.values, dtype=np.float32)
            fields.extend(values)
            dates.extend(dt.date.fromisoformat(str(t)[:10]) for t in ds.time.values)
    order = np.argsort(np.asarray(dates, dtype="datetime64[D]"))
    return np.stack(fields)[order], [dates[i] for i in order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-data", type=Path, default=ROOT / "work/datasets/g02202-v6/south")
    ap.add_argument("--origins", type=int, default=12, help="Evenly spaced origins in 2018-2022")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/paper-convlstm-transfer.json")
    args = ap.parse_args()
    if args.origins < 1 or args.origins > 50:
        ap.error("origins must be between 1 and 50")
    for p, expected in ((MODEL_PATH, "26ea046d9b097f56853a0f705eed933477b0c34b5f4f510c27183292da23c4db"),
                        (TEST_PATH, "8c3b42a1ddcf66a299c4ebdf3e171ba02a86a23ee64443a14773ba3c03f90579")):
        if sha256(p) != expected:
            raise ValueError(f"Artifact checksum mismatch: {p}")

    train, train_dates = daily_training_fields(args.training_data)
    with xr.open_dataset(TEST_PATH) as ds:
        test = np.asarray(ds["sic"].values, dtype=np.float32)
        dates = [dt.date.fromisoformat(str(t)[:10]) for t in ds.time.values]
        landmask = np.asarray(ds["landmask"].values)
        # In this released test file, finite landmask==1 denotes land; ocean is NaN.
        ocean = ~np.isfinite(landmask)
        area = np.asarray(ds["areacello"].values, dtype=np.float32)
        y100 = np.asarray(ds.y.values)
        x100 = np.asarray(ds.x.values)
    # Assert exact target grid registration rather than silently resizing.
    with xr.open_dataset(next(args.training_data.rglob("*.nc"))) as ds:
        y25, x25 = np.asarray(ds.y.values), np.asarray(ds.x.values)
    if not (np.allclose(y25.reshape(83, 4).mean(1), y100) and
            np.allclose(x25.reshape(79, 4).mean(1), x100)):
        raise ValueError("G02202 coarsened grid does not align exactly with paper test grid")

    # Paper states Gaussian normalization, but not all implementation details.
    # Fit one scalar mean/std per continuous feature on training ocean samples.
    ocean_train = np.broadcast_to(ocean, train.shape)
    sic_mean = float(np.nanmean(np.where(ocean_train, train, np.nan)))
    sic_std = float(np.nanstd(np.where(ocean_train, train, np.nan)))
    doy = np.array([(d.timetuple().tm_yday - 1) % 365 for d in train_dates])
    clim = np.full((365, *train.shape[1:]), np.nan, dtype=np.float32)
    for day in range(365):
        ix = doy == day
        if ix.any():
            clim[day] = np.nanmean(train[ix], axis=0)
    # Fill missing leap-climatology days from adjacent dates.
    for day in range(365):
        if not np.isfinite(clim[day]).any():
            clim[day] = (clim[(day - 1) % 365] + clim[(day + 1) % 365]) / 2
    std = np.full_like(clim, np.nan)
    for day in range(365):
        ix = doy == day
        std[day] = np.nanstd(train[ix], axis=0) if ix.any() else np.nan
    std = np.where(np.isfinite(std), std, 0).astype(np.float32)
    clim = np.where(np.isfinite(clim), clim, 0).astype(np.float32)
    # Match target 100-km grid fields and chronological history.
    train = np.where(np.isfinite(train), train, np.nan)
    model = load_architecture()
    candidates = [i for i, d in enumerate(dates) if dt.date(2018, 1, 1) <= d <= dt.date(2022, 12, 1) and i >= 90]
    origins = np.unique(np.linspace(0, len(candidates) - 1, args.origins).round().astype(int))
    rows = []
    for oi in origins:
        t = candidates[oi]
        history = test[t - 90:t]
        if not np.isfinite(history).all():
            # Preserve a clean ocean-only input; fill missing points by same-day climatology.
            pass
        feature_days = dates[t - 90:t]
        chans = []
        for field, mean, scale in ((history, sic_mean, sic_std),):
            chans.append((field - mean) / max(scale, 1e-6))
        cl = np.stack([clim[(d.timetuple().tm_yday - 1) % 365] for d in feature_days])
        sd = np.stack([std[(d.timetuple().tm_yday - 1) % 365] for d in feature_days])
        cmean, cscale = float(np.mean(cl[:, ocean])), float(np.std(cl[:, ocean]))
        smean, sscale = float(np.mean(sd[:, ocean])), float(np.std(sd[:, ocean]))
        phase = np.array([2 * math.pi * (d.timetuple().tm_yday - 1) / 365.2425 for d in feature_days])
        sin = np.broadcast_to(np.sin(phase)[:, None, None], history.shape)
        cos = np.broadcast_to(np.cos(phase)[:, None, None], history.shape)
        mask = np.broadcast_to(ocean.astype(np.float32), history.shape)
        x = np.stack((chans[0], (cl - cmean) / max(cscale, 1e-6),
                      (sd - smean) / max(sscale, 1e-6), sin, cos, mask), axis=1)
        # Missing SIC is filled from the fitted seasonal climatology after normalization.
        fallback = (cl - sic_mean) / max(sic_std, 1e-6)
        x[:, 0] = np.where(np.isfinite(x[:, 0]), x[:, 0], fallback)
        x[:, 5] = mask
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(x[None].astype(np.float32)))[0, -1, 0].numpy()
        pred = np.clip(pred_norm * sic_std + sic_mean, 0, 1)
        target = test[t]
        valid = ocean & np.isfinite(target) & np.isfinite(history[-1])
        w = np.where(valid, area, 0).astype(np.float64)
        denom = float(w.sum())
        if denom <= 0:
            continue
        truth = target[valid].astype(np.float64)
        predv = pred[valid].astype(np.float64)
        persist = history[-1][valid].astype(np.float64)
        rows.append({"origin": dates[t - 1].isoformat(), "target": dates[t].isoformat(),
                     "valid_cells": int(valid.sum()), "convlstm_area_weighted_rmse": float(np.sqrt(np.sum(w[valid] * (predv - truth) ** 2) / denom)),
                     "persistence_area_weighted_rmse": float(np.sqrt(np.sum(w[valid] * (persist - truth) ** 2) / denom))})
    report = {
        "status": "preprocessing_compatibility_probe_not_valid_model_skill_evidence",
        "model": "Dong et al. Antarctic ConvLSTM checkpoint (2024)",
        "paper_doi": "10.1016/j.ocemod.2024.102386",
        "checkpoint_sha256": sha256(MODEL_PATH), "test_dataset_sha256": sha256(TEST_PATH),
        "training_product": "NOAA/NSIDC G02202 v6 1989-2016; related passive-microwave source lineage to the paper's NSIDC-0051/0081 inputs, but a distinct processed product",
        "source_independence": "Not an independent satellite/source-family evaluation: the paper and this probe use related passive-microwave observation products. G02202 is not the original NSIDC-0051/0081 input sequence.",
        "test_product": "author-packaged held-out 100-km SIC testing_set.nc (2017-2022)",
        "preprocessing_caveat": "Approximate Gaussian normalization and G02202-derived climatology/std; paper's exact preprocessing, normalization parameters, original input files and inference pipeline are unavailable in this package.",
        "interpretation": "The approximate pipeline produces substantially worse values than persistence. Because preprocessing compatibility with the published checkpoint is unverified, these numbers are not a valid estimate of the checkpoint's model skill and must not be used for model selection.",
        "valid_model_skill_evidence": False,
        "origin_count": len(rows), "results": rows,
        "mean_model_rmse": float(np.mean([r["convlstm_area_weighted_rmse"] for r in rows])) if rows else None,
        "mean_persistence_rmse": float(np.mean([r["persistence_area_weighted_rmse"] for r in rows])) if rows else None,
        "decision": "Do not promote to application without broader validation and preprocessing reproduction."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "origin_count", "mean_model_rmse", "mean_persistence_rmse", "decision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
