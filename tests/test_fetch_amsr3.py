from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr

from ml.fetch_amsr3 import CATALOG_NS, DATASET_ID, discover_latest, select_files, validate_file


def catalog(entries):
    root = ET.Element(f"{CATALOG_NS}catalog")
    for name, path in entries:
        ET.SubElement(root, f"{CATALOG_NS}dataset", {"name": name, "urlPath": path})
    return ET.tostring(root)


def test_catalog_selects_only_standard_southern_amsr3_files():
    good = "ice_conc_sh_polstere-100_amsr3_202610111200.nc"
    base = "osisaf/met.no/ice/amsr2_conc/2026/10/"
    result = select_files(catalog([
        (good, base + good),
        ("ice_conc_sh_polstere-100_amsr3-tud_202610111200.nc", base + "tud.nc"),
        ("ice_conc_nh_polstere-100_amsr3_202610111200.nc", base + "nh.nc"),
        ("ice_conc_sh_polstere-100_amsr2_202610111200.nc", base + "amsr2.nc"),
    ]))
    assert [item["file"] for item in result] == [good]
    assert result[0]["timestamp"] == "2026-10-11T12:00:00+00:00"


def test_catalog_rejects_path_traversal():
    name = "ice_conc_sh_polstere-100_amsr3_202610111200.nc"
    with pytest.raises(ValueError, match="Unsafe"):
        select_files(catalog([(name, "../" + name)]))


def test_discovery_returns_latest_requested_files():
    class Response:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            return None
    class Session:
        def get(self, url, timeout):
            names = [f"ice_conc_sh_polstere-100_amsr3_202610{day:02d}1200.nc" for day in (9, 10, 11)]
            base = "osisaf/met.no/ice/amsr2_conc/2026/10/"
            return Response(catalog([(name, base + name) for name in names]))
    result = discover_latest(Session(), datetime(2026, 10, 12, tzinfo=timezone.utc), 2)
    assert [item["timestamp"] for item in result] == [
        "2026-10-10T12:00:00+00:00", "2026-10-11T12:00:00+00:00"]


def test_validation_preserves_quality_and_uncertainty(tmp_path):
    path = tmp_path / "amsr3.nc"
    dims = ("time", "yc", "xc")
    ds = xr.Dataset(
        {
            "Polar_Stereographic_Grid": xr.DataArray(0, attrs={"grid_mapping_name": "polar_stereographic"}),
            "ice_conc": (dims, np.array([[[0.0, 65.0], [100.0, np.nan]]], dtype=np.float32)),
            "status_flag": (dims, np.array([[[0.0, 0.0], [2.0, np.nan]]], dtype=np.float32)),
            "total_uncertainty": (dims, np.array([[[1.0, 5.0], [8.0, np.nan]]], dtype=np.float32)),
            "smearing_uncertainty": (dims, np.ones((1, 2, 2), dtype=np.float32)),
            "algorithm_uncertainty": (dims, np.ones((1, 2, 2), dtype=np.float32)),
        },
        coords={"time": [np.datetime64("2026-10-11T12")], "yc": [10.0, 0.0], "xc": [0.0, 10.0],
                "lat": (("yc", "xc"), np.full((2, 2), -66.0, dtype=np.float32)),
                "lon": (("yc", "xc"), np.full((2, 2), -60.0, dtype=np.float32))},
        attrs={"product_id": "OSI-408-g", "instrument": "AMSR3", "platform": "GOSAT-GW",
               "product_status": "operational", "product_version": "4.1", "software_version": "3.0.35"},
    )
    ds.to_netcdf(path)
    result = validate_file(path, "2026-10-11T12:00:00+00:00")
    assert result["shape"] == [2, 2]
    assert result["valid_sic_cells"] == 3
    assert result["nominal_cells"] == 2
    assert result["nominal_total_uncertainty_percent"]["median"] == 3.0
    assert DATASET_ID == "EUMETSAT-OSI-408-g"
