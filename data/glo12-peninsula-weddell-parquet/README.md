# Peninsula–Weddell GLO12 Parquet snapshot

This folder contains four forecast files from the 11 November 2026 Copernicus Marine GLO12 bulletin, for lead days 1, 3, 5, and 7 (valid 12, 14, 16, and 18 November). Each file contains 153,517 grid cells inside the frozen Peninsula–Weddell bounding box, with latitude, longitude, and 14 float32 ocean/sea-ice fields.

The original four NetCDF files total 560,370,398 bytes (about 534.6 MiB). These regional Parquet files total 20,047,019 bytes (about 19.1 MiB), a 96.4% size reduction. Most of the reduction comes from keeping only the project’s regional box; Parquet Zstandard compression adds further savings. The crop excludes the rest of the global grid and must not be treated as a global replacement.

Longitudes are stored in degrees east in the range [-180, 180). NetCDF fill values decoded by xarray are represented as NaN. Variable units and source file checksums are stored in the Parquet metadata and in manifest.json.

These are gridded sea-ice and ocean forecast fields used as environmental context. They are not iceberg positions or trajectories. The BYU/NIC historical iceberg-track archive is a separate input and is not included here. Original NetCDF files remain necessary for the project’s exact as-issued verification workflow.

## Recreate the export

Fetch the original bulletin files and their checksum manifests into work/datasets/copernicus-glo12-forecast/, then install the optional dependency and export to a new empty destination:

~~~powershell
work\convlstm-env\Scripts\python.exe -m pip install -r requirements-parquet.txt
work\convlstm-env\Scripts\python.exe -m tools.export_glo12_parquet --run-date 2026-11-11 --output-dir data\glo12-peninsula-weddell-parquet-rebuilt
~~~

The exporter refuses to overwrite a non-empty destination and verifies the original NetCDF checksums before converting. Use a new destination folder for a different bulletin date.

## Source and attribution

Source: [Copernicus Marine Global Ocean Physics Analysis and Forecast](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description), daily dataset cmems_mod_glo_phy_anfc_0.083deg_P1D-m, version 202406, DOI [10.48670/moi-00016](https://doi.org/10.48670/moi-00016).

Redistribution attribution: “E.U. Copernicus Marine Service Information; Source: European Union, Copernicus Marine Service Information or Data (2026), ©Mercator Ocean.”
