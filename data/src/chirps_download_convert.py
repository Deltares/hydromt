"""Implementation of data downloading functionality."""

import glob
import gzip
import os
from os.path import isdir

import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from requests import HTTPError


def download_file(url, outdir=os.path.dirname(__file__)):
    """Download a file from the given URL and save it to the specified output directory.

    Args:
    ----
    url (str): The URL of the file to download.
    outdir (str): The output directory to save the downloaded file.

    Returns:
    -------
        None
    """
    basename = url.split("/")[-1]
    local_filename = os.path.join(outdir, basename)
    if os.path.isfile(local_filename):
        return
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        print(f"downloading {basename} ..")
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)


def download_africa_daily(outroot, start="2020-01-01", end="2021-03-31"):
    """Download daily CHIRPS data for Africa within the specified date range.

    Arguments:
    ---------
    outroot :
        The root directory to save the downloaded files.
    start :
        The start date in the format "YYYY-MM-DD" (default: "2020-01-01").
    end :
        The end date in the format "YYYY-MM-DD" (default: "2021-03-31").

    Returns:
    -------
        None
    """
    BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05"
    for date in pd.date_range(start=start, end=end, freq="d"):
        if date > pd.to_datetime("today").date():
            return
        date_str = date.strftime("%Y.%m.%d")
        year = date.year
        outdir = os.path.join(outroot, str(year))
        if not isdir(outdir):
            os.makedirs(outdir)
        basename = f"chirps-v2.0.{date_str}.tif.gz"
        url = f"{BASE_URL}/{year}/{basename}"
        try:
            download_file(url, outdir)
        except HTTPError:
            fn_out = os.path.join(outdir, basename)
            if os.path.isfile(fn_out):
                os.unlink(fn_out)
            try:  # without .gz
                download_file(url[:-3], outdir)
            except HTTPError as r:
                print(r)
                fn_out = os.path.join(outdir, basename[:-3])
                if os.path.isfile(fn_out):
                    os.unlink(fn_out)
                continue


def tifs_to_nc(folder, year):
    """Convert downloaded CHIRPS TIFF files for a specific year to NetCDF format.

    Arguments:
    ---------
    folder: str
        The root folder where the downloaded TIFF files are located.
    year: int
        The year to convert to NetCDF.

    """
    path = os.path.join(folder, str(year), f"chirps-v2.0.{year}*.tif*")
    nodata = -9999.0
    files = sorted(glob.glob(path))
    fn_out = os.path.join(folder, f"CHIRPS_rainfall_{year}.nc")
    if len(files) == 0 or os.path.isfile(fn_out):
        return

    for i, f in enumerate(files):
        infile = gzip.open(f) if f.endswith("gz") else f
        with rasterio.open(infile) as src:
            print(f"processing file : {os.path.basename(f)}")
            data = src.read(1)
            src_profile = src.profile
            out_transform = src_profile["transform"]

            nx, ny = data.shape[1], data.shape[0]
            x = (np.arange(nx) + 0.5) * out_transform[0] + out_transform[2]
            y = (np.arange(ny) + 0.5) * out_transform[4] + out_transform[5]

            time = "-".join(os.path.basename(f).split(".")[2:5])

            if i == 0:
                t = pd.to_datetime([time])
                ds = np.reshape(data, (1, ny, nx))
            else:
                t = pd.DatetimeIndex.append(t, pd.to_datetime([time]))
                ds = np.append(ds, np.reshape(data, (1, ny, nx)), axis=0)
        if f.endswith("gz"):
            infile.close()

    ds_img = xr.Dataset(
        {"precipitation": (["time", "lat", "lon"], ds)},
        coords={"lon": x, "lat": y, "time": t},
    )
    ds_img.precipitation.attrs["standard_name"] = "precipitation"
    ds_img.precipitation.attrs["units"] = "mm"
    ds_img.lon.attrs["standard_name"] = "longitude"
    ds_img.lon.attrs["units"] = "degrees _east"
    ds_img.lon.attrs["axis"] = "X"
    ds_img.lat.attrs["standard_name"] = "latitude"
    ds_img.lat.attrs["units"] = "degrees _north"
    ds_img.lat.attrs["axis"] = "Y"
    encoding = {
        "precipitation": {
            "_FillValue": nodata,
            "complevel": 4,
            "zlib": True,
            "chunksizes": [1, 320, 300],
        },
        "time": {"chunksizes": [90]},
    }
    print(f"writing to {fn_out} ..")
    ds_img.to_netcdf(fn_out, engine="netcdf4", encoding=encoding, unlimited_dims="time")


if __name__ == "__main__":
    outroot = r"p:\wflow_global\hydromt_staging\chirps"

    for year in range(2022, 2023):
        download_africa_daily(outroot, start=f"{year}-01-01", end=f"{year}-12-31")
        tifs_to_nc(outroot, year)
