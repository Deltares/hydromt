# -*- coding: utf-8 -*-
from os.path import join, isfile, basename
import os
import pdb
import dask
import xarray as xr
from dask.diagnostics import ProgressBar
import shutil
import pandas as pd
import numpy as np
import glob
import time

# global vars
dt_era5t = pd.to_timedelta(95, unit="d")
era5_variables = {
    "ssrd": "surface_solar_radiation_downwards",
    "t2m": "2m_temperature",
    "tisr": "toa_incident_solar_radiation",
    "tp": "total_precipitation",
    "msl": "mean_sea_level_pressure",
    "cape": "convective_available_potential_energy",
    "tcwv": "total_column_water_vapour",
    "shww": "significant_height_of_wind_waves",
    "ro": "runoff",
    "pev": "potential_evaporation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "d2m": "2m_dewpoint_temperature",
    "ssr": "surface_net_solar_radiation",
    "tcc": "total_cloud_cover",
}
daily_attrs = {
    "tp": {
        "units": "mm d**-1",  # NOTE unit conversion m to mm
        "long_name": "Total precipitation",
    },
    "tmin": {"units": "K", "long_name": "2 meter mininum temperature"},
    "tmax": {"units": "K", "long_name": "2 meter maximum temperature"},
    "t2m": {"units": "K", "long_name": "2 meter mean temperature"},
    "d2m": {"units": "K", "long_name": "2 meter dewpoint temperature"},
    "msl": {
        "units": "Pa",
        "long_name": "Mean sea level pressure",
        "standard_name": "air_pressure_at_mean_sea_level",
    },
    "tisr": {"units": "J m**-2", "long_name": "TOA incident solar radiation"},
    "ssrd": {
        "units": "J m**-2",
        "long_name": "Surface solar radiation downwards",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
    },
    "tcwv": {
        "units": "kg m**-2",
        "long_name": "Total column water vapour",
    },
    "u10": {
        "units": "m s**-1",
        "long_name": "Neutral wind at 10 m u-component",
        "standard_name": "10m_u_component_of_wind",
    },
    "v10": {
        "units": "m s**-1",
        "long_name": "Neutral wind at 10 m v-component",
        "standard_name": "10m_v_component_of_wind",
    },
    "ssr": {
        "units": "J m**-2",
        "long_name": "Surface net solar radiation",
        "standard_name": "surface_net_downward_shortwave_flux",
    },
    "tcc": {
        "units": "-",
        "long_name": "Total cloud cover",
        "standard_name": "cloud_area_fraction",
    },
}


# donwload single year!
def download_era5_year(
    fn_out: str, variable: str, year: int, months: list = None, days: list = None
) -> None:
    """Download single ERA5 variable for single year from CDS.

    Parameters
    ----------
    fn_out : str
        output path
    variable : str
        CDS variable name
    """
    import cdsapi

    if not isfile(fn_out):
        # Set default months, days and hours options
        _months = [f"{m:02d}" for m in range(1, 13)]
        _days = [f"{d:02d}" for d in range(1, 32)]
        _hrs = [f"{h:02d}:00" for h in range(24)]

        if months is None:
            months = _months
        if days is None:
            days = _days

        dataset = "reanalysis-era5-single-levels"
        if int(year) < 1979:
            # raise ValueError("Years before 1979 not supported, please start download at 1979")
            dataset = f"{dataset}-complete-preliminary-back-extension"

        # Download ERA5 data
        c = cdsapi.Client()
        c.retrieve(
            dataset,
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": variable,
                "year": [year],
                "month": months,
                "day": _days,
                "time": _hrs,
            },
            fn_out,
        )


def flatten_era5_temp(
    fn,
    nodata=-9999,
    chunks={"time": 30, "latitude": 250, "longitude": 480},
    dask_kwargs={},
) -> None:
    """Flatten ERA5 output to remove expver dimension with ERA5T data."""
    with xr.open_dataset(fn, chunks=chunks) as ds:
        era5t = "expver" in ds.coords

    if era5t:
        print(f"flattening {basename(fn)}")
        fn_out = fn.replace(".nc", "_flat.nc")
        with xr.open_dataset(fn, chunks=chunks) as ds:
            # expver5 (ERA5T) contains NaNs when out of range,
            # while expver1 might contain large odd values
            # https://confluence.ecmwf.int/display/CUSF/ERA5+CDS+requests+which+return+a+mixture+of+ERA5+and+ERA5T+data
            ds_out = ds.sel(expver=5).combine_first(ds.sel(expver=1)).fillna(nodata)
            chunksizes = tuple([s[0] for s in ds_out.chunks.values()])
            e0 = {
                "zlib": True,
                "dtype": "float32",
                "_FillValue": nodata,
                "chunksizes": chunksizes,
            }
            encoding = {var: e0 for var in ds_out.data_vars}
            obj = ds_out.to_netcdf(fn_out, encoding=encoding, mode="w", compute=False)
            with ProgressBar():
                obj.compute(**dask_kwargs)
        if isfile(fn_out):
            # replace original file
            os.unlink(fn)
            shutil.move(fn_out, fn)


# resample variable for single year!
def resample_year(
    year: int,
    ddir: str,
    outdir: str,
    var: str,
    decimals: int = None,
    nodata=-9999,
    chunks: dict = {"time": 30, "latitude": 250, "longitude": 480},
    dask_kwargs: dict = {},
) -> None:
    """Resample hourly variables to daily timestep.
    The data is saved with the time labels at the end of the timestep.
    By default the data is aggregated using the mean.

    Exceptions:
    - the unit of tp is changed from m/hr to mm/day
    - from t2m, the mean, min and max daily values are saved

    Parameters
    ----------
    year : int
        year
    ddir : str
        Root of hourly nc files with path format {ddir}/{var}/era5_{var}_{year}_hourly.nc
    outdir : str
        Temporary output folder
    var : str
        input variable
    decimals : int, optional
        _description_, by default None
    nodata : int, optional
        _description_, by default -9999
    chunks : _type_, optional
        _description_, by default {'time':30, 'latitude':250, 'longitude':480}
    dask_kwargs : dict, optional
        _description_, by default {}
    """

    # nc out settings
    chunksizes = tuple([s for s in chunks.values()])
    e0 = {
        "zlib": True,
        "dtype": "float32",
        "_FillValue": nodata,
        "chunksizes": chunksizes,
    }

    fns = []
    # read hourly data for year and year-1!
    for year in [year - 1, year]:
        fn = join(ddir, var, f"era5_{var:s}_{year:d}_hourly.nc")
        if isfile(fn):
            fns.append(fn)
    kwargs = dict(
        compat="no_conflicts",
        parallel=True,
        decode_times=True,
    )
    ds = xr.open_mfdataset(fns, chunks=chunks, **kwargs)
    assert "expver" not in ds.coords
    ds = ds.sel(time=slice(f"{year-1}-12-31 01:00", f"{year}-12-31 00:00"))
    if ds["time"][-1].dt.hour != 0:  # clip incomplete days
        month, day = ds["time"][-1].dt.month.item(), ds["time"][-1].dt.day.item()
        ds = ds.sel(time=slice(f"{year-1}-12-31 01:00", f"{year}-{month}-{day} 00:00"))

    # resample to daily freq
    kwargs = dict(time="1D", label="right", closed="right")
    dvars = {}
    if "tp" in ds:
        # NOTE unit conversion m to mm
        dvars["tp"] = ds["tp"].resample(**kwargs).sum("time") * 1000
        dvars["tp"].attrs.update(daily_attrs.get("tp", {}))
    if "t2m" in ds:
        dvars["tmin"] = (
            ds[["t2m"]].resample(**kwargs).min("time").rename({"t2m": "tmin"})
        )
        dvars["tmin"].attrs.update(daily_attrs.get("tmin", {}))
        dvars["tmax"] = (
            ds[["t2m"]].resample(**kwargs).max("time").rename({"t2m": "tmax"})
        )
        dvars["tmax"].attrs.update(daily_attrs.get("tmax", {}))
    for var in ds.data_vars.keys():
        if var in dvars:
            continue
        dvars[var] = ds[var].resample(**kwargs).mean("time")
        dvars[var].attrs.update(daily_attrs.get(var, {}))
    ds_out = xr.merge(dvars.values()).chunk(chunks).fillna(nodata)
    if decimals:
        ds_out = ds_out.round(decimals=decimals)
    # assert np.all(ds_out['time'].dt.year == year)

    # write
    fns = []
    for var in ds_out.data_vars.keys():
        print(f"{var}: {year} - {ds_out.time.size} days")
        fn_out = join(outdir, f"era5_{var:s}_{year:d}_daily.nc")
        fns.append(fn_out)
        if isfile(fn_out):
            continue
        obj = ds_out[[var]].to_netcdf(
            fn_out, encoding={var: e0}, mode="w", compute=False
        )
        with ProgressBar():
            obj.compute(**dask_kwargs)

    return fns


def move_replace(src: str, dst: str, timeout: int = 300) -> None:
    """try replacing old file which might be locked"""
    if not isfile(src):
        return
    if not os.path.isdir(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    while isfile(dst):
        try:
            os.unlink(dst)
        except:
            print(f"FAILED deleting {dst} (retry in {timeout} sec)")
            time.sleep(timeout)
    os.rename(src, dst)


def get_last_timestep_nc(fns: list) -> pd.Timedelta:
    if len(glob.glob(fns)) == 0:
        raise ValueError(f"Files not found {fns}")
    with xr.open_mfdataset(fns, chunks="auto") as ds:
        t0 = pd.to_datetime(ds["time"][-1].values)
    return t0


def append_zarr(
    store: str,
    ds: xr.Dataset,
    chunks_out: dict,
    append_dim: str = "time",
) -> None:
    """Write/append/overwrite data to a zarr store.

    Parameters
    ----------
    store : str
        path to zarr store
    ds : xarray.Dataset
        Dataset to append
    chunks_out : dict
        zarr chunks (only used for new store)
    append_dim : str, optional
        dimension along wicht to append, by default 'time'
    """
    import zarr
    from cftime import date2num

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    encoding = {}
    ds.encoding = {}
    for v in ds.data_vars:
        ds[v].encoding = {}
        if compressor is not None:
            encoding[v] = {"compressor": compressor}

    # pre-allocate new zarr store
    if not os.path.isdir(store):
        ds.chunk(chunks_out).to_zarr(
            store,
            compute=False,
            consolidated=False,
            encoding=encoding,
        )

    ds0 = xr.open_zarr(store, consolidated=False)
    for v in ds.data_vars:
        # pre-allocate new zarr variables
        if v not in ds0:
            _da0 = ds0[list(ds0.data_vars.keys())[0]]
            _ds = xr.DataArray(
                name=v,
                data=dask.array.empty_like(_da0),
                coords=_da0.coords,
                dims=_da0.dims,
                attrs=ds[v].attrs,
            ).to_dataset()
            _ds.chunk(chunks_out).to_zarr(
                store,
                mode="a",
                compute=False,
                consolidated=False,
                encoding={v: encoding.get(v, {})},
            )

    # make sure ds does not start before ds0
    ds = ds.sel(
        {append_dim: slice(ds0[append_dim].values[0], ds[append_dim].values[-1])}
    )
    # get start- and end- time of both datsets and calculate offset
    zgroup = zarr.open_group(store)
    t0 = zgroup[append_dim][0]
    dt = int(np.diff(zgroup[append_dim][:2])[0])
    time_encoding = ds0[append_dim].encoding
    _t0 = pd.Timestamp(ds[append_dim].values[0]).to_pydatetime()
    _t1 = pd.Timestamp(ds[append_dim].values[-1]).to_pydatetime()
    _t0, _t1 = date2num([_t0, _t1], time_encoding["units"], time_encoding["calendar"])
    offset = int((_t0 - t0) / dt)

    # check if time dimension needs to be extended
    if _t1 > zgroup[append_dim][-1]:
        dates = np.arange(t0, _t1 + 1, dt, dtype=zgroup[append_dim].dtype)
        zgroup[append_dim].resize(dates.size)
        zgroup[append_dim][:] = dates
        # expand time on all variables
        for v in ds0.data_vars:
            if append_dim in ds0[v].dims:
                assert ds0[v].get_axis_num(append_dim) == 0
                arr = zgroup[v]
                shape = list(arr.shape)
                shape[0] = dates.size
                zgroup[v].resize(shape)

    # align dims and coords
    # re-read ds as new variables might have been appended
    ds0 = xr.open_zarr(store, consolidated=False)
    for v in ds.data_vars:
        ds[v] = ds[v].transpose(*ds0[v].dims)
        for idim, dim in enumerate(ds.dims):
            if dim == append_dim:
                continue
            assert ds[v][dim].shape == ds0[v][dim].shape
            # TODO: raise error instead of reindex
            if not np.allclose(ds0[v][dim], ds[v][dim]):
                print(f"reindex {v}")
                print(ds0[v][dim], ds[v][dim])
                ds = ds.reindex({dim: ds0[v][dim]})

    # write output using to_zarr with region argument
    # drop dims (already written)
    region_out = {append_dim: slice(offset, offset + ds[append_dim].size)}
    ds.drop(ds0[v].dims).to_zarr(
        store,
        region=region_out,
        compute=True,
        safe_chunks=True,
        consolidated=False,
    )
    # consolidate metadata
    zarr.convenience.consolidate_metadata(store)


def update_hourly_nc(
    outdir: str,
    ddir: str,
    variables: list,
    start_year: int = None,
    end_year: int = None,
    dt_era5t: pd.Timedelta = dt_era5t,
    move_to_ddir: bool = False,
    dask_kwargs: dict = {},
) -> None:
    """Update the hourly by downloading the latest data from the CDS

    Parameters
    ----------
    outdir : str
        temporary output folder
    ddir : str
        Root of hourly nc files with path format {ddir}/{var}/era5_{var}_{year}_hourly.nc
    variables : list
        list of variable names to update
    start_year : int, optional
        start year, by default None and read from last timestep of hourly nc files minus dt_era5t
    end_year : int, optional
        end year, by default None and based on todays' date
    dt_era5t : pd.Timedelta, optional
        time period of ERA5T data which is overwritten by new data, by default dt_era5t
    move_to_ddir : bool, optional
        Move files to root of hourly nc, by default False
    """
    t1 = pd.to_datetime("today").to_numpy()
    if end_year:
        t1 = pd.to_datetime(f"{end_year}0101")
    if start_year:
        t0 = pd.to_datetime(f"{start_year}0101")

    # download
    jobs = []
    for var in variables:
        if start_year is None:  # get last date of hourly file
            fns = join(ddir, var, f"era5_{var}_*_hourly.nc")
            t0 = get_last_timestep_nc(fns)
            if (t0 > t1) or ((t1 - t0) < pd.to_timedelta(7, unit="d")):
                continue  ## skip if last date within 7 days
            t0 = t0 - dt_era5t
        years = np.unique(pd.date_range(t0, t1, freq="1d").year).tolist()

        # download complete years
        for year in years:
            fn_out = join(outdir, f"era5_{var}_{year}_hourly.nc")
            job = dask.delayed(download_era5_year)(
                fn_out, era5_variables[var], year=year
            )
            jobs.append(job)
    dask.compute(*jobs, **dask_kwargs)

    # flatten if the data contains both ERA5 & ERA5T data
    for var in variables:
        fns = glob.glob(join(outdir, f"era5_{var}_*_hourly.nc"))
        for fn in fns:
            flatten_era5_temp(fn, dask_kwargs=dask_kwargs)
            # TODO make plot at random point

    # write files from tmpdir to ddir
    if move_to_ddir:
        for var in variables:
            fns = glob.glob(join(outdir, f"era5_{var}_*_hourly.nc"))
            for fn in fns:
                move_replace(fn, join(ddir, var, basename(fn)))


def update_daily_nc(
    outdir: str,
    ddir_hour: str,
    ddir_day: str,
    variables: list,
    start_year: int = None,
    end_year: int = None,
    dt_era5t: pd.Timedelta = dt_era5t,
    move_to_ddir: bool = False,
    dask_kwargs: dict = {},
) -> None:
    """Update the daily nc files based on hourly files

    Parameters
    ----------
    outdir : str
        temporary output folder
    ddir_hour : str
        Root of hourly nc files with path format {ddir}/{var}/era5_{var}_{year}_hourly.nc
    ddir_day : str
        Root of daily nc files with path format {ddir}/{var}/era5_{var}_{year}_daily.nc
    variables : list
        list of variable names to update
    start_year : int, optional
        start year, by default None and read from last timestep of daily nc files minus dt_era5t
    end_year : int, optional
        end year, by default None and read from last timestep of hourly nc files
    dt_era5t : pd.Timedelta, optional
        time period of ERA5T data which is overwritten by new data, by default dt_era5t
    move_to_ddir : bool, optional
        Move files to root of daily nc, by default False
    """
    if end_year:
        t1 = pd.to_datetime(f"{end_year}0101")
    if start_year:
        t0 = pd.to_datetime(f"{start_year}0101")

    fn_lst = []
    for var in variables:
        if var not in daily_attrs:
            print(f'no attributes found set for daily "{var}" - skipping')
            continue
        if end_year is None:  # get last date from hourly nc files
            fns = join(ddir_hour, var, f"era5_{var}_*_hourly.nc")
            t1 = get_last_timestep_nc(fns)
        if start_year is None:  # get last date of daily nc files
            fns = join(ddir_day, var, f"era5_{var}_*_daily.nc")
            t0 = get_last_timestep_nc(fns)
            if (t0 > t1) or ((t1 - t0) < pd.to_timedelta(7, unit="d")):
                continue  ## skip if last date within 7 days
            t0 = t0 - dt_era5t
        years = np.unique(pd.date_range(t0, t1, freq="1d").year).tolist()

        # resample to daily values
        for year in years:
            fns0 = resample_year(
                year, ddir=ddir_hour, outdir=outdir, var=var, dask_kwargs=dask_kwargs
            )
            fn_lst.extend(fns0)

    # write files from outdir to ddir
    if move_to_ddir:
        for fn in fn_lst:
            var = basename(fn).split("_")[1]
            move_replace(fn, join(ddir_day, var, basename(fn)))


def update_zarr(
    fn_zarr: str,
    ddir: str,
    variables: list,
    start_date: str = None,
    end_date: str = None,
    dt_era5t: pd.Timedelta = dt_era5t,
    chunks=None,
    **kwargs,
) -> None:
    """Update zarr file based on nc files.

    Parameters
    ----------
    fn_zarr : str
        Zarr store path
    ddir : str
        Root of nc files with path format {ddir}/{var}/eray_{var}_*.nc
    variables : list
        list of variable names to update
    start_date : str, optional
        start date in YYYYMMDD format, by default None and read from last timestep of zarr files minus dt_era5t
    end_date : str, optional
        end date in YYYYMMDD format, by default None and read from last timestep of the nc files
    dt_era5t : pd.Timedelta, optional
        time period of ERA5T data which is overwritten by new data, by default dt_era5t
    """

    print(f"writing to {fn_zarr}")
    for var in variables:

        # get start & end dates from files
        fns = join(ddir, var, f"era5_{var}_*.nc")
        if len(glob.glob(fns)) == 0:
            print(f'no nc files found for "{var}" - skipping')
        with xr.open_mfdataset(fns, chunks="auto", mode="r", autoclose=True) as ds_nc:
            t1 = pd.to_datetime(ds_nc["time"][-1].values)
            t0 = pd.to_datetime(ds_nc["time"][0].values)
            if end_date:
                t1 = pd.to_datetime(end_date)
            if start_date:
                t0 = pd.to_datetime(start_date)
            elif os.path.exists(fn_zarr):
                with xr.open_zarr(fn_zarr, consolidated=False) as ds_zarr:
                    if var in ds_zarr:
                        # check last date with valid values at single location
                        # TODO: this takes long, find alternative way with zarr library to find last date
                        da_zarr = (
                            ds_zarr[var].isel(latitude=0, longitude=0).dropna("time")
                        )
                        t1_zr = pd.to_datetime(da_zarr["time"][-1].values)
                        if t1 > t1_zr:
                            t0 = t1_zr - dt_era5t
                        else:  # already up to date: skip var
                            print(f"{var}: no new data - skipping")
                            continue

        # get zarr chunks to read and process data
        if chunks is None and os.path.exists(fn_zarr):
            with xr.open_zarr(fn_zarr, consolidated=False) as ds_zarr:
                var0 = var if var in ds_zarr else list(ds_zarr.data_vars.keys())[0]
                chunks = {
                    dim: ds_zarr[var0].chunks[i][0]
                    for i, dim in enumerate(ds_zarr[var0].dims)
                }

        # process year by year a.k.a. file by file
        years = np.unique(pd.date_range(t0, t1, freq="1d").year).tolist()
        for year in years:
            fn = glob.glob(join(ddir, var, f"era5_{var}_{year}_*.nc"))[0]
            with xr.open_dataset(fn, chunks=chunks, mode="r", autoclose=True) as ds_nc:
                ds_nc = ds_nc.sel(time=slice(t0, t1))
                if ds_nc["time"].size > 0:
                    t0_str, t1_str = (
                        ds_nc["time"].dt.strftime("%Y-%m-%d %H:%M").values[[0, -1]]
                    )
                    print(f"{var}: {t0_str} - {t1_str}")
                    # retry at seemingly random Permission Errors ..
                    while True:
                        try:
                            append_zarr(
                                fn_zarr,
                                ds_nc,
                                append_dim="time",
                                chunks_out=chunks,
                                **kwargs,
                            )
                            break
                        except PermissionError:
                            print(f"{var}: PermissionError - retry")


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    dask_kwargs = {"n_workers": 4, "processes": True}

    save_zarr = False
    outdir = join(r"p:\wflow_global\hydromt_staging\era5")

    # meteo variables
    root = r"p:\wflow_global\hydromt\meteo"
    ddir_hour = join(root, "era5")
    ddir_day = join(root, "era5_daily")
    zarr_hour = join(root, "era5.zarr")
    zarr_day = join(root, "era5_daily.zarr")
    variables_hour = [
        "msl",
        "ssrd",
        "tp",
        "t2m",
        "tisr",
        "tcwv",
        "u10",
        "v10",
        "d2m",
        "ssr",
        "tcc",
        "cape",
    ]
    # NOTE: CAPE excluded from daily values, not sure how it is used and thus how to aggregate
    variables_day = variables_hour[:-1] + ["tmin", "tmax"]

    # hydro / ocean
    # NOTE these are kept in different ddir
    # variables_hour, ddir_hour = ["shww"], r"p:\wflow_global\hydromt\ocean\era5"

    print(f"downloading ..")
    update_hourly_nc(
        outdir,
        ddir=ddir_hour,
        variables=variables_hour,
        dask_kwargs=dask_kwargs,
        move_to_ddir=True,
        start_year=1979,
        end_year=2021,
    )

    print("resampling to daily values ..")
    update_daily_nc(
        outdir,
        ddir_hour=ddir_hour,
        ddir_day=ddir_day,
        variables=variables_hour,
        dask_kwargs=dask_kwargs,
        move_to_ddir=True,
    )

    if save_zarr:
        print("updating hourly zarr..")
        update_zarr(
            fn_zarr=zarr_hour,
            ddir=ddir_hour,
            variables=variables_hour,
        )

        print("updating daily zarr..")
        update_zarr(
            fn_zarr=zarr_day,
            ddir=ddir_day,
            variables=variables_day,
        )
