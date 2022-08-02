import pandas as pd
import xarray as xr
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


def precip(
    precip,
    da_like,
    clim=None,
    freq=None,
    reproj_method="nearest_index",
    resample_kwargs={},
    logger=logger,
):
    """Lazy reprojection of precipitation to model grid and resampling of time dimension to frequency.

    Parameters
    ----------
    precip: xarray.DataArray
        DataArray of precipitation forcing [mm]
    da_like: xarray.DataArray or Dataset
        DataArray of the target resolution and projection.
    clim: xarray.DataArray
        DataArray of monthly precipitation climatology. If provided this is used to
        to correct the precip downscaling.
    freq: str, Timedelta
        output frequency of time dimension
    reproj_method: str, optional
        Method for spatital reprojection of precip, by default 'nearest_index'
    resample_kwargs:
        Additional key-word arguments (e.g. label, closed) for time resampling method

    Returns
    --------
    p_out: xarray.DataArray (lazy)
        processed precipitation forcing
    """
    if precip.raster.dim0 != "time":
        raise ValueError(f'First precip dim should be "time", not {precip.raster.dim0}')
    # downscale precip (lazy); global min of zero
    p_out = np.fmax(precip.raster.reproject_like(da_like, method=reproj_method), 0)
    # correct precip based on high-res monthly climatology
    if clim is not None:
        # make sure first dim is month
        clim = clim.rename({clim.raster.dim0: "month"})
        if not clim["month"].size == 12:
            raise ValueError("Precip climatology does not contain 12 months.")
        # set missings to NaN
        clim = clim.raster.mask_nodata()
        # calculate downscaling multiplication factor
        clim_coarse = clim.raster.reproject_like(
            precip, method="average"
        ).raster.reproject_like(da_like, method="average")
        clim_fine = clim.raster.reproject_like(da_like, method="average")
        p_mult = xr.where(clim_coarse > 0, clim_fine / clim_coarse, 1.0).fillna(1.0)
        # multiply with monthly multiplication factor
        p_out = p_out.groupby("time.month") * p_mult
    # resample time
    p_out.name = "precip"
    p_out.attrs.update(unit="mm")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="sum", logger=logger)
        p_out = resample_time(p_out, freq, conserve_mass=True, **resample_kwargs)
    return p_out


# use dem_model (from staticmaps) and dem_forcing (meteo) in ini file
def temp(
    temp,
    dem_model,
    dem_forcing=None,
    lapse_correction=True,
    freq=None,
    reproj_method="nearest_index",
    lapse_rate=-0.0065,
    resample_kwargs={},
    logger=logger,
):
    """Lazy reprojection of temperature to model grid using lapse_rate for downscaling,
    and resampling of time dimension to frequency.

    Parameters
    ----------
    temp: xarray.DataArray
        DataArray of temperature forcing [°C]
    dem_model: xarray.DataArray
        DataArray of the target resolution and projection, contains elevation
        data
    dem_forcing: xarray.DataArray
        DataArray of elevation at forcing resolution. If provided this is used
        with `dem_model` to correct the temperature downscaling using a lapse rate
    lapse_correction : bool, optional
        If True, temperature is correctured based on lapse rate, by default True.
    freq: str, Timedelta
        output frequency of timedimension
    reproj_method: str, optional
        Method for spatital reprojection of precip, by default 'nearest_index'
    lapse_rate: float, optional
        lapse rate of temperature [C m-1] (default: -0.0065)
    resample_kwargs:
        Additional key-word arguments (e.g. label, closed) for time resampling method

    Returns
    --------
    t_out: xarray.DataArray (lazy)
        processed temperature forcing
    """
    if temp.raster.dim0 != "time":
        raise ValueError(f'First temp dim should be "time", not {temp.raster.dim0}')
    # apply lapse rate
    if lapse_correction:
        # if dem_forcing is not provided, reproject dem_model
        dem_model = dem_model.raster.mask_nodata()
        if dem_forcing is None:
            dem_forcing = dem_model.raster.reproject_like(temp, "average")
            if np.any(np.isnan(dem_forcing)):
                logger.warning(
                    "Temperature lapse rate could be computed for some (edge) cells. "
                    "Consider providing a full coverage dem_forcing."
                )
        else:
            # assume nans in dem_forcing occur above the ocean only -> set to zero
            dem_forcing = dem_forcing.raster.mask_nodata().fillna(0)
            dem_forcing = dem_forcing.raster.reproject_like(temp, "average")
        # compute temperature at quasi MSL
        t_add_sea_level = temp_correction(dem_forcing, lapse_rate=lapse_rate)
        temp = temp - t_add_sea_level
    # downscale temperature (lazy) and add zeros with mask to mask areas outside AOI
    t_out = temp.raster.reproject_like(dem_model, method=reproj_method)
    if lapse_correction:
        # correct temperature based on high-res DEM
        # calculate downscaling addition
        t_add_elevation = temp_correction(dem_model, lapse_rate=lapse_rate)
        t_out = t_out + t_add_elevation
    # resample time
    t_out.name = "temp"
    t_out.attrs.update(unit="degree C.")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean", logger=logger)
        t_out = resample_time(t_out, freq, conserve_mass=False, **resample_kwargs)
    return t_out


def press(
    press,
    dem_model,
    lapse_correction=True,
    freq=None,
    reproj_method="nearest_index",
    lapse_rate=-0.0065,
    resample_kwargs={},
    logger=logger,
):
    """Lazy reprojection of pressure to model grid and resampling of time dimension to frequency.

    Parameters
    ----------
    press: xarray.DataArray
        DataArray of pressure forcing [hPa]
    dem_model: xarray.DataArray
        DataArray of the target resolution and projection, contains elevation
        data
    lapse_correction: str, optional
       If True 'dem_model` is used to correct the pressure with the `lapse_rate`.
    freq: str, Timedelta
        output frequency of timedimension
    reproj_method: str, optional
        Method for spatital reprojection of precip, by default 'nearest_index'
    lapse_rate: float, optional
        lapse rate of temperature [C m-1] (default: -0.0065)
    resample_kwargs:
        Additional key-word arguments (e.g. label, closed) for time resampling method

    Returns
    --------
    press_out: xarray.DataArray (lazy)
        processed pressure forcing
    """
    if press.raster.dim0 != "time":
        raise ValueError(f'First press dim should be "time", not {press.raster.dim0}')
    # downscale pressure (lazy)
    press_out = press.raster.reproject_like(dem_model, method=reproj_method)
    # correct temperature based on high-res DEM
    if lapse_correction:
        # calculate downscaling addition
        press_factor = press_correction(dem_model, lapse_rate=lapse_rate)
        press_out = press_out * press_factor
    # resample time
    press_out.name = "press"
    press_out.attrs.update(unit="hPa")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean", logger=logger)
        press_out = resample_time(
            press_out, freq, conserve_mass=False, **resample_kwargs
        )
    return press_out


def pet(
    ds,
    temp,
    dem_model,
    method="debruin",
    press_correction=False,
    reproj_method="nearest_index",
    lapse_rate=-0.0065,
    freq=None,
    resample_kwargs={},
    logger=logger,
):

    """Determines reference evapotranspiration (lazy reprojection on model grid and resampling of time dimension to frequency).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with pressure [hPa], global radiation [W m-2], TOA incident solar radiation [W m-2]
    temp : xarray.DataArray
        DataArray with temperature [°C]
    dem_model : xarray.DataArray
        DataArray of the target resolution and projection, contains elevation
        data
    method : {'debruin', 'makkink'}
        Potential evapotranspiration method.
    press_correction : bool, default False
        If True pressure is corrected, based on elevation data of `dem_model`
    freq : str, Timedelta, default None
        output frequency of timedimension
    resample_kwargs:
        Additional key-word arguments (e.g. label, closed) for time resampling method

    Returns
    --------
    pet_out : xarray.DataArray (lazy)
        reference evapotranspiration
    """
    # # resample in time
    if temp.raster.dim0 != "time" or ds.raster.dim0 != "time":
        raise ValueError(f'First dimension of input variables should be "time"')
    # make sure temp and ds align both temporally and spatially
    if not np.all(temp["time"].values == ds["time"].values):
        raise ValueError("All input variables have same time index.")
    if not temp.raster.identical_grid(dem_model):
        raise ValueError("Temp variable should be on model grid.")
    # resample input to model grid
    ds_out = ds.raster.reproject_like(dem_model, method=reproj_method)
    if press_correction:
        ds_out["press"] = press(
            ds["press_msl"],
            dem_model,
            lapse_correction=press_correction,
            freq=None,  # do not change freq of press, put pet_out later
            reproj_method=reproj_method,
        )
    else:
        ds_out = ds_out.rename({"press_msl": "press"})
    timestep = to_timedelta(ds).total_seconds()
    if method == "debruin":
        pet_out = pet_debruin(
            temp,
            ds_out["press"],
            ds_out["kin"],
            ds_out["kout"],
            timestep=timestep,
        )
    if method == "makkink":
        pet_out = pet_makkink(temp, ds_out["press"], ds_out["kin"], timestep=timestep)
    # resample in time
    pet_out.name = "pet"
    pet_out.attrs.update(unit="mm")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="sum", logger=logger)
        pet_out = resample_time(pet_out, freq, conserve_mass=True, **resample_kwargs)
    return pet_out


def press_correction(
    dem_model, g=9.80665, R_air=8.3144621, Mo=0.0289644, lapse_rate=-0.0065
):
    """Pressure correction based on elevation lapse_rate.

    Parameters
    ----------
    dem_model : xarray.DataArray
        DataArray with high res lat/lon axis and elevation data
    g : float, default 9.80665
        gravitational constant [m s-2]
    R_air : float, default 8.3144621
        specific gas constant for dry air [J mol-1 K-1]
    Mo : float, default 0.0289644
        molecular weight of gas [g / mol]
    LapseRate : float, deafult -0.0065
        lapse rate of temperature [C m-1]

    Returns
    -------
    press_fact : xarray.DataArray
        pressure correction factor
    """
    # constant
    pow = g * Mo / (R_air * lapse_rate)
    press_fact = np.power(288.15 / (288.15 + lapse_rate * dem_model), pow).fillna(1.0)
    return press_fact


def temp_correction(dem, lapse_rate=-0.0065):

    """Temperature correction based on elevation data.

    Parameters
    ----------
    dem : xarray.DataArray
        DataArray with elevation
    lapse_rate : float, default -0.0065
        lapse rate of temperature [°C m-1]

    Returns
    -------
    temp_add : xarray.DataArray
        temperature addition
    """

    temp_add = (dem * lapse_rate).fillna(0)

    return temp_add


def pet_debruin(
    temp, press, k_in, k_ext, timestep=86400, cp=1005.0, beta=20.0, Cs=110.0
):
    """Determines De Bruin (2016) reference evapotranspiration.

    Parameters
    ----------
    temp : xarray.DataArray
        DataArray with temperature [°C]
    press : xarray.DataArray
        pressure at surface [hPa]
    k_in : xarray.DataArray
        global (=short wave incoming) radiation [W m-2]
    k_ext : xarray.DataArray}
        TOA incident solar radiation [W m-2]
    timestep : int, default 86400
        seconds per timestep
    cp : float, default 1005.0
        standard cp [J kg-1 K-1]
    beta : float, default 20.0
        correction constant [W m-2]
    Cs : float, default 110.0
        emperical constant [W m-2]

    Returns
    -------
    pet : xarray.DataArray
        reference evapotranspiration
    """
    # saturation and actual vapour pressure at given temperature [Pa]
    esat = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    # slope of vapour pressure curve
    slope = esat * (17.269 / (temp + 243.5)) * (1.0 - (temp / (temp + 243.5)))
    # compute latent heat of vapourization [J kg-1]
    lam = (2.502 * 10**6) - (2250.0 * temp)
    gamma = (cp * press) / (0.622 * lam)
    # compute ref. evaporation (with global radiation, therefore calling it potential)
    # in J m-2 over whole period
    ep_joule = (
        (slope / (slope + gamma))
        * (((1.0 - 0.23) * k_in) - (Cs * (k_in / (k_ext + 0.00001))))
    ) + beta
    ep_joule = xr.where(k_ext == 0.0, 0.0, ep_joule)
    pet = ((ep_joule / lam) * timestep).astype(np.float32)
    pet = xr.where(pet > 0.0, pet, 0.0)
    return pet


def pet_makkink(temp, press, k_in, timestep=86400, cp=1005.0):
    """Determnines Makkink reference evapotranspiration.

    Parameters
    ----------
    temp : xarray.DataArray
        DataArray with temperature [°C]
    press : xarray.DataArray
        DataArray with pressure [hPa]
    k_in : xarray.DataArray
        DataArray with global radiation [W m-2]
    timestep : int, default 86400
        seconds per timestep
    cp : float, default 1005.0
        standard cp [J kg-1 K-1]

    Returns
    --------
    pet : xarray.DataArray (lazy)
        reference evapotranspiration
    """
    # saturation and actual vapour pressure at given temperature [Pa]
    esat = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    # slope of vapour pressure curve
    slope = esat * (17.269 / (temp + 243.5)) * (1.0 - (temp / (temp + 243.5)))
    # compute latent heat of vapourization [J kg-1]
    lam = (2.502 * 10**6) - (2250.0 * temp)
    gamma = (cp * press) / (0.622 * lam)

    ep_joule = 0.65 * slope / (slope + gamma) * k_in
    pet = ((ep_joule / lam) * timestep).astype(np.float32)
    pet = xr.where(pet > 0.0, pet, 0.0)
    return pet


def resample_time(
    da,
    freq,
    label="right",
    closed="right",
    upsampling="bfill",
    downsampling="mean",
    conserve_mass=True,
    logger=logger,
):
    """Resample data to destination frequency.
    Skip if input data already at output frequency.

    Parameters
    ----------
    da: xarray.DataArray
        Input data
    freq: str, pd.timedelta
        Output frequency.
    label: {'left', 'right'}, optional
        Side of each interval to use for labeling. By default 'right'.
    closed: {'left', 'right'}, optional
        Side of each interval to treat as closed. By default 'right'.
    upsampling, downsampling: str, optional
        Resampling method if output frequency is higher, lower (resp.) compared
        to input frequency.
    conserve_mass: bool, optional
        If True multiply output with relative change in frequency to conserve mass

    Returns
    --------
    pet : xarray.DataArray
        Resampled data.
    """
    da_out = da
    dfreq = delta_freq(da, freq)
    if not np.isclose(dfreq, 1.0):
        resample = upsampling if dfreq < 1 else downsampling
        pre = "up" if dfreq < 1 else "down"
        logger.debug(
            f"{pre}sampling {da.name} using {resample}; conserve mass: {conserve_mass}"
        )
        if not hasattr(xr.core.resample.DataArrayResample, resample):
            raise ValueError(f"unknown resampling option {resample}")
        da_resampled = da.resample(time=freq, skipna=True, label=label, closed=closed)
        da_out = getattr(da_resampled, resample)()
        if conserve_mass:
            da_out = da_out * min(dfreq, 1)

    return da_out


def delta_freq(da_or_freq, da_or_freq1):
    """Returns the relative difference between the dataset mean timestep and destination freq
    <1 : upsampling
    1 : same
    >1 : downsampling
    """
    return to_timedelta(da_or_freq1) / to_timedelta(da_or_freq)


def to_timedelta(da_or_freq):
    if isinstance(da_or_freq, (xr.DataArray, xr.Dataset)):
        freq = da_to_timedelta(da_or_freq)
    else:
        freq = freq_to_timedelta(da_or_freq)
    return freq


def da_to_timedelta(da):
    return pd.to_timedelta(np.diff(da.time).mean())


def freq_to_timedelta(freq):
    # Add '1' to freq that doesn't have any digit
    if isinstance(freq, str) and not bool(re.search(r"\d", freq)):
        freq = "1{}".format(freq)

    # Convert str to datetime.timedelta
    return pd.to_timedelta(freq)
