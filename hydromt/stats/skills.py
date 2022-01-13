#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import bottleneck


def bias(sim: xr.DataArray, obs: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Returns the bias between two time series.

        .. math::
         Bias=\\frac{1}{N}\\sum_{i=1}^{N}(obs_{i}-sim_{i})

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        bias
    """
    # wrap numpy function
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    bias = xr.apply_ufunc(_bias, sim, obs, **kwargs)
    bias.name = "bias"
    return bias


def percentual_bias(
    sim: xr.DataArray, obs: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Returns the percentual bias between two time series.

        .. math::
         PBias= 100 * \\frac{\\sum_{i=1}^{N}(sim_{i}-obs_{i})}{\\sum_{i=1}^{N}(obs_{i})}

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        percentual bias
    """
    # wrap numpy function
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    pbias = xr.apply_ufunc(_pbias, sim, obs, **kwargs)
    pbias.name = "pbias"
    return pbias


def nashsutcliffe(
    sim: xr.DataArray, obs: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Returns the Nash-Sutcliffe model efficiency based on a simulated
    and observed time series.

        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(sim_{i}-obs_{i})^2}{\\sum_{i=1}^{N}(obs_{i}-\\bar{obs})^2}

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        the Nash-Sutcliffe model efficiency
    """
    # wrap numpy function
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    nse = xr.apply_ufunc(_nse, sim, obs, **kwargs)
    nse.name = "nse"
    return nse


def lognashsutcliffe(
    sim: xr.DataArray, obs: xr.DataArray, epsilon: float = 1e-6, dim: str = "time"
) -> xr.DataArray:
    """Returns the log Nash-Sutcliffe model efficiency based on simulated
    and observed time series.

        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(log(sim_{i})-log(obs_{i}))^2}{\\sum_{i=1}^{N}(log(sim_{i})-log(\\bar{obs})^2}-1)*-1

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    epsilon : float, optional
        small value to avoid taking the log of zero (the default is 1e-6)
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        the log of the Nash-Sutcliffe model efficiency
    """
    obs = np.log(obs + epsilon)
    sim = np.log(sim + epsilon)
    # wrap numpy function
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    log_nse = xr.apply_ufunc(_nse, sim, obs, **kwargs)
    log_nse.name = "log_nse"
    return log_nse


def pearson_correlation(
    sim: xr.DataArray, obs: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Returns the Pearson correlation coefficient of two time series.

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        the pearson correlation coefficient
    """
    # wrap numpy function
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    pearsonr = xr.apply_ufunc(_pearson_correlation, sim, obs, **kwargs)
    pearsonr.name = "pearson_coef"
    return pearsonr


def spearman_rank_correlation(
    sim: xr.DataArray, obs: xr.DataArray, dim: str = "time"
) -> xr.DataArray:
    """Returns the spearman rank correlation coefficient of
    two time series.

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        the spearman rank correlation
    """
    # wrap numpy function
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    spearmanr = xr.apply_ufunc(_spearman_correlation, sim, obs, **kwargs)
    spearmanr.name = "spearmanr_coef"
    return spearmanr


def kge_non_parametric(
    sim: xr.DataArray, obs: xr.DataArray, dim: str = "time"
) -> xr.Dataset:
    """Returns the Non Parametric Kling-Gupta Efficiency (KGE) of two
    time series with decomposed scores (Pool et al., 2018)

    .. ref::
    Pool, Vis, and Seibert, 2018 Evaluating model performance: towards
    a non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences Journal.

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataSet
        Non Parametric KGE and decomposed score
    """
    cc = spearman_rank_correlation(sim, obs, dim=dim)
    cc.name = "kge_np_spearman_rank_correlation_coef"
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    alpha = xr.apply_ufunc(_fdc_alpha, sim, obs, **kwargs)
    alpha.name = "kge_np_rel_var"
    beta = sim.sum(dim=dim) / obs.sum(dim=dim)
    beta.name = "kge_np_bias"
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    kge.name = "kge_np"
    ds_out = xr.merge([kge, cc, alpha, beta])
    return ds_out


def kge_non_parametric_flood(
    sim: xr.DataArray, obs: xr.DataArray, dim: str = "time"
) -> xr.Dataset:
    """Returns the Non Parametric Kling-Gupta Efficiency (KGE) of two time
    series optimized for flood peaks using Pearson correlation (Pool et al., 2018)

    .. ref::
    Pool, Vis, and Seibert, 2018 Evaluating model performance: towards
    a non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences Journal.

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataSet
        Non Parametric KGE optimize for flood peaks and decomposed scores
    """
    # cc = spearman_rank_correlation(sim, obs)
    cc = pearson_correlation(sim, obs, dim=dim)
    cc.name = "kge_np_flood_pearson_coef"
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    alpha = xr.apply_ufunc(_fdc_alpha, sim, obs, **kwargs)
    alpha.name = "kge_np_flood_rel_var"
    beta = sim.sum(dim=dim) / obs.sum(dim=dim)
    beta.name = "kge_np_flood_bias"
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    kge.name = "kge_np_flood"
    ds_out = xr.merge([kge, cc, alpha, beta])
    return ds_out


def rsquared(sim: xr.DataArray, obs: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Returns the coefficient of determination of two time series.

        .. math::
         r^2=(\\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})^2

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        the coefficient of determination
    """
    # R2 equals the square of the Pearson correlation coefficient between obs and sim
    rsquared = pearson_correlation(sim, obs, dim=dim) ** 2
    rsquared.name = "rsquared"
    return rsquared


def mse(sim: xr.DataArray, obs: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Returns the mean squared error (MSE) between two time series.

        .. math::
         MSE=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        the mean squared error
    """
    # wrap numpy function
    kwargs = dict(
        input_core_dims=[[dim], [dim]], dask="parallelized", output_dtypes=[float]
    )
    mse = xr.apply_ufunc(_mse, sim, obs, **kwargs)
    mse.name = "mse"
    return mse


def rmse(sim: xr.DataArray, obs: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Returns the root mean squared error between two time series.

        .. math::
         RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataArray
        the root mean squared error
    """
    rmse = np.sqrt(mse(sim, obs, dim=dim))
    rmse.name = "rmse"
    return rmse


def kge(sim: xr.DataArray, obs: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Returns the Kling-Gupta Efficiency (KGE) of two time series (Gupta et al. 2009)

        .. ref::
        Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean
        squared error and NSE performance criteria: Implications for improving
        hydrological modelling.

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataSet
        KGE with decomposed scores
    """
    cc = pearson_correlation(sim, obs, dim=dim)
    cc.name = "kge_pearson_coef"
    alpha = sim.std(dim=dim) / obs.std(dim=dim)
    alpha.name = "kge_rel_var"
    beta = sim.sum(dim=dim) / obs.sum(dim=dim)
    beta.name = "kge_bias"
    kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    kge.name = "kge"
    ds_out = xr.merge([kge, cc, alpha, beta])
    return ds_out


def kge_2012(sim: xr.DataArray, obs: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Returns the Kling-Gupta Efficiency (KGE) of two time series based on
    Kling et al. (2012)

        .. ref::
        Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the
        upper Danube basin under an ensemble of climate change scenarios.
        Journal of Hydrology, 424, 264-277, doi:10.1016/j.jhydrol.2012.01.011.

    Parameters
    ----------
    sim : xarray.DataArray
        simulations time series
    obs : xarray.DataArray
        observations time series
    dim : str, optional
        name of time dimension in sim and obs (the default is 'time')

    Returns
    -------
    xarray.DataSet
        KGE (2012) with decomposed scores
    """
    cc = pearson_correlation(sim, obs, dim=dim)
    cc.name = "kge_2012_pearson_coef"
    beta = sim.sum(dim=dim) / obs.sum(dim=dim)
    beta.name = "kge_2012_bias"
    # divide alpha by bias
    alpha = (sim.std(dim=dim) * obs.sum(dim=dim)) / (
        obs.std(dim=dim) * sim.sum(dim=dim)
    )
    alpha.name = "kge_2012_rel_var"
    kge_2012 = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    kge_2012.name = "kge_2012"
    ds_out = xr.merge([kge_2012, cc, alpha, beta])
    return ds_out


# correlation ufunc function
# from http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization
def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    return np.nanmean(
        (x - np.nanmean(x, axis=-1, keepdims=True))
        * (y - np.nanmean(y, axis=-1, keepdims=True)),
        axis=-1,
    )


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return _covariance(x, y) / (np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1))


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_ranks = bottleneck.nanrankdata(x, axis=-1)
    y_ranks = bottleneck.nanrankdata(y, axis=-1)
    return _pearson_correlation(x_ranks, y_ranks)


# numpy functions
def _fdc_alpha(sim: np.ndarray, obs: np.ndarray, axis: int = -1) -> float:
    fdc_s = np.sort(sim, axis=axis) / (np.nanmean(sim, axis=axis) * len(sim))
    fdc_o = np.sort(obs, axis=axis) / (np.nanmean(obs, axis=axis) * len(obs))
    return 1 - 0.5 * np.nanmean(np.abs(fdc_s - fdc_o))


def _bias(sim: np.ndarray, obs: np.ndarray, axis: int = -1) -> float:
    """bias"""
    return np.nansum(sim - obs, axis=axis) / np.nansum(np.isfinite(obs), axis=axis)


def _pbias(sim: np.ndarray, obs: np.ndarray, axis: int = -1) -> float:
    """percentual bias"""
    return np.nansum(sim - obs, axis=axis) / np.nansum(obs, axis=axis)


def _mse(sim: np.ndarray, obs: np.ndarray, axis: int = -1) -> float:
    """mean squared error"""
    mse = np.nansum((obs - sim) ** 2, axis=axis)
    return mse


def _nse(sim: np.ndarray, obs: np.ndarray, axis: int = -1) -> float:
    """nash-sutcliffe efficiency"""
    obs_mean = np.nanmean(obs, axis=axis)
    a = np.nansum((sim - obs) ** 2, axis=axis)
    b = np.nansum((obs - obs_mean[..., None]) ** 2, axis=axis)
    return 1 - a / b
