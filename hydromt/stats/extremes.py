"""Functions for extreme value analysis."""

import math as math
from typing import Optional

import dask
import numpy as np
import xarray as xr
from numba import njit
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "eva_block_maxima",
    "eva_peaks_over_threshold",
    "eva",
    "get_peaks",
    "get_return_value",
    "fit_extremes",
]

_RPS: NDArray[np.int_] = np.array([2, 5, 10, 25, 50, 100, 250, 500])  # years
_DISTS = {
    "POT": ["exp", "gpd"],
    "BM": ["gumb", "gev"],
}

_DEFAULT_PERIOD_STR = "365.25D"

## high level methods


def eva(
    da: xr.DataArray,
    ev_type: str = "BM",
    min_dist: int = 0,
    qthresh: float = 0.9,
    period: str = _DEFAULT_PERIOD_STR,
    min_sample_size: int = 0,
    distribution: Optional[str] = None,
    rps: NDArray[np.int_] = _RPS,
    criterium: str = "AIC",
) -> xr.Dataset:
    """Return Extreme Value Analysis.

    Extreme value analysis based on block maxima (BM) or Peaks Over Threshold (POT).
    The method selects the peaks, fits a distribution and calculates return values for
    provided return periods.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries data, must have a regular spaced 'time' dimension.
    ev_type: {"POT", "BM"}
        Peaks over threshold (POT) or block maxima (BM) peaks, by default "BM"
    period : str, optional
        Period string, by default "365.25D". See pandas.Timedelta for options.
    min_dist : int, optional
        Minimum distance between peaks measured in time steps, by default 0
    qthresh : float, optional
        Quantile threshold used with peaks over threshold method, by default 0.9
    min_sample_size : int, optional
        Minimum number of finite values in a valid block, by default 0. Peaks of
        invalid blocks are set to NaN.
    distribution : str, optional
        Short name of distribution. If None (default) the optimal block maxima
        distribution ("gumb" or "gev" for BM and "exp" or "gpd" for POT) is selected
        based on `criterium`.
    rps : np.ndarray, optional
        Array of return periods, by default [2, 5, 10, 25, 50, 100, 250, 500]
    criterium: {'AIC', 'AICc', 'BIC'}
        Selection criterium, by default "AIC"

    Returns
    -------
    xr.Dataset
        Dataset with peaks timeseries, distribution name and parameters
        and return values.
    """
    da_peaks = get_peaks(
        da,
        ev_type=ev_type,
        min_dist=min_dist,
        qthresh=qthresh,
        period=period,
        min_sample_size=min_sample_size,
    )
    # fit distribution using lmom
    da_params = fit_extremes(
        da_peaks, ev_type=ev_type, distribution=distribution, criterium=criterium
    )
    # get return values
    da_rps = get_return_value(da_params, rps=rps)
    # combine data
    return xr.merge([da_peaks, da_params, da_rps])


# In theory could be removed because redundant with eva
def eva_block_maxima(
    da: xr.DataArray,
    period: str = _DEFAULT_PERIOD_STR,
    min_dist: int = 0,
    min_sample_size: int = 0,
    distribution: Optional[str] = None,
    rps: NDArray[np.int_] = _RPS,
    criterium: str = "AIC",
) -> xr.Dataset:
    """Return EVA based on block maxima.

    Extreme valua analysis based on block maxima. The method selects the peaks,
    fits a distribution and calculates return values for provided return periods.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries data, must have a regular spaced 'time' dimension.
    period : str, optional
        Period string, by default "365.25D". See pandas.Timedelta for options.
    min_dist : int, optional
        Minimum distance between peaks measured in time steps, by default 0
    min_sample_size : int, optional
        Minimum number of finite values in a valid block, by default 0. Peaks of
        invalid blocks are set to NaN.
    distribution : str, optional
        Short name of distribution. If None (default) the optimal block maxima
        distribution ("gumb" or "gev") is selected based on `criterium`.
    rps : np.ndarray, optional
        Array of return periods, by default [2, 5, 10, 25, 50, 100, 250, 500]
    criterium: {'AIC', 'AICc', 'BIC'}
        Selection criterium, by default "AIC"

    Returns
    -------
    xr.Dataset
        Dataset with peaks timeseries, distribution name and parameters
        and return values.
    """
    da_bm = get_peaks(
        da,
        ev_type="BM",
        min_dist=min_dist,
        min_sample_size=min_sample_size,
        period=period,
    )
    # fit distribution using lmom
    da_params = fit_extremes(
        da_bm, ev_type="BM", distribution=distribution, criterium=criterium
    )
    # get return values
    da_rps = get_return_value(da_params, rps=rps)
    # combine data
    return xr.merge([da_bm, da_params, da_rps])


# In theory could be removed because redundant with eva
def eva_peaks_over_threshold(
    da: xr.DataArray,
    qthresh: float = 0.9,
    min_dist: int = 0,
    min_sample_size: int = 0,
    period: str = _DEFAULT_PERIOD_STR,
    distribution: Optional[str] = None,
    rps: NDArray[np.int_] = _RPS,
    criterium: str = "AIC",
) -> xr.Dataset:
    """Return EVA based on peaks over threshold.

    Extreme valua analysis based on peaks over threshold. The method selects the
    peaks, fits a distribution and calculates return values for provided return
    periods.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries data, must have a regular spaced 'time' dimension.
    qthresh : float, optional
        Quantile threshold used with peaks over threshold method, by default 0.9
    min_dist : int, optional
        Minimum distance between peaks measured in time steps, by default 0
    min_sample_size : int, optional
        Minumimum number of finite values in a valid block, by default 0. Peaks of
    period : str, optional
        Period string, by default "365.25D". See pandas.Timedelta for options.
    distribution : str, optional
        Short name of distribution. If None (default) the optimal block maxima
        distribution ("exp" or "gpd") is selected based on `criterium`.
    rps : np.ndarray, optional
        Array of return periods, by default [1.5, 2, 5, 10, 20, 50, 100, 200, 500]
    criterium: {'AIC', 'AICc', 'BIC'}
        distrition selection criterium, by default "AIC"

    Returns
    -------
    xr.Dataset
        Dataset with peaks timeseries, distribution name and parameters
        and return values.
    """
    da_bm = get_peaks(
        da,
        ev_type="POT",
        min_dist=min_dist,
        period=period,
        qthresh=qthresh,
        min_sample_size=min_sample_size,
    )
    # fit distribution using lmom
    da_params = fit_extremes(
        da_bm, ev_type="POT", distribution=distribution, criterium=criterium
    )
    # get return values
    da_rps = get_return_value(da_params, rps=rps)
    return xr.merge([da_bm, da_params, da_rps])


def get_peaks(
    da: xr.DataArray,
    ev_type: str = "BM",
    min_dist: int = 0,
    qthresh: float = 0.9,
    period: str = "year",
    min_sample_size: int = 0,
    time_dim: str = "time",
) -> xr.DataArray:
    """Return peaks from time series.

    Return the timeseries with all but the peak values set to NaN.

    For block maxima (BM) peaks, peaks are determined by finding the maximum within
    each period and then ensuring a minimum distance (min_dist) and minimum sample size
    (min_sample_size) per period.

    For Peaks Over Threshold (POT), peaks are determined solely based on the minimum
    distance between peaks.

    The average interarrival time (extreme_rates) is calculated by dividing the number
    of peaks by the total duration of the timeseries and converting it to a yearly
    rate.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries data, must have a regular spaced 'time' dimension.
    ev_type : {"POT", "BM"}
        Peaks over threshold (POT) or block maxima (BM) peaks, by default "BM"
    min_dist : int, optional
        Minimum distance between peaks measured in time steps, by default 0
    qthresh : float, optional
        Quantile threshold used with peaks over threshold method, by default 0.9
    period : {'year', 'month', 'quarter', pandas.Timedelta}, optional
        Period string, by default "year".
    min_sample_size : int, optional
        Minimum number of finite values in a valid block, by default 0. Peaks of
        invalid bins are set to NaN.

    Returns
    -------
    xr.DataArray
        Timeseries data with only peak values, all other values are set to NaN.
        Average interarrival time calculated based on the average number of peaks
        per year and stored as "extreme_rates"
    """
    if not (0 < qthresh < 1.0):
        raise ValueError("Quantile 'qthresh' should be between (0,1)")
    if time_dim not in da.dims:
        raise ValueError(f"Input array should have a '{time_dim}' dimension")
    if ev_type.upper() not in _DISTS.keys():
        raise ValueError(
            f"Unknown ev_type {ev_type.upper()}, select from {_DISTS.keys()}."
        )
    bins = None
    nyears = (da[time_dim][-1] - da[time_dim][0]).dt.days / 365.2425
    if period in ["year", "quarter", "month"] and ev_type.upper() == "BM":
        bins = getattr(da[time_dim].dt, period).values
    elif ev_type.upper() == "BM":
        tstart = da[time_dim].resample(time=period, label="left").first()
        bins = tstart.reindex_like(da, method="ffill").values.astype(float)
    else:
        min_sample_size = 0  # min_sample_size not used for POT

    def func(x):
        return local_max_1d(
            x, min_dist=min_dist, bins=bins, min_sample_size=min_sample_size
        )

    duck = dask.array if isinstance(da.data, dask.array.Array) else np
    lmax = duck.apply_along_axis(func, da.get_axis_num(time_dim), da.data)
    # apply POT threshold
    peaks = da.where(lmax)
    if ev_type.upper() == "POT":
        peaks = da.where(peaks > da.quantile(qthresh, dim=time_dim))
    # get extreme rate per year
    da_rate = np.isfinite(peaks).sum(time_dim) / nyears
    peaks = peaks.assign_coords({"extremes_rate": da_rate})
    peaks.name = "peaks"
    return peaks


def get_return_value(
    da_params: xr.DataArray,
    rps: NDArray[np.int_] = _RPS,
    extremes_rate: float = 1.0,
) -> xr.DataArray:
    """Return return value based on EVA.

    Return return values based on a fitted extreme value distribution using the
    :py:meth:`fit_extremes` method based on the scipy inverse survival function (isf).

    Parameters
    ----------
    da_params : xr.DataArray
        Short name and parameters of extreme value distribution,
        see also :py:meth:`fit_extremes`.
    rps : np.ndarray, optional
        Array of return periods in years, by default
        [1.5, 2, 5, 10, 20, 50, 100, 200, 500]
    extremes_rate : float, optional
        Average number of peaks per period, by default 1.0
        Only used if extremes_rate is not provided as coordinate in da_params.

    Returns
    -------
    xr.DataArray
        Return values
    """

    def _return_values_1d(p, r, d, rps=rps):
        if np.isnan(p).all():
            return np.full(rps.size, np.nan)
        if d == "gumb" and len(p) == 3:
            p = p[1:]
        return _get_return_values(p, d, rps=rps, extremes_rate=r)

    if isinstance(rps, list):
        rps = np.asarray(rps)
    elif not isinstance(rps, np.ndarray):
        raise ValueError("rps should be a list or numpy array")
    if "dparams" not in da_params.dims:
        raise ValueError("da_params should have a 'dparams' dimension")
    if "distribution" not in da_params.coords:
        raise ValueError("da_params should have a 'distribution' coordinate")
    distributions = da_params["distribution"].load()
    if "extremes_rate" in da_params.coords:
        extremes_rate = da_params["extremes_rate"].load()
    elif isinstance(extremes_rate, (int, float)):
        extremes_rate = np.full(distributions.shape, extremes_rate)
    elif extremes_rate.shape != distributions.shape:
        raise ValueError(
            "extremes_rate should be a scalar or have the same shape as distribution"
        )

    if da_params.ndim == 1:  # fix case of single dim
        da_params = da_params.expand_dims("index")
    da_rvs = xr.apply_ufunc(
        _return_values_1d,
        da_params,
        extremes_rate,
        distributions,
        input_core_dims=(["dparams"], [], []),
        output_core_dims=[["rps"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs=dict(output_sizes={"rps": rps.size}),
        output_dtypes=[float],
    )
    da_rvs["rps"] = xr.IndexVariable("rps", rps)
    da_rvs.name = "return_values"
    return da_rvs.squeeze()


def fit_extremes(
    da_peaks: xr.DataArray,
    distribution: Optional[str] = None,
    ev_type: str = "BM",
    criterium: str = "AIC",
    time_dim: str = "time",
) -> xr.DataArray:
    """Return distribution fit from extremes.

    Return the fitted parameters of the extreme value `distribution` based on
    the lmoments method. If no distribution name is provided `distribution=None`,
    the optimal distribution is selected based on `criterium` from a list
    of distributions associated with `ev_type`.

    Block maximum distributions: gumbel ("gumb") and general extreme value ("gev").
    Peak over threshold distributions: exponential ("exp") and general pareto
    distribution ("gdp").

    Parameters
    ----------
    da_peaks : xr.DataArray
        Timeseries data with only peak values, any other values are set to NaN.
        The DataArray should contain as coordinate `extreme_rates` indicating
        the yearly rate of extreme events. If not provided, this value is set to 1.0
    distribution: {'gev', 'gpd', 'gumb', 'exp'}, optional
        Short distribution name. If None (default) the optimal distribution
        is calculated based on `criterium`
    ev_type : {"POT", "BM"}
        Peaks over threshold (POT) or block maxima (BM) peaks, by default "BM"
    criterium: {'AIC', 'AICc', 'BIC'}
        Selection criterium, by default "AIC"

    Returns
    -------
    xr.DataArray
        Parameters and short name of optimal extreme value distribution.
    """
    distributions = _DISTS.get(ev_type.upper(), None)
    if distribution is not None:
        if isinstance(distribution, str):
            distributions = [distribution]
        elif not isinstance(distribution, list):
            raise ValueError(
                f"distribution should be a string or list, got {type(distribution)}"
            )
    elif ev_type.upper() not in _DISTS:
        raise ValueError(
            f"Unknown ev_type {ev_type.upper()}, select from {_DISTS.keys()}."
        )

    def _fitopt_1d(x, distributions=distributions, criterium=criterium):
        if np.isnan(x).all():
            return np.array([np.nan, np.nan, np.nan, -1])
        params, d = lmoment_fitopt(x, distributions=distributions, criterium=criterium)
        if len(params) == 2:
            params = np.concatenate([[0], params])
        # trick to include distribution name; -1 if too little data to fit -> NA
        idist = distributions.index(d) if d in distributions else -1
        return np.concatenate([params, [idist]])

    if da_peaks.ndim == 1:  # fix case of single dim
        da_peaks = da_peaks.expand_dims("index")
    da_params = xr.apply_ufunc(
        _fitopt_1d,
        da_peaks,
        input_core_dims=[[time_dim]],
        output_core_dims=[["dparams"]],
        dask_gufunc_kwargs=dict(output_sizes={"dparams": 4}),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    # split output
    idist = da_params.isel(dparams=-1).values.astype(int)
    distributions = np.atleast_1d(np.array(distributions + ["NA"])[idist])
    da_params = da_params.isel(dparams=slice(0, -1))
    da_params.name = "parameters"
    # add coordinates
    dist_dims = list([d for d in da_params.dims if d != "dparams"])
    coords = dict(
        dparams=xr.IndexVariable("dparams", ["shape", "loc", "scale"]),
        distribution=xr.DataArray(dims=dist_dims, data=distributions),
    )
    # forward extremes_rate if provided
    if "extremes_rate" in da_peaks.coords:
        coords["extremes_rate"] = da_peaks["extremes_rate"]
    da_params = da_params.assign_coords(coords)
    return da_params.squeeze()


## PEAKS
@njit
def local_max_1d(
    arr: NDArray[np.number],
    bins: Optional[NDArray[np.int_]] = None,
    min_dist: int = 0,
    min_sample_size: int = 0,
) -> NDArray[np.bool_]:
    """Return boolean index of local maxima in `arr` which are `min_dist` apart.

    Parameters
    ----------
    arr : np.ndarray
        1D time series
    bins : np.ndarray, optional
        1D array of with uniquely numbered bins (blocks), by default None.
        If provided only the largest peak per block is flagged.
    min_dist : int, optional
        Minimum distance between peaks, by default 0
    min_sample_size : int, optional
        minimum number of samples per block, by default 0

    Returns
    -------
    np.ndarray
        boolean index of local maxima
    """
    a0 = arr[0]
    amax = -np.inf  # peak value
    imax = -min_dist  # peak index
    bsize = 0
    min_sample_size = 0 if bins is None else min_sample_size
    up = False  # sign of difference between subsequent values
    out = np.array([bool(0) for _ in range(arr.size)], np.bool_)
    for i in range(arr.size):
        a1 = arr[i]
        if not np.isfinite(a1):
            a0 = a1
            continue
        dd = i - 1 - imax  # distance to previous peak
        if (imax > 0) and (
            (bins is None and dd == (min_dist + 1))
            or (bins is not None and bins[i - 1] != bins[imax] and dd > min_dist)
        ):
            if bsize >= min_sample_size:
                out[imax] = True
            amax = -np.inf
            bsize = 0
        if up and a1 < a0 and a0 > amax:  # peak
            imax = i - 1
            amax = a0
        if a1 < a0:
            up = False
        elif a1 > a0:
            up = True
        bsize += 1
        a0 = a1
    if imax > 0 and bsize >= min_sample_size:
        out[imax] = True
    return out


## LINK TO SCIPY.STATS


def get_dist(distribution: str) -> stats.rv_continuous:
    """Return scipy.stats distribution."""
    _DISTS = {
        "gev": "genextreme",
        "gpd": "genpareto",
        "gumb": "gumbel_r",
        "exp": "genpareto",
    }
    _scipy_dist_name = _DISTS.get(distribution, distribution)
    dist = getattr(stats, _scipy_dist_name, None)
    if dist is None:
        raise ValueError(f'Distribution "{_scipy_dist_name}" not found in scipy.stats.')
    return dist


def get_frozen_dist(params, distribution: str):
    """Return frozen distribution.

    Returns scipy.stats frozen distribution, i.e.: with set parameters.
    """
    return get_dist(distribution)(*params[:-2], loc=params[-2], scale=params[-1])


## STATS


def _aic(x, params, distribution: str):
    """Return Akaike Information Criterion for a frozen distribution."""
    k = len(params)
    nll = get_frozen_dist(params, distribution).logpdf(x).sum()
    aic = 2 * k - 2 * nll
    return aic


def _aicc(x, params, distribution: str):
    """Return AICC.

    Return Akaike Information Criterion with correction for small sample size
    for a frozen distribution.
    """
    k = len(params)
    aic = _aic(x, params, distribution)
    aicc = aic + ((2 * k) ** 2 + 2 * k) / (len(x) - k - 1)
    return aicc


def _bic(x, params, distribution: str):
    """Return Bayesian Information Criterion for a frozen distribution."""
    k = len(params)
    nll = get_frozen_dist(params, distribution).logpdf(x).sum()
    bic = k * np.log(len(x)) - 2 * nll
    return bic


## TRANSFORMATIONS


def _get_return_values(
    params, distribution: str, rps: NDArray[np.int_] = _RPS, extremes_rate=1.0
):
    q = 1 / rps / extremes_rate
    return get_frozen_dist(params, distribution).isf(q)


def _get_return_periods(x, a=0.0, extremes_rate=1.0):
    assert np.all(np.isfinite(x))
    b = 1.0 - 2.0 * a
    ranks = (len(x) + 1) - stats.rankdata(x, method="average")
    freq = ((ranks - a) / (len(x) + b)) * extremes_rate
    rps = 1 / freq
    return rps


## CONFIDENCE INTERVALS


def lmoment_ci(x, distribution, nsample=1000, alpha=0.9, rps=_RPS, extremes_rate=1.0):
    q = 1 / rps / extremes_rate
    dist = get_dist(distribution)

    def func(x, distribution=distribution, q=q):
        p = lmoment_fit(x, distribution)
        return dist.isf(q, *p[:-2], loc=p[-2], scale=p[-1])

    x_sample = np.random.choice(x, size=[nsample, x.size], replace=True)
    xrv = np.apply_along_axis(func, 1, x_sample)

    percentiles = np.array([(1 - alpha) / 2, 1 - (1 - alpha) / 2]) * 100
    return np.percentile(xrv, percentiles, axis=0)


## PLOTS


def plot_return_values(
    x: xr.DataArray,
    params: xr.DataArray,
    distribution: str,
    x_scale: Optional[str] = "gumbel",
    ax=None,
    color: Optional[str] = "k",
    a: Optional[float] = 0,
    alpha: Optional[float] = 0.9,
    nsample: int = 1000,
    rps: NDArray[np.int_] = _RPS,
    extremes_rate: float = 1.0,
):
    """Return figure of EVA fit and empirical data.

    Parameters
    ----------
    x : xr.DataArray
        Timeseries data with only peak values, all other values are set to NaN
    params : xr.DataArray
        Short name and parameters of extreme value distribution,
        see also :py:meth:`fit_extremes`.
    distribution : str
        Name of distribution used for fit. Can be set to `empirical` if no fit is
        performed
    x_scale : str, optional
        transformation of the x-axis for the figure
    ax :

    color : str, optional
        Color of the line. By default colord is black 'k'.
    a : float, optional
        Float for the Gringorten position. By default a = 0
    alpha : float, optional
        alpha value for the confidence interval of the EV fit. By default, alpha = 0.9
    nsample : int
        number of samples used to calculate the confidence interval. By default,
        nsample = 1000
    rps : np.ndarray, optional
        Specific return periods to highlight as points in the plot. By default
        [2, 5, 10, 25, 50, 100, 250, 500] years
    extremes_rate : float.
        Average interarrival time calculated based on the average number of peaks
        per year. For annual maxima, extremes_rate = 1

    Returns
    -------
    ax.figure
        Figure of EVA fit and empirical data
    """
    import matplotlib.pyplot as plt

    rvs_obs = np.sort(x[np.isfinite(x)])
    if distribution != "empirical":
        params = params[-2:] if distribution == "gumb" else params
        rvs_sim = _get_return_values(
            params, distribution, rps=rps, extremes_rate=extremes_rate
        )
    else:
        rvs_sim = rvs_obs
    rps_obs = _get_return_periods(rvs_obs, a=a, extremes_rate=extremes_rate)

    if x_scale == "gumbel":
        xsim = -np.log(-np.log(1.0 - 1.0 / rps))
        xobs = -np.log(-np.log(1.0 - 1.0 / rps_obs))
    else:
        xsim = rps
        xobs = rps_obs

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(xobs, rvs_obs, color=color, marker="o", label="plot position", lw=0)

    if distribution != "empirical":
        ax.plot(
            xsim, rvs_sim, color=color, ls="--", label=f"{distribution.upper()} fit"
        )

    if alpha is not None and nsample > 0 and distribution != "empirical":
        urvs = lmoment_ci(
            x,
            distribution,
            nsample=nsample,
            alpha=alpha,
            rps=rps,
            extremes_rate=extremes_rate,
        )
        ax.plot(
            xsim,
            urvs[0, :],
            color=color,
            ls=":",
            label=f"conf. interval (alpha = {alpha:.2f})",
        )
        ax.plot(
            xsim,
            urvs[1, :],
            color=color,
            ls=":",
        )

    ax.legend(loc="upper left")

    ymin = 0
    ymax = np.nanmax([np.nanmax(rvs_obs), np.nanmax(rvs_sim), 0])
    ymax = ymax * 1.1
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel("Return value")
    if x_scale == "gumbel":
        ax.set_xticks(xsim)
        ax.set_xticklabels(rps)
    ax.set_xlabel("Return period")
    ax.grid()

    return ax


## LMOMENTS FITTING
# credits Ferdinand Diermanse


def lmoment_fitopt(x, distributions=None, criterium="AIC"):
    """Return parameters based on lmoments fit.

    Lmomentfit routine derive parameters of a distribution function
    based on lmoments. The distribution selection is based on the
    AIC criterium.

    Based on the theory of Hosking and Wallis (1997) Appendix A.

    Parameters
    ----------
    X: 1D array of float
        data series
    distributions: iterable of {'gev', 'gpd', 'gumb', 'exp'}
        iterable of distribution names
    criterium: {'AIC', 'AICc', 'BIC'}
        distrition selection criterium

    Returns
    -------
    params: ndarray of float
        array of distribution parameters
    distribution: str
        selected distribution
    """
    fgof = {"AIC": _aic, "AICC": _aicc, "BIC": _bic}.get(criterium.upper())
    # make sure the timeseries does not contain NaNs
    x = x[~np.isnan(x)]

    # derive first four L-moments from data
    lmom = get_lmom(x, 4)
    # derive parameters of distribution function
    params = {}
    gof_values = []
    for distribution in distributions:
        params[distribution] = _lmomentfit(lmom, distribution)
        gof_values.append(fgof(x, params[distribution], distribution))
    distribution = distributions[np.argmin(gof_values)]

    return params[distribution], distribution


def lmoment_fit(x, distribution):
    """Return parameters based on lmoments.

    Lmomentfit routine derive parameters of a distribution function
    based on lmoments.

    Based on the theory of Hosking and Wallis (1997) Appendix A.

    Parameters
    ----------
    X: 1D array of float
        data series
    distribution: {'gev', 'gpd', 'gumb', 'exp'}
        Short name of distribution function to be fitted.

    Returns
    -------
    params: ndarray of float
        array of distribution parameters
    lambda: 1D array of float
        vector of (nmom) L-moments
    """
    # make sure the timesiries does not contain NaNs
    x = x[~np.isnan(x)]

    # derive first four L-moments from data
    lmom = get_lmom(x, 4)

    # derive parameters of distribution function
    params = _lmomentfit(lmom, distribution)

    return params


def _lmomentfit(lmom, distribution):
    """Return parameters based on lmoments.

    Lmomentfit routine to derive parameters of a distribution function
    based on given lmoments.

    Based on the theory of Hosking and Wallis (1997) Appendix A.

    Parameters
    ----------
    lmom: 1D array of float
        l-moments, derived from data
    distribution: {'gev', 'gpd', 'gumb', 'exp'}
        Short name of distribution function to be fitted.

    Returns
    -------
    params: ndarray of float
        array of distribution parameters
    """
    # l-moment ratios from l-moments
    tau3 = lmom[2] / lmom[1]  # tau3 is in L-SK

    # derive parameters for selected distribution
    if distribution in ["gev", "genextreme"]:
        c1 = 2.0 / (3.0 + tau3) - np.log(2.0) / np.log(3.0)
        k1 = 7.859 * c1 + 2.9554 * (c1**2.0)
        s1 = (lmom[1] * k1) / ((1.0 - 2.0 ** (-k1)) * math.gamma(1.0 + k1))
        m1 = lmom[0] - (s1 / k1) * (1.0 - math.gamma(1.0 + k1))
        params = (k1, m1, s1)
    elif distribution in ["gumb", "gumbel_r"]:
        s1 = lmom[1] / np.log(2.0)
        m1 = lmom[0] - 0.5772 * s1
        params = (m1, s1)
    elif distribution in ["gpd", "genpareto"]:
        k1 = (1 - 3 * tau3) / (1 + tau3)
        s1 = (1 + k1) * (2 + k1) * lmom[1]
        m1 = lmom[0] - (2 + k1) * lmom[1]
        params = (-k1, m1, s1)
    elif distribution in ["exp", "genexpon"]:
        k1 = 1e-8
        s1 = (1 + k1) * (2 + k1) * lmom[1]
        m1 = lmom[0] - (2 + k1) * lmom[1]
        params = (0.0, m1, s1)
    else:
        raise ValueError("Unknown distribution")

    return params


def legendre_shift_poly(n):
    """Shifted Legendre polynomial.

    Based on recurrence relation
        (n + 1)Pn+1 (x) - (1 + 2 n)(2 x - 1)Pn (x) + n Pn-1 (x) = 0

    Given nonnegative integer n, compute the shifted Legendre polynomial P_n.
    Return the result as a vector whose mth element is the coefficient of x^(n+1-m).
    polyval(legendre_shift_poly(n),x) evaluates P_n(x).
    """
    if n == 0:
        pk = 1
    elif n == 1:
        pk = [2, -1]
    else:
        pkm2 = np.zeros(n + 1)
        pkm2[-1] = 1
        pkm1 = np.zeros(n + 1)
        pkm1[-1] = -1
        pkm1[-2] = 2

        for k in range(2, n + 1):
            pk = np.zeros(n + 1)

            for e in range(n - k + 1, n + 1):
                pk[e - 1] = (
                    (4 * k - 2) * pkm1[e]
                    + (1 - 2 * k) * pkm1[e - 1]
                    + (1 - k) * pkm2[e - 1]
                )

            pk[-1] = (1 - 2 * k) * pkm1[-1] + (1 - k) * pkm2[-1]
            pk = pk / k

            if k < n:
                pkm2 = pkm1
                pkm1 = pk

    return pk


def get_lmom(x, nmom=4):
    """Compute L-moments for a data series.

    Based on calculation of probability weighted moments and the coefficient
    of the shifted Legendre polynomial.

    lmom by Kobus N. Bekker, 14-09-2004

    Parameters
    ----------
    x: 1D array of float
        data series
    nmom: int
        number of L-Moments to be computed, by default 4.

    Returns
    -------
    lmom: 1D array of float
        vector of (nmom) L-moments
    """
    n = len(x)
    xs = np.sort(x, axis=0)
    bb = np.zeros(nmom - 1)
    ll = np.zeros(nmom - 1)
    b0 = xs.mean(axis=0)

    for r in range(1, nmom):
        num1 = np.kron(np.ones((r, 1)), np.arange(r + 1, n + 1))
        num2 = np.kron(np.ones((n - r, 1)), np.arange(1, r + 1)).T
        num = np.prod(num1 - num2, axis=0)

        den = np.prod(np.kron(np.ones((1, r)), n) - np.arange(1, r + 1))
        bb[r - 1] = (((num / den) * xs[r:n]).sum()) / n

    B = np.concatenate([np.array([b0]), bb.T])[::-1]

    for i in range(1, nmom):
        spc = np.zeros(len(B) - (i + 1))
        coeff = np.concatenate([spc, legendre_shift_poly(i)])
        ll[i - 1] = np.sum(coeff * B)

    lmom = np.concatenate([np.array([b0]), ll.T])

    return lmom
