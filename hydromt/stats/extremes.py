"""Functions for extreme value analysis."""
import math as math
from typing import Optional

import dask
import numpy as np
import xarray as xr
from numba import njit
from scipy import stats

__all__ = [
    "eva_idf",
    "eva_block_maxima",
    "eva_peaks_over_threshold",
    "eva",
    "get_peaks",
    "get_return_value",
    "fit_extremes",
]

_RPS = np.array([2, 5, 10, 25, 50, 100, 250, 500])  # years
_DISTS = {
    "POT": ["exp", "gpd"],
    "BM": ["gumb", "gev"],
}

## high level methods


def eva_idf(
    da: xr.DataArray,
    durations: np.ndarray = np.array([1, 2, 3, 6, 12, 24, 36, 48], dtype=int),
    distribution: str = "gumb",
    rps: np.ndarray = _RPS,
    **kwargs,
) -> xr.Dataset:
    """Return IDF based on EVA.

    Return a intensity-frequency-duration (IDF) table based on block
    maxima of `da`.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries data, must have a regular spaced 'time' dimension.
    durations : np.ndarray
        List of durations, provided as multiply of the data time step,
        by default [1, 2, 3, 6, 12, 24, 36, 48]
    distribution : str, optional
        Short name of distribution, by default 'gumb'
    rps : np.ndarray, optional
        Array of return periods, by default [2, 5, 10, 25, 50, 100, 250, 500]
    **kwargs :
        key-word arguments passed to the :py:meth:`eva` method.

    Returns
    -------
    xr.Dataset
        IDF table
    """
    assert np.all(
        np.diff(durations) > 0
    ), "durations should be monotonically increasing"
    dt_max = int(durations[-1])
    da_roll = da.rolling(time=dt_max).construct("duration")
    # get mean intensity for each duration and concat into single dataarray
    da1 = [da_roll.isel(duration=slice(0, d)).mean("duration") for d in durations]
    da1 = xr.concat(da1, dim="duration")
    da1["duration"] = xr.IndexVariable("duration", durations)
    # return
    if "min_dist" not in kwargs:
        kwargs.update(min_dist=dt_max)
    # return eva_block_maxima(da1, distribution=distribution, rps=rps, **kwargs)
    return eva(da1, ev_type="BM", distribution=distribution, rps=rps, **kwargs)


def eva(
    da: xr.DataArray,
    ev_type: str = "BM",
    min_dist: int = 0,
    qthresh: float = 0.9,
    period: str = "365.25D",
    min_sample_size: int = 0,
    distribution: Optional[str] = None,
    rps: np.ndarray = _RPS,
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
    period: str = "365.25D",
    min_dist: int = 0,
    min_sample_size: int = 0,
    distribution: Optional[str] = None,
    rps: np.ndarray = _RPS,
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
    period: str = "365.25D",
    distribution: Optional[str] = None,
    rps: np.ndarray = _RPS,
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
    assert 0 < qthresh < 1.0, 'Quantile "qthresh" should be between (0,1)'
    if ev_type.upper() not in _DISTS.keys():
        raise ValueError(
            f"Unknown ev_type {ev_type.upper()}, select from {_DISTS.keys()}."
        )
    bins = None
    duration = (da["time"][-1] - da["time"][0]).dt.days / 365.2425
    if period in ["year", "quarter", "month"]:
        bins = getattr(da["time"].dt, period).values
        np.unique(bins).size  # FIXME
    else:
        tstart = da.resample(time=period, label="left").first()["time"]
        bins = tstart.reindex_like(da, method="ffill").values.astype(float)
        np.unique(bins).size
    if ev_type.upper() != "BM":
        # TODO - Need to fix periods for POT!
        bins = None
        min_sample_size = 0
        # Add nperiods

    def func(x):
        return local_max_1d(
            x, min_dist=min_dist, bins=bins, min_sample_size=min_sample_size
        )

    duck = dask.array if isinstance(da.data, dask.array.Array) else np
    lmax = duck.apply_along_axis(func, da.get_axis_num("time"), da)
    # apply POT threshold
    peaks = da.where(lmax)
    if ev_type.upper() == "POT":
        peaks = da.where(peaks > da.quantile(qthresh, dim="time"))
    # TODO get extreme rate - this rate is peak rate per period!! not correct
    # da_rate = np.isfinite(peaks).sum("time") / nperiods
    da_rate = np.isfinite(peaks).sum("time") / duration
    peaks = peaks.assign_coords({"extremes_rate": da_rate})
    peaks.name = "peaks"
    return peaks


def get_return_value(da_params: xr.DataArray, rps: np.ndarray = _RPS) -> xr.DataArray:
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

    Returns
    -------
    xr.DataArray
        Return values
    """

    def _return_values_1d(p, r, d, rps=rps):
        if d == "gumb" and len(p) == 3:
            p = p[1:]
        return _get_return_values(p, d, rps=rps, extremes_rate=r)

    assert "dparams" in da_params.dims
    assert "distribution" in da_params.reset_coords()  # coord or variable
    distributions = da_params["distribution"].load()
    assert "extremes_rate" in da_params.reset_coords()  # coord or variable
    extremes_rate = da_params["extremes_rate"].load()

    if da_params.ndim == 1:  # fix case of single dim
        da_params = da_params.expand_dims("index")
    da_rvs = xr.apply_ufunc(
        _return_values_1d,
        da_params.chunk({"dparams": -1}),
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
        distributions = [distribution]
    elif ev_type.upper() not in _DISTS:
        raise ValueError(
            f"Unknown ev_type {ev_type.upper()}, select from {_DISTS.keys()}."
        )

    if "extremes_rate" not in da_peaks.reset_coords():  # coord or variable
        print(
            "Setting extremes_rates to 1.0"
        )  # This would be better with a logger I guess?
        list([d for d in da_peaks.dims if d != "time"])
        # TODO - check how to add this
        # da_peaks.assign_coords({"extremes_rate":
        #       xr.Variable(dims=dim, np.ones(len(dim)))})

    def _fitopt_1d(x, distributions=distributions, criterium=criterium):
        params, d = lmoment_fitopt(x, distributions=distributions, criterium=criterium)
        if len(params) == 2:
            params = np.concatenate([[0], params])
        # trick to include distribution name
        return np.concatenate([params, [distributions.index(d)]])

    if da_peaks.ndim == 1:  # fix case of single dim
        da_peaks = da_peaks.expand_dims("index")
    da_params = xr.apply_ufunc(
        _fitopt_1d,
        da_peaks.chunk({"time": -1}),
        input_core_dims=[["time"]],
        output_core_dims=[["dparams"]],
        dask_gufunc_kwargs=dict(output_sizes={"dparams": 4}),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    # split output
    idist = da_params.isel(dparams=-1).values.astype(int)
    distributions = np.atleast_1d(np.array(distributions)[idist])
    da_params = da_params.isel(dparams=slice(0, -1))
    da_params.name = "parameters"
    # add coordinates
    dims = list([d for d in da_params.dims if d != "dparams"])
    coords = dict(
        dparams=xr.IndexVariable(
            "dparams", ["shape", "loc", "scale"]
        ),  # I think this should have the dimension of duration as well
        distribution=xr.IndexVariable(dims=dims, data=distributions),
    )
    # keep extremes_rate meta data
    coords.update(extremes_rate=da_peaks["extremes_rate"])
    da_params = da_params.assign_coords(coords)
    return da_params.squeeze()


## PEAKS
@njit
def local_max_1d(
    arr: np.ndarray,
    bins: np.ndarray = None,
    min_dist: int = 0,
    min_sample_size: int = 0,
) -> np.ndarray:
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
    out = np.array([bool(0) for _ in range(arr.size)])
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


def get_dist(distribution):
    """Return scipy.stats distribution."""
    _DISTS = {
        "gev": "genextreme",
        "gpd": "genpareto",
        "gumb": "gumbel_r",
        "exp": "genpareto",
    }
    distribution = _DISTS.get(distribution, distribution)
    dist = getattr(stats, distribution, None)
    if dist is None:
        raise ValueError(f'Distribution "{distribution}" not found in scipy.stats.')
    return dist


def get_frozen_dist(params, distribution):
    """Return frozen distribution.

    Returns scipy.stats frozen distribution, i.e.: with set parameters.
    """
    return get_dist(distribution)(*params[:-2], loc=params[-2], scale=params[-1])


## STATS

# TODO add ks and cmv tests
# cvm = stats.cramervonmises(x, frozen_dist.cdf)
# ks = stats.kstest(x, frozen_dist.cdf)


def _aic(x, params, distribution):
    """Return Akaike Information Criterion for a frozen distribution."""
    k = len(params)
    nll = get_frozen_dist(params, distribution).logpdf(x).sum()
    aic = 2 * k - 2 * nll
    return aic


def _aicc(x, params, distribution):
    """Return AICC.

    Return Akaike Information Criterion with correction for small sample size
    for a frozen distribution.
    """
    k = len(params)
    aic = _aic(x, params, distribution)
    aicc = aic + ((2 * k) ** 2 + 2 * k) / (len(x) - k - 1)
    return aicc


def _bic(x, params, distribution):
    """Return Bayesian Information Criterion for a frozen distribution."""
    k = len(params)
    nll = get_frozen_dist(params, distribution).logpdf(x).sum()
    bic = k * np.log(len(x)) - 2 * nll
    return bic


## TRANSFORMATIONS


def _get_return_values(params, distribution, rps=_RPS, extremes_rate=1.0):
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

    # np.random.seed(12456)
    x_sample = np.random.choice(x, size=[nsample, x.size], replace=True)
    xrv = np.apply_along_axis(func, 1, x_sample)

    percentiles = np.array([(1 - alpha) / 2, 1 - (1 - alpha) / 2]) * 100
    return np.percentile(xrv, percentiles, axis=0)


## PLOTS


def plot_return_values(
    x,
    params,
    distribution,
    x_scale="gumbel",
    ax=None,
    color="k",
    a=0,
    alpha=0.9,
    nsample=1000,
    rps=_RPS,
    extremes_rate=1.0,
):
    import matplotlib.pyplot as plt

    rvs_obs = np.sort(x[np.isfinite(x)])
    if distribution != "emperical":
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

    if distribution != "emperical":
        ax.plot(
            xsim, rvs_sim, color=color, ls="--", label=f"{distribution.upper()} fit"
        )

    if alpha is not None and nsample > 0 and distribution != "emperical":
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
    ymax = np.max([np.max(rvs_obs), np.max(rvs_sim), 0])
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


def lmoment_fitopt(x, distributions=["gumb", "gev"], criterium="AIC"):
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
    # tau  = lmom[2]/lmom[1]   # tau  = L-CV
    tau3 = lmom[2] / lmom[1]  # tau3 = L-SK
    # tau4 = lmom[4]/lmom[2]   # tau4 = L-KU

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
        params = (-k1, m1, s1)
    else:
        raise ValueError("Unknow distribution")

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
    xs = np.msort(x)
    bb = np.zeros(nmom - 1)
    ll = np.zeros(nmom - 1)
    b0 = xs.mean(axis=0)

    for r in range(1, nmom):
        Num1 = np.kron(np.ones((r, 1)), np.arange(r + 1, n + 1))
        Num2 = np.kron(np.ones((n - r, 1)), np.arange(1, r + 1)).T
        Num = np.prod(Num1 - Num2, axis=0)

        Den = np.prod(np.kron(np.ones((1, r)), n) - np.arange(1, r + 1))
        bb[r - 1] = (((Num / Den) * xs[r:n]).sum()) / n

    B = np.concatenate([np.array([b0]), bb.T])[::-1]

    for i in range(1, nmom):
        Spc = np.zeros(len(B) - (i + 1))
        Coeff = np.concatenate([Spc, legendre_shift_poly(i)])
        ll[i - 1] = np.sum(Coeff * B)

    lmom = np.concatenate([np.array([b0]), ll.T])

    return lmom
