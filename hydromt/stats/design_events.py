import numpy as np
import xarray as xr
import math as math
from typing import Optional
from numba import njit

__all__ = [
    "get_peak_hydrographs",
    "get_hyetograph",
]


def get_peak_hydrographs(
    da: xr.DataArray,
    da_peaks: xr.DataArray,
    wdw_size: int,
    n_peaks: Optional[int] = None,
    normalize: bool = True,
) -> xr.DataArray:
    """Returns a hydrograph of `wdw_size` length around each peak
    in `da_peaks` with a max value or 1 at the peak. The mean hydrograph can be derived
    by applying stastics along the 'peak' output dimension.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries data, must have a regular spaced 'time' dimension.
    da_peaks : xr.DataArray
        Timeseries data with only peak values, all other values are set to NaN
    wdw_size : int
        Length of hydrographs measured in the time series time step.
    n_peaks : int, optional
        N largest peaks to return. If None (default) all peaks are returned.
    normalize : bool, optional
        If True (default) return peak hydrographs normalized by peak value.

    Returns
    -------
    xr.DataArray
        Hydrographs with new 'peak' and 'dt' dimensions.
    """
    assert da.shape == da_peaks.shape, "da and da_peaks must have identical shape"
    if da_peaks.dtype != "bool":
        da_peaks = np.isfinite(da_peaks)
    if n_peaks is None:  # n_peaks required for output dimensions
        n_peaks = da_peaks.sum("time").max().compute().item()

    # temp method with arguments set
    def _func(ts, peaks, wdw_size=wdw_size, n_peaks=n_peaks, normalize=normalize):
        return hydrograph_1d(ts, peaks, wdw_size, n_peaks, normalize)

    if da.ndim == 1:  # fix case with single dim
        da = da.expand_dims("index")
        da_peaks = da_peaks.expand_dims("index")
    da_shape = xr.apply_ufunc(
        _func,
        da.chunk({"time": -1}),
        da_peaks.chunk({"time": -1}),
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["peak", "dt"]],
        dask_gufunc_kwargs=dict(output_sizes={"peak": n_peaks, "dt": wdw_size}),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).rename({"dt": "time"})
    # set time coordinate
    t0 = int(np.floor(wdw_size / 2))
    da_shape["time"] = xr.IndexVariable("time", np.arange(-t0, t0 + wdw_size % 2))
    return da_shape.squeeze()


def get_hyetograph(da_idf: xr.DataArray, dt: float, length: int) -> xr.DataArray:
    """Returns design storm hyetograph based on intensity-frequency-duration (IDF) table.

    The input `da_idf` can be obtained as the output of the :py:meth:`eva_idf`.
    Note: here we use the precipitation intensitity and not the depth as input!

    Parameters
    ----------
    da_idf : xr.DataArray
        IDF data, should contain a 'duration' dimension
    dt : float
        Time-step for output hyetograph, same unit as IDF duration.
    length : int
        Number of time-step intervals in design storms.

    Returns
    -------
    xr.DataArray
        Design storm hyetograph
    """

    durations = da_idf["duration"]
    assert np.all(np.diff(durations) > 0)
    assert dt >= durations[0]

    t = np.arange(0, durations[-1] + dt, dt)
    alt_order = np.append(np.arange(1, length, 2)[::-1], np.arange(0, length, 2))
    # get cummulative precip depth
    pdepth = (da_idf * durations).reset_coords(drop=True).rename({"duration": "time"})
    # interpolate to dt temporal resolution
    pstep = pdepth.interp(time=t).fillna(0).diff("time") / dt
    # reorder using alternating blocks method
    pevent = pstep.isel(time=slice(0, length)).isel(time=alt_order)
    # set time coordinate
    t0 = int(np.ceil((length + 1) / 2))
    pevent["time"] = xr.IndexVariable("time", (t[1 : length + 1] - t0))
    pevent.attrs.update(**da_idf.attrs)
    return pevent


@njit
def hydrograph_1d(
    ts: np.ndarray,
    peaks: np.ndarray,
    wdw_size: int,
    n_peaks: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Returns 2D array of shape (`n_peaks`, `wdw_size`) with normalized hydrographs
    from time series `ts`

    Parameters
    ----------
    ts : np.ndarray of float
        1D array with constant spaced time series
    peaks : np.ndarray of bool
        1D array with constant spaced time series, True where peaks
    wdw_size : int
        Size of hydrograph in unit of time series time step
    n_peaks : int, optional
        N largest peaks to return. If None (default) all peaks are returned.
    normalize : bool, optional
        If True (default) return peak hydrographs normalized by peak value.

    Returns
    -------
    np.ndarray
        normalized hydrographs
    """
    assert ts.shape == peaks.shape, "the shapes of ts and peaks mismatch"
    idxs = np.where(peaks)[0]
    n0 = idxs.size if n_peaks is None else int(n_peaks)
    out = np.full((n0, wdw_size), np.nan, ts.dtype)
    seq = np.argsort(ts[idxs])[::-1]  # sort from large to small
    n = ts.size
    d0 = int(np.floor(wdw_size / 2))
    d1 = wdw_size - d0
    for i in range(n0):
        idx = idxs[seq[i]]
        idx0 = idx - d0
        idx1 = idx + d1
        s = slice(max(0, idx0), min(n + 1, idx1))
        s1 = slice(max(0, -idx0), n - idx0 if idx1 > n else idx1 - idx0)
        out[i, s1] = ts[s] / ts[idx] if normalize else ts[s]
    return out
