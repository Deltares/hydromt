"""Tests for the stats/design_events submodule."""

import numpy as np

from hydromt.stats import design_events, extremes


def test_get_peak_hydrographs(ts_extremes):
    ts_extremes = ts_extremes.isel(time=slice(365 * 10))  # smaller sample
    da_peaks = extremes.get_peaks(
        ts_extremes,
        ev_type="BM",
        period="182.625D",  # this returns 20 peaks
    )  # default: ev_type='BM', period='year'
    da = design_events.get_peak_hydrographs(
        ts_extremes, da_peaks, wdw_size=7, n_peaks=20, normalize=False
    ).load()
    assert da.time.shape[0] == 7  # wdw_size=7
    assert (da.argmax("time") == 3).all()  # peak at time=3

    # Testing if maximum values are the same as the top 20
    peaks_1 = da_peaks.sel(stations=1).dropna(dim="time")
    max_station1 = da.sel(stations=1, peak=0, time=0)
    assert max_station1 == np.max(peaks_1)
    del da

    # Testing if normalize values are set to 1
    da = design_events.get_peak_hydrographs(
        ts_extremes, da_peaks, wdw_size=7, normalize=True
    ).load()
    damax_1 = da.sel(stations=1, time=0)
    assert (damax_1 == 1).all()

    # test when number of peaks varies between stations
    # set last 4 peaks of station 2 (index 1) to nan
    da_peaks[dict(stations=1, time=slice(365 * 8, -1))] = np.nan
    da = design_events.get_peak_hydrographs(ts_extremes, da_peaks, wdw_size=7).load()
    assert da.isel(time=3, stations=1).count("peak") == 16
    assert da.isel(time=3, stations=0).count("peak") == 20
