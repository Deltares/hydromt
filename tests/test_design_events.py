"""Tests for the stats/design_events submodule."""

import numpy as np

from hydromt.stats import design_events, extremes


def test_get_peak_hydrographs(ts_extremes):
    da_peaks = extremes.get_peaks(
        ts_extremes, ev_type="BM", period="182.625D"
    ).load()  # default: ev_type='BM', period='year'
    da = design_events.get_peak_hydrographs(
        ts_extremes, da_peaks, wdw_size=7, n_peaks=20, normalize=False
    ).load()

    # Testing if wdw_size argument is respected
    np.testing.assert_equal(da.time.shape[0], 7)

    # Testing if maximum values are the same as the top 20
    peaks_1 = da_peaks.sel(stations=1).dropna(dim="time")
    max_station1 = da.sel(stations=1, peak=0, time=0).values
    np.testing.assert_array_equal(max_station1, np.max(peaks_1.values))
    del da

    # Testing if normalize values are set to 1
    n_peaks = 20
    da = design_events.get_peak_hydrographs(
        ts_extremes, da_peaks, wdw_size=7, n_peaks=n_peaks, normalize=True
    ).load()
    damax_1 = da.sel(stations=1, time=0).values
    np.testing.assert_array_equal(damax_1, np.repeat(1.0, repeats=n_peaks))


def test_get_hyetograph(ts_extremes):
    ds_idf = extremes.eva_idf(ts_extremes.isel(stations=0), distribution="gumb").load()
    # testing function for two different durations
    for dt in [1, 6]:
        da = design_events.get_hyetograph(ds_idf, dt=dt, length=7).load()
        # Testing if length of event is equal to length
        np.testing.assert_equal(da.time.shape[0], 7)
        # Testing is max value equal to ds_idf
        max_hyeto_100 = da.sel(rps=100)["return_values"].max()
        idf_100 = ds_idf.sel(rps=100, duration=dt)["return_values"].values
        np.testing.assert_equal(max_hyeto_100, idf_100)
