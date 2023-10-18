"""Tests for the stats/extremes submodule."""
import numpy as np
import pandas as pd
import pytest
import scipy
import xarray as xr

from hydromt.stats import extremes


def test_peaks(ts_extremes):
    # testing block maxima for 6 months time windows
    ts_bm = extremes.get_peaks(
        ts_extremes, ev_type="BM", period="182.625D"
    )  # default: ev_type='BM', period='year'

    # Testing expected number of peaks
    dt_tot = ts_extremes.time[-1] - ts_extremes.time[0]
    nb_peaks = int(np.round(dt_tot / pd.Timedelta("182.625D")))
    assert all(ts_bm.notnull().sum(dim="time") == nb_peaks)
    # Testing expected maximum value
    assert all(ts_bm.max(dim="time") == ts_extremes.max(dim="time"))

    # testing for POT
    ts_pot = extremes.get_peaks(ts_extremes, ev_type="POT", qthresh=0.996)
    # Testing expected number of peaks
    assert all(ts_pot.notnull().sum(dim="time") == [145, 146])
    # Testing expected maximum value
    assert all(ts_pot.max(dim="time") == ts_extremes.max(dim="time"))

    # test errors
    with pytest.raises(ValueError, match="Quantile 'qthresh' should be "):
        extremes.get_peaks(ts_extremes, ev_type="POT", qthresh=1.1)
    with pytest.raises(ValueError, match="Input array should have a 'time' dimension"):
        extremes.get_peaks(ts_extremes.rename({"time": "test"}), ev_type="BM")
    with pytest.raises(ValueError, match="Unknown ev_type"):
        extremes.get_peaks(ts_extremes, ev_type="unknown")


def test_fit_extremes(ts_extremes):
    from scipy.stats import genextreme, genpareto

    # Fitting BM-Gumbel to Gumbel generated data
    bm_peaks = extremes.get_peaks(ts_extremes, ev_type="BM", period="182.625D")
    da_params = extremes.fit_extremes(bm_peaks, ev_type="BM", distribution="gumb")
    # Testing if parameters is 'gumb' as expected
    assert all(da_params["distribution"] == "gumb")
    # Testing if shape parameter is 0 as expected
    assert all(da_params.sel(dparams="shape") == 0)
    # testing if we get the value from the function
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="loc").values, [106, 104], decimal=0
    )
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="scale").values, [17, 16], decimal=0
    )

    # Checking the values of the parameters based on Method of Moments estimators
    # for Gumbel
    for i in range(len(da_params["stations"])):
        mean = bm_peaks.isel(stations=i).mean().values
        std = bm_peaks.isel(stations=i).std().values
        # MoM estimators for Gumbel
        mom_beta = std * np.sqrt(6) / np.pi
        mom_alpha = mean - 0.5772 * mom_beta
        # Testing if about equal
        np.testing.assert_approx_equal(
            da_params.isel(stations=i).sel(dparams="loc"), mom_alpha, significant=2
        )
        np.testing.assert_approx_equal(
            da_params.isel(stations=i).sel(dparams="scale"), mom_beta, significant=1
        )
    del da_params

    # Fitting BM-GEV to Gumbel generated data
    da_params = extremes.fit_extremes(bm_peaks, ev_type="BM", distribution="gev")
    # Testing if parameters is 'gev' as expected
    assert all(da_params["distribution"] == "gev")
    # testing if we get the value from the function
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="loc").values, [103.9, 101.7], decimal=1
    )
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="scale").values, [11.05, 7.43], decimal=1
    )
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="shape").values, [-0.347, -0.503], decimal=2
    )
    # Checking the values of the parameters appx based on genextreme
    for i in range(len(da_params["stations"])):
        # Tests for genextreme too sensitive for scale and shape and are failing now...
        da_param_loc = float(da_params.isel(stations=i).sel(dparams="loc").values)
        gen_loc = genextreme.fit(
            bm_peaks.isel(stations=i).to_series().dropna().values, floc=da_param_loc
        )[1]
        np.testing.assert_approx_equal(
            da_params.isel(stations=i).sel(dparams="loc"), gen_loc, significant=2
        )
    del da_params

    # Fitting POT-GPD to Gumbel generated data
    pot_peaks = extremes.get_peaks(ts_extremes, ev_type="POT", qthresh=0.996)
    da_params = extremes.fit_extremes(pot_peaks, ev_type="POT", distribution="gpd")
    # Testing if parameters is 'gpd' as expected
    assert all(da_params["distribution"] == "gpd")
    # testing if we get the value from the function
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="loc").values, [96.2, 95.65], decimal=1
    )
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="scale").values, [41.30, 29.87], decimal=1
    )
    np.testing.assert_array_almost_equal(
        da_params.sel(dparams="shape").values, [-0.365, -0.098], decimal=2
    )
    # Checking the values of the parameters appx based on genpareto
    for i in range(len(da_params["stations"])):
        # Testing values based on genpareto
        data = pot_peaks.isel(stations=i).to_series().dropna().values
        floc = float(da_params.isel(stations=i).sel(dparams="loc").values)
        gen_loc = genpareto.fit(data, floc=floc)[1]
        gen_shape = genpareto.fit(data, floc=floc)[0]
        gen_scale = genpareto.fit(data, floc=floc)[2]

        np.testing.assert_approx_equal(
            da_params.isel(stations=i).sel(dparams="loc"), gen_loc, significant=2
        )
        np.testing.assert_approx_equal(
            da_params.isel(stations=i).sel(dparams="shape"), gen_shape, significant=1
        )
        np.testing.assert_approx_equal(
            da_params.isel(stations=i).sel(dparams="scale"), gen_scale, significant=1
        )


def test_return_values(ts_extremes):
    from scipy.stats import genpareto, gumbel_r

    rps = np.array([1.5, 5, 10, 25, 50, 100, 250, 500])

    # Block Maxima - GUMBEL
    bm_peaks = extremes.get_peaks(ts_extremes, ev_type="BM", period="182.625D")
    da_params = extremes.fit_extremes(bm_peaks, ev_type="BM", distribution="gumb")
    da_rps = extremes.get_return_value(da_params, rps=rps)
    # Shape of da_rps should match
    assert da_rps.shape == (len(da_params["stations"]), len(rps))
    # Testing if values are the same as from the function
    out_rps = np.array([121.6, 144.8, 157.2, 173.2, 185.2, 197.2, 213.0, 225.0])
    np.testing.assert_array_almost_equal(
        da_rps.sel(stations=1).values, out_rps, decimal=0
    )
    # Value should be quite close to scipy fits with similar parameters
    for i in range(len(da_params["stations"])):
        gumb_loc = float(da_params.isel(stations=i)[1])
        gumb_scale = float(da_params.isel(stations=i)[2])
        vals = gumbel_r.isf(
            1 / (rps * float(bm_peaks["extremes_rate"].isel(stations=i))),
            loc=gumb_loc,
            scale=gumb_scale,
        )
        assert np.allclose(da_rps.isel(stations=i).to_numpy(), vals)
    del da_rps, da_params

    # Peaks Over Threshold - GPD
    pot_peaks = extremes.get_peaks(ts_extremes, ev_type="POT", qthresh=0.996)
    da_params = extremes.fit_extremes(pot_peaks, ev_type="POT", distribution="gpd")
    da_rps = extremes.get_return_value(da_params, rps=rps)
    # Shape of da_rps should match
    assert da_rps.shape == (len(da_params["stations"]), len(rps))
    # Testing if values are the same as from the function
    out_rps = np.array([124.1, 154.4, 166.6, 178.7, 185.5, 190.8, 196.0, 198.9])
    np.testing.assert_array_almost_equal(
        da_rps.sel(stations=1).values, out_rps, decimal=0
    )
    # Value should be quite close to scipy fits with similar parameters
    for i in range(len(da_params["stations"])):
        gen_loc = float(da_params.isel(stations=i)[1])
        gen_scale = float(da_params.isel(stations=i)[2])
        gen_c = float(da_params.isel(stations=i)[0])
        vals = genpareto.isf(
            1 / (rps * float(pot_peaks["extremes_rate"].isel(stations=i))),
            c=gen_c,
            loc=gen_loc,
            scale=gen_scale,
        )
        assert np.allclose(da_rps.isel(stations=i).to_numpy(), vals)

    # test errors get_return_value
    with pytest.raises(ValueError, match="rps should be a list or numpy array"):
        extremes.get_return_value(da_params, rps=1)
    with pytest.raises(ValueError, match="da_params should have a 'dparams' dimension"):
        extremes.get_return_value(da_params.rename({"dparams": "test"}), rps=rps)
    with pytest.raises(ValueError, match="da_params should have a 'distribution' "):
        extremes.get_return_value(da_params.drop("distribution"), rps=rps)
    with pytest.raises(ValueError, match="extremes_rate should be a scalar or "):
        extremes.get_return_value(
            da_params.drop("extremes_rate"), rps=rps, extremes_rate=np.array([1])
        )
    with pytest.raises(ValueError, match="distribution should be a string or list"):
        extremes.fit_extremes(pot_peaks, ev_type="POT", distribution=1)
    with pytest.raises(ValueError, match="Unknown ev_type"):
        extremes.fit_extremes(pot_peaks, ev_type="unknown")


def test_eva(ts_extremes):
    # Test for 6M BM
    bm_eva = extremes.eva(
        ts_extremes, ev_type="BM", period="quarter", distribution="gumb"
    )
    bm_peaks = extremes.get_peaks(ts_extremes, ev_type="BM", period="quarter")
    da_params = extremes.fit_extremes(bm_peaks, ev_type="BM", distribution="gumb")
    da_rps = extremes.get_return_value(da_params)
    bm_test = xr.merge([bm_peaks, da_params, da_rps])

    xr.testing.assert_equal(bm_eva, bm_test)
    del da_params, da_rps

    # test eva_block_maxima
    bm_eva2 = extremes.eva_block_maxima(
        ts_extremes, period="quarter", distribution="gumb"
    )
    xr.testing.assert_equal(bm_eva, bm_eva2)

    # Test fot POT
    pot_eva = extremes.eva(
        ts_extremes, ev_type="POT", qthresh=0.996, distribution="gpd"
    )
    # Peaks Over Threshold - GPD
    pot_peaks = extremes.get_peaks(ts_extremes, ev_type="POT", qthresh=0.996)
    da_params = extremes.fit_extremes(pot_peaks, ev_type="POT", distribution="gpd")
    da_rps = extremes.get_return_value(
        da_params.drop("extremes_rate"), extremes_rate=pot_peaks.extremes_rate
    )
    pot_test = xr.merge([pot_peaks, da_params, da_rps])
    xr.testing.assert_equal(pot_test, pot_eva)

    pot_eva2 = extremes.eva_peaks_over_threshold(
        ts_extremes, qthresh=0.996, distribution="gpd"
    )
    xr.testing.assert_equal(pot_eva, pot_eva2)

    # test plot
    # not sure how to work with assertions here but at least check that it runs
    extremes.plot_return_values(
        x=pot_peaks.isel(stations=0).values,
        params=da_params.isel(stations=0).values,
        distribution=da_params["distribution"].isel(stations=0).item(),
    )


def test_peaks_eva_3d_dask():
    # create a test dataset with dask arrays
    data = xr.DataArray(
        np.random.rand(365, 10, 10),
        dims=("time", "y", "x"),
        coords={
            "time": pd.date_range("2000-01-01", periods=365),
            "y": np.arange(10),
            "x": np.arange(10),
        },
    ).chunk({"time": -1, "y": 5, "x": 5})
    # calculate peaks using get_peaks method
    peaks = extremes.get_peaks(data, ev_type="POT", qthresh=0.95)
    # check that the output has the expected values
    thresh = data.quantile(0.95, dim="time")
    assert np.all(np.logical_or(peaks >= thresh, np.isnan(peaks)))
    # check peaks is dask array
    assert peaks.chunks is not None

    # fit extreme value distribution using fit_extremes method
    params = extremes.fit_extremes(peaks, ev_type="POT")
    # test that the output has the expected dimensions
    assert params.chunks is not None

    # get return values using get_return_value method
    rps = np.array([1.5, 5, 10, 25, 50, 100, 250, 500])
    rvs = extremes.get_return_value(
        params.drop("extremes_rate"), rps=rps, extremes_rate=1
    )
    assert rvs.chunks is not None

    # trigger the computation of the dask arrays
    assert peaks.load().shape == (365, 10, 10)
    assert params.load().shape == (10, 10, 3)
    assert rvs.load().shape == (10, 10, len(rps))


def test_aic_bic_aicc():
    # create a gumb test dataset with scipy seed
    np.random.seed(1234)
    x = scipy.stats.gumbel_r.rvs(loc=100, scale=20, size=100)

    params = extremes.lmoment_fit(x, distribution="gumb")
    with pytest.raises(ValueError, match="Unknown distribution"):
        extremes.lmoment_fit(x, distribution="unknown")

    # calculate AIC, BIC, and AICC
    aic = extremes._aic(x, params, distribution="gumb")
    bic = extremes._bic(x, params, distribution="gumb")
    aicc = extremes._aicc(x, params, distribution="gumb")

    # test that the output has the expected values
    assert np.isclose(aic, 914.299023345937)
    assert np.isclose(aicc, 914.5052089129473)
    assert np.isclose(bic, 919.5093637179132)


def test_get_return_periods():
    x = np.arange(1, 11)
    rps = extremes._get_return_periods(x)
    assert rps[-1] == 11
    rps = extremes._get_return_periods(x, extremes_rate=11)
    assert rps[-1] == 1


def test_lmoment_ci():
    np.random.seed(1234)
    rps = np.array([1.5, 5, 10, 25, 50, 100, 250, 500])
    x = scipy.stats.gumbel_r.rvs(loc=100, scale=20, size=15)
    ci = extremes.lmoment_ci(x, distribution="gumb", nsample=10)
    assert ci.shape == (2, rps.size)
    assert np.all(ci[0, :] < ci[1, :])
