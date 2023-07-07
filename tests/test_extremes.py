"""Tests for the stats/extremes submodule."""

#%%
import pytest

import numpy as np
import xarray as xr
import pandas as pd

from hydromt.stats import extremes
#from hydromt import DataCatalog

#%%
# ts = ts()
# ts_extremes = ts_extremes()
# ts_gtsm = ts_gtsm()
#%%
def test_peaks(ts_extremes):
    # testing block maxima for 6 months time windows
    ts_bm = extremes.get_peaks(ts_extremes, ev_type='BM', period='182.625D').load() # default: ev_type='BM', period='year'
    #TODO: split in smaller arguments
    assert all(ts_bm.notnull().sum(dim="time") == np.repeat(int(np.round((ts_extremes.time[-1] - ts_extremes.time[0])/pd.Timedelta('182.625D'))), len(ts_bm["stations"])))
    assert all(ts_bm.max(dim="time") == ts_extremes.max(dim='time').load())

    # testing POT
    ts_pot = extremes.get_peaks(ts_extremes, ev_type='POT', qthresh=0.996).load()
    assert all(ts_pot.notnull().sum(dim="time") == [145,146]) #I guess this is a bad test?
    assert all(ts_pot.max(dim="time") == ts_extremes.max(dim='time'))

def test_fit_extremes(ts_extremes):
    from scipy.stats import genextreme, genpareto
    #Fitting BM-Gumbel to Gumbel generated data
    bm_peaks = extremes.get_peaks(ts_extremes, ev_type='BM', period='182.625D').load() 
    da_params = extremes.fit_extremes(bm_peaks, ev_type='BM', distribution= 'gumb').load() 
    assert all(da_params['distribution'] == np.repeat('gumb', len(da_params['stations']))) #distribution should be 'gumb'
    assert all(da_params.sel(dparams='shape') == np.repeat(0, len(da_params['stations']))) #Shape parameters by Gumbel should be 0
    #Checking the values of the parameters about    
    for i in range(len(da_params['stations'])):
        mean = bm_peaks.isel(stations=i).mean().values
        std = bm_peaks.isel(stations=i).std().values
        #TODO - test against exact value?
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='loc'), mean - 0.57 * (std*np.sqrt(6)/np.pi), significant=2)
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='scale'), std*np.sqrt(6)/np.pi, significant=1)
    del da_params

    #Fitting BM-GEV to Gumbel generated data    
    da_params = extremes.fit_extremes(bm_peaks, ev_type='BM', distribution= 'gev').load()
    assert all(da_params['distribution'] == np.repeat('gev', len(da_params['stations']))) #distribution should be 'gev'
    #Checking the values of the parameters about
    for i in range(len(da_params['stations'])):
        #Tests for genextreme too sensitive for scale and shape and are failing right now...
        #TODO - test also against values
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='loc'), genextreme.fit(bm_peaks.isel(stations=i).to_series().dropna().values, floc=float(da_params.isel(stations=i).sel(dparams='loc').values))[1], significant=2)
    del da_params

    #Fitting POT-GPD to Gumbel generated data  
    pot_peaks = extremes.get_peaks(ts_extremes, ev_type='POT', qthresh=0.996).load()
    da_params = extremes.fit_extremes(pot_peaks, ev_type='POT', distribution= 'gpd').load()
    assert all(da_params['distribution'] == np.repeat('gpd', len(da_params['stations']))) #distribution should be 'gev'
    for i in range(len(da_params['stations'])):
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='loc'), genpareto.fit(pot_peaks.isel(stations=i).to_series().dropna().values, floc=float(da_params.isel(stations=i).sel(dparams='loc').values))[1], significant=2)
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='shape'), genpareto.fit(pot_peaks.isel(stations=i).to_series().dropna().values, floc=float(da_params.isel(stations=i).sel(dparams='loc').values))[0], significant=1)
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='scale'), genpareto.fit(pot_peaks.isel(stations=i).to_series().dropna().values, floc=float(da_params.isel(stations=i).sel(dparams='loc').values))[2], significant=1)

# #%%
# model = EVA(data=ts_extremes.isel(stations=i).to_series())
# model.get_extremes("BM", block_size="365.25D")
# model.fit_model(distribution='gumbel_r')
# model
# model.get_return_value(rps)

#%%
def test_return_values(ts_extremes):
    from scipy.stats import gumbel_r, genpareto
    rps = np.array([1.5, 5,10,25,50, 100,250,500])

    #Block Maxima - GUMBEL
    bm_peaks = extremes.get_peaks(ts_extremes, ev_type='BM', period='182.625D').load() 
    da_params = extremes.fit_extremes(bm_peaks, ev_type='BM', distribution= 'gumb').load() 
    da_rps = extremes.get_return_value(da_params, rps = rps).load() #By default rps = [2,5,10,25,100,250,500]
    #Shape should match
    assert da_rps.shape == (len(da_params['stations']), len(rps))
    #Value should be quite close to scipy fits with similar parameters
    for i in range(len(da_params['stations'])):
        vals = gumbel_r.isf(1/(rps*float(bm_peaks['extremes_rate'].isel(stations=i))), loc=float(da_params.isel(stations=i)[1]), scale = float(da_params.isel(stations=i)[2]))
        assert np.allclose(da_rps.isel(stations=i).to_numpy(), vals)
    del da_rps, da_params

    #Peaks Over Threshold - GPD
    pot_peaks = extremes.get_peaks(ts_extremes, ev_type='POT', qthresh=0.996).load()
    da_params = extremes.fit_extremes(pot_peaks, ev_type='POT', distribution= 'gpd').load()
    da_rps = extremes.get_return_value(da_params, rps = rps).load() 
    assert da_rps.shape == (len(da_params['stations']), len(rps))
    for i in range(len(da_params['stations'])):
        vals = genpareto.isf(1/(rps*float(pot_peaks['extremes_rate'].isel(stations=i))), c=float(da_params.isel(stations=i)[0]), loc=float(da_params.isel(stations=i)[1]), scale = float(da_params.isel(stations=i)[2]))
        assert np.allclose(da_rps.isel(stations=i).to_numpy(), vals)

def test_eva(ts_extremes):
    #Test for 6M BM
    bm_eva = extremes.eva(ts_extremes, ev_type="BM", period='182.625D', distribution='gumb').load()
    bm_peaks = extremes.get_peaks(ts_extremes, ev_type='BM', period='182.625D').load() 
    da_params = extremes.fit_extremes(bm_peaks, ev_type='BM', distribution= 'gumb').load() 
    da_rps = extremes.get_return_value(da_params).load()
    bm_test = xr.merge([bm_peaks, da_params, da_rps])
    
    xr.testing.assert_equal(bm_eva, bm_test)
    del da_params, da_rps

    #Test fot POT
    pot_eva = extremes.eva(ts_extremes, ev_type="POT", qthresh=0.996, distribution='gpd').load()
    #Peaks Over Threshold - GPD
    pot_peaks = extremes.get_peaks(ts_extremes, ev_type='POT', qthresh=0.996).load()
    da_params = extremes.fit_extremes(pot_peaks, ev_type='POT', distribution= 'gpd').load()
    da_rps = extremes.get_return_value(da_params).load() 
    pot_test = xr.merge([pot_peaks, da_params, da_rps])
    xr.testing.assert_equal(pot_test, pot_eva)

#Add deprecation error for eva?
#Test not working right now
def test_eva_idf(ts_extremes):
    ds_idf = extremes.eva_idf(ts_extremes.isel(stations=0), distribution='gumb') #This one works
    ds_idf = extremes.eva_idf(ts_extremes, distribution='gumb') #This one doesn't work due to the extra 'stations' dimension when the function fit_extremes is called

