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
def test_peaks(ts_gtsm):
    # testing block maxima
    ts_bm = extremes.get_peaks(ts_gtsm, ev_type='BM', period='3D').load() # default: ev_type='BM', period='year'
    assert all(ts_bm.notnull().sum(dim="time") == np.repeat(int(np.round((ts_gtsm.time[-1] - ts_gtsm.time[0])/pd.Timedelta('3D'))), len(ts_bm["stations"])))
    assert all(ts_bm.max(dim="time") == ts_gtsm.max(dim='time').load())

    # testing POT
    ts_pot = extremes.get_peaks(ts_gtsm, ev_type='POT', qthresh=0.8).load()
    assert all(ts_pot.notnull().sum(dim="time") == [16,17]) #I guess this is a bad test?
    assert all(ts_pot.max(dim="time") == ts_gtsm.max(dim='time'))

def test_fit_extremes(ts_extremes):
    #Fitting BM-Gumbel to Gumbel generated data
    da_params = extremes.fit_extremes(ts_extremes, ev_type='BM', distribution= 'gumb') #check also for 6Mdata
    assert all(da_params['distribution'] == np.repeat('gumb', len(da_params['stations']))) #distribution should be 'gumb'
    assert all(da_params.sel(dparams='shape') == np.repeat(0, len(da_params['stations']))) #Shape parameters by Gumbel should be 0
    #Checking the values of the parameters about    
    for i in range(len(da_params['stations'])):
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='loc'), 100, significant=2)
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='scale'), 25, significant=1)
    del da_params

    #Fitting BM-GEV to Gumbel generated data    
    da_params = extremes.fit_extremes(ts_extremes, ev_type='BM', distribution= 'gev')
    assert all(da_params['distribution'] == np.repeat('gev', len(da_params['stations']))) #distribution should be 'gumb'
    #Checking the values of the parameters about
    for i in range(len(da_params['stations'])):
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='loc'), 100, significant=2)
        np.testing.assert_approx_equal(da_params.isel(stations=i).sel(dparams='scale'), 25, significant=1)
        np.testing.assert_approx_equal(np.abs(da_params.isel(stations=i).sel(dparams='shape')+1), 1, significant=1)
    
    #TODO - also do for PoT (which time series to use?) - should create a time series with daily data and 2 peaks per year

def test_return_values(ts_extremes):
    da_params = extremes.fit_extremes(ts_extremes, ev_type='BM', distribution= 'gumb').load() #check also for 6Mdata
    rps = np.array([5,10,25,50, 100,250,500])
    da_rps = extremes.get_return_value(da_params, rps = rps).load() #By default rps = [2,5,10,25,100,250,500]

    #Shape should match
    assert da_rps.shape == (len(da_params['stations']), len(rps))
    #Value of the 1.5-year return period should be quite close to empirical
    locs = np.abs(1/((ts_extremes*(-1)).rank(dim='time')/(len(ts_extremes["time"])+1))-rps[0]).argmin(dim='time')
    for i in range(len(da_params['stations'])):
        emp_vals = float(ts_extremes.isel(stations=i, time=locs.values[i]))
        gumb_val = float(da_rps.isel(stations=i, rps=0))
        np.testing.assert_approx_equal(emp_vals, gumb_val, significant=2)
        print(f'station {i}: emp_val = {emp_vals} gumb_values = {gumb_val}')

def test_eva():
    



#Functions to test

# eva_idf - intensity-frequency-duration (IDF) table based on block maxima of `da`
# eva_block_maxima - EVA with regular space time dimensions - can be done
# eva_peaks_over_threshold 
