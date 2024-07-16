.. _stat:

Statistical methods
===================

.. _skil_stats:

Skill statistics
----------------

HydroMT provides different functions to apply :ref: model skill statistics <statistics> to compare model results with observations.
The following statistics are available:

- Absolute and percentual bias
- Nash-Sutcliffe model Efficiency (NSE) and log Nash-Sutcliffe model Efficiency (log-NSE)
- Various versions of the Kling-Gupta model Efficiency (KGE)
- Coefficient of determination (R-squared)
- Mean Squared Error (MSE) and Root Mean Squared Error  (RMSE)

Example application
^^^^^^^^^^^^^^^^^^^

As HydroMT provides methods to easily read the model results, applying a skill statistic just takes a few lines of code and can be
applied directly across all observation locations in your model.

.. code-block:: console

    from hydromt.stats import nashsutcliffe
    from hydromt_wflow import WflowModel
    import xarray as xr
    # read model results
    # NOTE: the name of the results depends on the wflow run configuration (toml file)
    mod = WflowModel(root=r'/path/to/wflow_model/root', mode='r')
    sim = mod.results['Q_gauges_grdc']
    # read observations
    obs = xr.open_dataset(r'/path/to/grdc_obs.nc')
    # calculate skill statistic
    nse = nashsutcliffe(sim, obs)
