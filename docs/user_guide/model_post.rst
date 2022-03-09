.. _post:

Postprocessing model results
============================

.. _stat:

Statistics
----------

HydroMT provides different functions to apply :ref:`Statistics and performance metrics <Statistics>` to the model
results. The following analysis fields are addressed:

- Bias analysis
- Model efficiency analysis
- Correlation analysis
- Determination of the efficiency of two time series
- Determination of the determination of two time series
- Determination of the (root) mean squared error

**Example statistics function**

.. code-block:: console

    from hydromt.stats import skills as skillstats
    from hydromt_wflow import WflowModel
    import xarray as xr
    # read model results
    # NOTE: the name of the results depends on the wflow run configuration (toml file)
    mod = WflowModel(root=r'/path/to/wflow_model/root', mode='r')
    sim = mod.results['Q_gauges_grdc']  
    # read observations
    obs = xr.open_dataset(r'/path/to/grdc_obs.nc')
    # calculate skill statistic
    nse = skillstats.nashsutcliffe(sim, obs)

.. _plot:

Visualization
-------------
.. warning::

    Not all plugins provide a built-in plot function yet. Please check the documentation of the respective
    :ref:`plugin<plugins>` for more information on whether specific plot functions are available.

Specific plugins such as the plugin for *SFINCS* offer the possibility to use HydroMT plot functions. For other plugins
examples exist how to use existing Python packages for plotting.

**Example plot function**

.. code-block:: console

    from hydromt_sfincs import SfincsModel
    mod = SfincsModel(root=r'/path/to/sfincs_model/root', mode='r')
    mod.plot_basemap()

