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
    nse = skillstats.nashsutcliffe().values.round()

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
    mod = SfincsModel()
    mod.read()
    mod.plot_basemap()

