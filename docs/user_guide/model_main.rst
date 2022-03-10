.. _model_main:

Working with models in HydroMT  
==============================

HydroMT is commonly used in combination with a model plugin, relevant functions for setting up or adjusting models include: 

* :ref:`building a model <cli_build>`: building a model from scratch.
* :ref:`updating a model <cli_update>`: updating an existing model (e.g. update datafeeds).
* :ref:`clipping a model <cli_clip>`: changing the spatial domain of an existing model (e.g. select subbasins from a larger model).
* :ref:`postprocessing model results <post>`: generating statistics, visualizing results.

.. _SupportedModels: 

Supported models
^^^^^^^^^^^^^^^^

HydroMT currently supports the following models:

* hydromt_wflow_: A framework for distributed rainfall-runoff (wflow_sbm) sediment transport (wflow_sediment) modelling.
* hydromt_delwaq_: A framework for water quality (D-Water Quality) and emissions (D-Emissions) modelling.
* hydromt_sfincs_: A fast 2D hydrodynamic flood model.
* hydromt_fiat_: A flood impact model.

.. _hydromt_wflow: https://deltares.github.io/hydromt_wflow
.. _hydromt_sfincs: https://deltares.github.io/hydromt_sfincs
.. _hydromt_delwaq: https://deltares.github.io/hydromt_delwaq
.. _hydromt_fiat: https://deltares.github.io/hydromt_fiat

.. toctree::
    :hidden:
    
    model_build.rst
    model_update.rst
    model_clip.rst
    model_post.rst