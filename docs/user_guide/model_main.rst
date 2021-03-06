.. _model_main:

Working with models in HydroMT  
==============================

High level functionality
------------------------

HydroMT has the following high-level functionality for setting up models from raw data or adjusting models: 

* :ref:`building a model <model_build>`: building a model from scratch.
* :ref:`updating a model <model_update>`: adding or changing model components of an existing model.
* :ref:`clipping a model <model_clip>`: changing the spatial domain of an existing model (e.g. select subbasins from a larger model).

The building and clipping methods required the user to provide a :ref:`region <region>` of interest. HydroMT provides 
several options to define a region based on a geospatial or hydrographic region.

The exact process of building or updating a model can be configured in a single configuration :ref:`.ini file <model_config>`.
This file describes the full pipeline of model methods and their arguments. The methods vary for the 
different model :ref:`plugins` and are documented for each at their respective documentation websites.

.. _model_interface:

Model data components
---------------------

.. currentmodule:: hydromt

Model data components are data attributes which together define a model instance and are identical for all models. 
Each component represents a specific model component and is parsed to a specific Python data object that should adhere
to certain specifications. An overview is given below.

.. list-table::
   :widths: 20 45 15
   :header-rows: 1

   * - Component
     - Explanation
     - API
   * - Staticmaps
     - Static gridded data
     - :py:attr:`~Model.staticmaps`
       :py:func:`~Model.set_staticmaps`
       :py:func:`~Model.read_staticmaps`
       :py:func:`~Model.write_staticmaps`
   * - Staticgeoms
     - Static vector data
     - :py:attr:`~Model.staticgeoms`
       :py:func:`~Model.set_staticgeoms`
       :py:func:`~Model.read_staticgeoms`
       :py:func:`~Model.write_staticgeoms`
   * - Forcing
     - (Dynamic) forcing data (meteo or hydrological for example)
     - :py:attr:`~Model.forcing`
       :py:func:`~Model.set_forcing`
       :py:func:`~Model.read_forcing`
       :py:func:`~Model.write_forcing`
   * - Results
     - Model output
     - :py:attr:`~Model.results`
       :py:func:`~Model.set_results`
       :py:func:`~Model.read_results`
   * - States
     - Initial model conditions
     - :py:attr:`~Model.states`
       :py:func:`~Model.set_states`
       :py:func:`~Model.read_states`
       :py:func:`~Model.write_states`
   * - Config
     - Settings for the model kernel simulation
     - :py:attr:`~Model.config`
       :py:func:`~Model.set_config`
       :py:func:`~Model.read_config`
       :py:func:`~Model.write_config`


Supported models
----------------

For a list of supported models see the :ref:`plugins` page.

.. toctree::
    :hidden:
    
    model_build.rst
    model_update.rst
    model_clip.rst
    model_config.rst
    model_region.rst
