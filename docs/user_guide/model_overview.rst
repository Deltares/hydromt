.. _model_overview:

Overview
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
different model classes and :ref:`plugins`, as documented in this documentation or for each plugin documentation websites.

.. _model_interface:

Since v0.5.9, HydroMT can support both generalized model types (for example gridded, lumped or mesh models) and specific model types (plugins). 

Generalized model class implementation
--------------------------------------
The model API from HydroMT allows to build a model from scratch for different model concepts. This implementation is flexible such that users can create a model instance that matches
their need. By default, the model components are returned and read from standard formats, as documented in the :ref:`API reference <api_reference>`. As of version 0.5.9, the 
grid model (distributed model), lumped model (e.g. semi-distributed, bucket models), mesh model (e.g. unstructured models) have been implemented. Other model classes such as network models will follow in future versions.


Specific model class implementation
-----------------------------------

For a list of supported models see the :ref:`plugins` page.


Model data components
---------------------

.. currentmodule:: hydromt

Model data components are data attributes which together define a model instance and are identical for all models. 
Each component represents a specific model component and is parsed to a specific Python data object that should adhere
to certain specifications. These specification are class dependent. An overview is given below.

The table below lists model components common to all model classes

.. list-table::
   :widths: 20 45 15
   :header-rows: 1

   * - Component
     - Explanation
     - API
   * - maps
     - Auxiliary data maps
     - | :py:attr:`~Model.maps`
       | :py:func:`~Model.set_maps`
       | :py:func:`~Model.read_maps`
       | :py:func:`~Model.write_maps`
   * - geoms
     - Static vector data
     - | :py:attr:`~Model.geoms`
       | :py:func:`~Model.set_geoms`
       | :py:func:`~Model.read_geoms`
       | :py:func:`~Model.write_geoms`
   * - forcing
     - (Dynamic) forcing data (meteo or hydrological for example)
     - | :py:attr:`~Model.forcing`
       | :py:func:`~Model.set_forcing`
       | :py:func:`~Model.read_forcing`
       | :py:func:`~Model.write_forcing`
   * - results
     - Model output
     - | :py:attr:`~Model.results`
       | :py:func:`~Model.set_results`
       | :py:func:`~Model.read_results`
   * - states
     - Initial model conditions
     - | :py:attr:`~Model.states`
       | :py:func:`~Model.set_states`
       | :py:func:`~Model.read_states`
       | :py:func:`~Model.write_states`
   * - config
     - Settings for the model kernel simulation or model class
     - | :py:attr:`~Model.config`
       | :py:func:`~Model.set_config`
       | :py:func:`~Model.read_config`
       | :py:func:`~Model.write_config`


For each generalized model class, the respective components exist:

.. list-table::
   :widths: 20 20 25 15
   :header-rows: 1

   * - Component
     - Explanation
     - Model class
     - API
   * - grid
     - Static gridded data
     - GridModel
     - | :py:attr:`~GridModel.grid` 
       | :py:func:`~GridModel.set_grid`
       | :py:func:`~GridModel.read_grid`
       | :py:func:`~GridModel.write_grid`
   * - response_units
     - Static lumped data over the response_units  
     - LumpedModel
     - | :py:attr:`~LumpedModel.response_units`
       | :py:func:`~LumpedModel.set_response_units`
       | :py:func:`~LumpedModel.read_response_units`
       | :py:func:`~LumpedModel.write_response_units`
   * - mesh
     - Static mesh data
     - MeshModel 
     - | :py:attr:`~MeshModel.mesh` 
       | :py:func:`~MeshModel.set_mesh`
       | :py:func:`~MeshModel.read_mesh`
       | :py:func:`~MeshModel.write_mesh`
       


.. NOTE::

    Prior to v0.5.9, the following model components were available. They have been deprecated and will no longer be supported in future versions

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

