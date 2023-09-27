.. _model_main:

Overview models
===============

High level functionality
------------------------

HydroMT has the following high-level functionality for setting up models from raw data or adjusting models:

* :ref:`building a model <model_build>`: building a model from scratch.
* :ref:`updating a model <model_update>`: adding or changing model components of an existing model.
* :ref:`clipping a model <model_clip>`: changing the spatial domain of an existing model (e.g. select subbasins from a larger model).

The building and clipping methods required the user to provide a :ref:`region <region>` of interest. HydroMT provides
several options to define a region based on a geospatial or hydrographic region.

The exact process of building or updating a model can be configured in a single configuration :ref:`.yaml file <model_config>`.
This file describes the full pipeline of model methods and their arguments. The methods vary for the
different model classes and :ref:`plugins`, as documented in this documentation or for each plugin documentation website.

.. _model_interface:

Model API
---------

.. currentmodule:: hydromt

HydroMT defines any model through the model-agnostic Model API based on several *general* components and *computational* unit components.
Each component represents a specific model data type and is parsed to a specific Python data object.
The general components are **maps** (raster data), **geoms** (vector data), **forcing**, **results**, **states**, and **config** (the model simulation configuration). These are available to all model classes and plugins.

The computational components are different for different types of models: i.e. **grid** for distributed or grid models, **vector** for vector or semi-distributed models, **mesh** for mesh or unstructured grid models, and **network** for network models (to be developed).

By default, the model components are returned and read from standard formats, as documented in the :ref:`API reference <api_reference>`.
While a generalized model class can readily be used, it can also be tailored to specific model software through so-called :ref:`plugins`. These plugins have the same model components (i.e. Model API), but with model-specific file readers and writers and workflows.

.. NOTE::

  As of version 0.6.0, the grid model (distributed grid model), vector model (semi-distributed and lumped models), mesh model (unstructured grid(s) models) have been implemented. Other model classes such as network models will follow in future versions.

The table below lists the base model components common to all model classes.
All base model attributes and methods can be found the :ref:`API reference <model_api>`

.. list-table::
   :widths: 15 25 20
   :header-rows: 1

   * - Component
     - Explanation
     - API
   * - maps
     - Map data (resolution and CRS may vary between maps)
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


For each generalized model class, the respective computational unit components exist:

.. list-table::
   :widths: 15 20 25 20
   :header-rows: 1

   * - Component
     - Model class
     - Explanation
     - API
   * - grid
     - :ref:`GridModel <grid_model_api>`
     - Static gridded data with on unified grid
     - | :py:attr:`~GridModel.grid`
       | :py:func:`~GridModel.set_grid`
       | :py:func:`~GridModel.read_grid`
       | :py:func:`~GridModel.write_grid`
   * - vector
     - :ref:`VectorModel <vector_model_api>`
     - Static polygon data over the vector units
     - | :py:attr:`~VectorModel.vector`
       | :py:func:`~VectorModel.set_vector`
       | :py:func:`~VectorModel.read_vector`
       | :py:func:`~VectorModel.write_vector`
   * - mesh
     - :ref:`MeshModel <mesh_model_api>`
     - Static mesh (unstructured grid(s)) data
     - | :py:attr:`~MeshModel.mesh`
       | :py:func:`~MeshModel.set_mesh`
       | :py:func:`~MeshModel.get_mesh`
       | :py:func:`~MeshModel.read_mesh`
       | :py:func:`~MeshModel.write_mesh`



.. NOTE::

    Prior to v0.6.0, the *staticmaps* and *staticgeoms* components were available.
    *staticmaps* is replaced with *grid* in GridModel,
    whereas *staticgeoms* is renamed to *geoms* for consistency but still available in the Model class.
