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

The computational components are different for different types of models: i.e. **grid** for distributed or grid models, **vector** for lumped or semi-distributed models, **mesh** for mesh or unstructured grid models, and **network** for network models (to be developed).

By default, the model components are returned and read from standard formats, as documented in the :ref: `API reference`.
While a generalized model class can readily be used, it can also be tailored to specific model software through so-called :ref:`plugins`. These plugins have the same model components (i.e. Model API), but with model-specific file readers and writers and workflows.
