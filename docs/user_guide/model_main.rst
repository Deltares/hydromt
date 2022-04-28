.. _model_main:

Working with models in HydroMT  
==============================

High level functionality
------------------------

HydroMT has the following high-level functionality for setting up models from raw data or adjusting models: 

* :ref:`building a model <model_build>`: building a model from scratch.
* :ref:`updating a model <model_update>`: adding or changing model components of an existing model.
* :ref:`clipping a model <model_clip>`: changing the spatial domain of an existing model (e.g. select subbasins from a larger model).

The building and clipping methods required the user to provide a :ref:`region` of interest. HydroMT provides 
several options to define a region based on a geospatial or hydrographic regions.

The exact process of building or updating a model can be configured in a single configuration *.ini* file. 
This file describes the full pipeline of model methods and their arguments. The methods vary for the 
different model :ref:`plugins` and are documented for each at their respective documentation websites.

.. _model_interface:

Model data components
---------------------

#TODO

Supported models
----------------

For a list of supported models see the :ref:`plugins` page.

.. toctree::
    :hidden:
    
    model_build.rst
    model_update.rst
    model_clip.rst
    model_region.rst
    model_config.rst
