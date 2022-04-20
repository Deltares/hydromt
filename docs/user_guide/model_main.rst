.. _model_main:

Working with models in HydroMT  
==============================

High level functionality
------------------------

HydroMT has the following high-level functionality for setting up models from raw data or adjusting models: 

* :ref:`defining a model region <region>` for *building* or *clipping* models.
* :ref:`building a model <cli_build>`: building a model from scratch.
* :ref:`updating a model <cli_update>`: updating an existing model (e.g. add model components or change data source).
* :ref:`clipping a model <cli_clip>`: changing the spatial domain of an existing model (e.g. select subbasins from a larger model).

The exact process of building or updating a model can be configured in a single configuration *.ini* file. 
This file describes the full pipeline of model components and their arguments. The components vary for the 
different model :ref:`plugins` and are documented for each at their respective documentation websites.

.. _model_interface:

Model interface
---------------

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
    Example: Hydrographic regions <../_examples/delineate_basin.ipynb>
