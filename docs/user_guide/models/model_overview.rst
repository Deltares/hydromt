.. _model_main:

Overview models
===============

High level functionality
------------------------

HydroMT has the following high-level functionality for setting up models from raw data or adjusting models:

* :ref:`building a model <model_build>`: building a model from scratch.
* :ref:`updating a model <model_update>`: adding or changing model components of an existing model.

The exact process of building or updating a model can be configured in a single configuration :ref:`.yaml file <model_workflow>`.
This file describes the full pipeline of model methods and their arguments. The methods vary for the
different model classes and :ref:`plugins`, as documented in this documentation or for
each plugin documentation website.
