.. _model_clip:

Clipping a model
================

The ``clip`` method allows to clip a subregion from a existing model, including all static maps,
static geometries and forcing data.

**Steps in brief:**

1) You have an **existing model** schematization ready
2) Define your **model subregion**, see the overview of :ref:`region options <region>`. 
3) **Clip** you model using the CLI or Python interface

.. NOTE::

    This method is not yet implemented for all plugins. Please check the documentation of the respective
    :ref:`plugin<plugins>` for more information on whether the clip method is available.


.. _cli_clip:

From CLI
--------

**Example usage**

The ``hydromt clip`` command line intrefce (CLI) method can be run from the command line after the right conda environment is activated. 

The following example clips a ``subbasin`` region (based on its outflow location) from an existing Wflow model 
at `/path/to/model_root` and saves the output to `/path/to/model_destination`. The subbasin is defined based
on an outflow location snapped to a stream order 4 river which based on a map called wflow_streamorder 
of the Wflow model. 

.. code-block:: console
   
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'subbasin': [-7.24, 62.09], 'wflow_streamorder': 4}"


**Overview of options**

To check all options do:

.. code-block:: console

    hydromt clip --help

.. include:: ../_generated/cli_clip.rst


.. _python_clip:

From Python
-----------

.. NOTE:: 

    A general clip method for the Model class is currently not yet available and the Python signature of the might change in the future.

**Example usage**

.. code-block::  python

    from hydromt_wflow import WflowModel
    mod = WflowModel(r'/path/to/model_root', mode='r')  # initialize model with default logger in read mode
    mod.clip_staticmaps(region={'subbasin': [-7.24, 62.09], 'wflow_streamorder': 4})
    mod.clip_forcing()
    mod.set_root(r'path/to/model_destination')  # change root to output directory
    mod.write()