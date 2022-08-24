.. _model_update:

Updating a model
================

To add or change one or more components of an existing model the ``update`` method can be used.
The update method works identical for all :ref:`HydroMT model plugins <plugins>`, 
but the model methods (i.e. sections and options in the :ref:`.ini configuration file <model_config>`) are different for each model.

**Steps in brief:**

1) You have an **existing model** schematization. This model does not have to be complete.
2) Prepare or use a pre-defined **data catalog** with all the required data sources, see :ref:`working with data <get_data>`
3) Prepare a **model configuration** with the methods that you want to use to add or change components of your model: see :ref:`model configuration <model_config>`.
4) **Update** your model using the CLI or Python interface

.. TIP::

    By default all model data is written at the end of the update method. If your update however 
    only affects a certain model data (e.g. staticmaps or forcing) you can add a write_* method 
    (e.g. `write_staticmaps`, `write_forcing`) to the .ini file and only these data will be written.
    
    Note that the model config is often changed as part of the a model method and `write_config` 
    should thus be added to the .ini file to keep the model data and config consistent.

.. _cli_update:

From CLI
--------

The ``hydromt update`` command line interface (CLI) method can be run from the command line after the right conda environment is activated. 

By default, the model is updated in place, overwriting the existing model schematization. 
To save a copy of the model provide a new output model root directory with the ``-o`` option.

By default, all model methods in the .ini configuration file provided with ``-i`` will be updated. 
To update only certain methods, the ``-c <method>`` option can be used to select methods 
in combination with :ref:`.ini file <model_config>`.
Besides the ini file, method arguments can be set from the CLI with ``--opt <method.argument=value>``.
If used in combination with an .ini file, it will overwrite the same arguments in the .ini file. 
Both ``-c`` and ``-opt`` can be used repeatedly in a single update.


**Example usage**

In the following example a Wflow model at ``/path/to/model`` is updated and the results are written to a new directory ``/path/to/model_out``.
The pipeline with methods which are updated are outlined in the ``wflow_config.ini`` configuration file and used data sources
in the ``data_catalog.yml`` catalog file.

.. code-block:: console

    hydromt update wflow /path/to/model_root -o /path/to/model_out -i /path/to/wflow_config.ini -d /path/to/data_catalog.yml -v

The following example updates (overwrites!) the landuse-landcover based staticmaps in a Wflow model with the ``setup_lulcmaps`` method 
based on a the different landuse-landcover dataset according to ``setup_lulcmaps.lulc_fn=vito``. 
The ``vito`` dataset must be defined in the ``data_catalog.yml`` catalog file.
Note that no .ini file is used here but instead the methods and options are defined in the update command.

.. code-block:: console

    hydromt update wflow /path/to/model_root -c setup_lulcmaps -c write_staticmaps --opt setup_lulcmaps.lulc_fn=vito -d /path/to/data_catalog.yml -v


**Overview of options**

To check all options do:

.. code-block:: console

    hydromt update --help

.. include:: ../_generated/cli_update.rst


.. _python_update:

From Python
-----------

All HydroMT models have an :py:func:`~hydromt.Model.update` method which can be used when updating models from Python.
The data catalog yaml files and logging have to be set when initializing the model. 
The configuration file can be parsed using :py:func:`~hydromt.config.configread` and passed to the build method using the ``opt`` argument.

**Example usage**

To update a Wflow model based on methods in an .ini file, as also shown in the first CLI example above, the following Python code is required.
Note that compared to building a model, the model should be initialized in read (if you save the output to a new root) 
or append (if you update the model data in place) mode.

.. code-block::  python

    from hydromt_wflow import WflowModel
    from hydromt.config import configread
    data_libs = [r'/path/to/data_catalog.yml']
    opt=configread(r'/path/to/wflow_config.ini')  # parse .ini configuration
    mod = WflowModel(r'/path/to/model_root', data_libs=data_libs, mode='r')  # initialize model with default logger in read mode
    mod.update(model_out=r'/path/to/model_out', opt=opt)

To update a single component of a Wflow model from Python, the model methods can also be called directly instead of using the update method.
Note that this will however not log the used methods and arguments making your model harder to reproduce. To change the model root before writing 
the updated model use the :py:func:`~hydromt.Model.set_root` method (not shown in this example).

.. code-block::  python

    from hydromt_wflow import WflowModel
    data_libs = [r'/path/to/data_catalog.yml']  # this catalog contains the 'vito' data source
    mod = WflowModel(r'/path/to/model_root', data_libs=data_libs, mode='r+')  # initialize model with default logger in append mode
    mod.setup_lulcmaps(lulc_fn='vito')
    mod.write_staticmaps()  # write static maps component with updated lulc maps