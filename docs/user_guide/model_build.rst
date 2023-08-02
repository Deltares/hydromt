.. _model_build:

Building a model
================

To build a complete model from scratch using available data the ``build`` method can be used.
The build method is identical for all :ref:`HydroMT model plugins <plugins>`,
but the model methods (i.e. sections and options in the .yaml configuration file) are different for each model.

**Steps in brief:**

1) Prepare or use a pre-defined **data catalog** with all the required data sources, see :ref:`working with data <get_data>`
2) Define your **model region**, see the overview of :ref:`region options <region>`.
3) Prepare a **model configuration** which describes the complete pipeline to build your model: see :ref:`model configuration <model_config>`.
4) **Build** you model using the CLI or Python interface

.. _cli_build:

From CLI
--------

The ``hydromt build`` command line interface (CLI) method can be run from the command line after the right conda environment is activated.
The HydroMT core package contain implementation for generalized model classes. Specific model implementation for softwares have to be built
from associated :ref:`HydroMT plugin <plugins>` that needs to be installed to your Python environment.

To check which HydroMT model plugins are installed, do:

.. code-block:: console

    hydromt --models


.. NOTE::
    From version 0.7.0 onwards it is required to add a ``-r (--region)`` flag before REGION when defining a region with the CLI.
    Prior to this change the ``-r`` flag was reserved for resolution.
    Resolution for gridded models now have to be set in the .yaml configuration files, e.g., under setup_basemaps for HydroMT-Wflow.

**Example usage**

The following line of code builds a SFINCS model for a region defined by a bounding box ``bbox`` and based on the model methods
in the ``sfincs_config.yaml`` file and the data sources in the ``data_catalog.yml`` file.

.. code-block:: console

    hydromt build sfincs /path/to/model_root -r "{'bbox': [4.6891,52.9750,4.9576,53.1994]}" -i /path/to/sfincs_config.yaml -d /path/to/data_catalog.yml -v


The following line of code builds a SFINCS model for a region defined by a bounding box ``bbox`` and based on the model methods
in the ``grid_model_config.yaml`` file and the data sources in the ``data_catalog.yml`` file.

.. code-block:: console

    hydromt build grid_model /path/to/model_root -r "{'bbox': [4.6891,52.9750,4.9576,53.1994]}" -i /path/to/grid_model_config.yaml -d /path/to/data_catalog.yml -v

.. Tip::

    The verbosity of the log messages can be increased with ``-v`` for info and ``-vv`` for debug messages.

**Overview of options**

To check all options do:

.. code-block:: console

    hydromt build --help

.. include:: ../_generated/cli_build.rst

.. _python_build:

From Python
-----------

All HydroMT models have a :py:func:`~hydromt.Model.build` method which can be used when building models from Python.
The data catalog yaml files and logging have to be set when initializing the model.
The configuration file can be parsed using :py:func:`~hydromt.config.configread` and passed to the build method using the ``opt`` argument.

**Example usage**

To create the same SFINCS model as shown above in the CLI example the following block of Python code is required.

.. code-block::  python

    from hydromt_sfincs import SfincsModel
    from hydromt.config import configread
    data_libs = [r'/path/to/data_catalog.yml']
    model_root = r'/path/to/model_root
    opt=configread(r'/path/to/sfincs_config.yaml')  # parse .yaml configuration
    mod = SfincsModel(model_root, data_libs=data_libs)  # initialize model with default logger
    mod.build(region={'bbox': [4.6891,52.9750,4.9576,53.1994]}, opt=opt)

To create the same gridded model:

.. code-block::  python

    from hydromt.models.model_grid import GridModel
    from hydromt.config import configread
    data_libs = [r'/path/to/data_catalog.yml']
    model_root = r'/path/to/model_root
    opt=configread(r'/path/to/grid_model_config.yaml')  # parse .yaml configuration
    mod = GridModel(model_root, data_libs=data_libs)  # initialize model with default logger
    mod.build(region={'bbox': [4.6891,52.9750,4.9576,53.1994]}, opt=opt)
