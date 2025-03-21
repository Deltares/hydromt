.. _model_build:

Building a model
================

To build a complete model from scratch using available data the ``build`` method can be used.
The build method is identical for all :ref:`HydroMT model plugins <plugins>`,
but the model methods (i.e. sections and options in the .yaml configuration file) are different for each model.

**Steps in brief:**

1) Prepare or use a pre-defined **data catalog** with all the required data sources, see :ref:`working with data <get_data>`
2) Prepare a **model workflow** which describes the complete pipeline to build your model: see :ref:`model workflow <model_workflow>`.
3) **Build** you model using the CLI or Python interface

.. _cli_build:

From CLI
--------

The ``hydromt build`` command line interface (CLI) method can be run from the command line after the right conda environment is activated.
The HydroMT core package contain implementation for generalized model classes. Specific model implementation for softwares have to be built
from associated :ref:`HydroMT plugin <plugins>` that needs to be installed to your Python environment.

To check which HydroMT model plugins are installed, do:

.. code-block:: console

    hydromt --models


**Example usage**

The following line of code builds a SFINCS model based on the model methods
in the ``sfincs_workflow.yaml`` file and the data sources in the ``data_catalog.yml`` file.

.. code-block:: console

    hydromt build sfincs /path/to/model_root -i /path/to/sfincs_config.yaml -d /path/to/data_catalog.yml -v


The following line of code builds a SFINCS model based on the model methods
in the ``grid_model_workflow.yaml`` file and the data sources in the ``data_catalog.yml`` file.

.. code-block:: console

    hydromt build grid_model /path/to/model_root -i /path/to/grid_model_workflow.yaml -d /path/to/data_catalog.yml -v

.. Tip::

    The verbosity of the log messages can be increased with ``-v`` for info and ``-vv`` for debug messages.

**Overview of options**

To check all options see :ref: `the CLI API`, or do:

.. code-block:: console

    hydromt build --help
