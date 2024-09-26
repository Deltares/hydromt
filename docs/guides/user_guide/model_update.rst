.. _model_update:

Updating a model
================

To add or change one or more components of an existing model the ``update`` method can be used.
The update method works identical for all :ref:`HydroMT model plugins <plugins>`,
but the model methods (i.e. sections and options in the :ref:`.yaml workflow file <model_workflow>`) are different for each model.

**Steps in brief:**

1) You have an **existing model** schematization. This model does not have to be complete.
2) Prepare or use a pre-defined **data catalog** with all the required data sources, see :ref:`working with data <get_data>`
3) Prepare a **model workflow** with the methods that you want to use to add or change components of your model: see :ref:`model workflow <model_workflow>`.
4) **Update** your model using the CLI or Python interface

.. _cli_update:

From CLI
--------

The ``hydromt update`` command line interface (CLI) method can be run from the command line after the right conda environment is activated.

By default, the model is updated in place, overwriting the existing model schematization.
To save a copy of the model provide a new output model root directory with the ``-o`` option.

By default, all model methods in the .yaml configuration file provided with ``-i`` will be updated.
To update only certain methods, the ``-c <method>`` option can be used to select methods
in combination with :ref:`.yaml file <model_workflow>`.
Besides the .yaml file, method arguments can be set from the CLI with ``--opt <method.argument=value>``.
If used in combination with an .yaml file, it will overwrite the same arguments in the .yaml file.
Both ``-c`` and ``-opt`` can be used repeatedly in a single update.


**Example usage**

In the following example a Wflow model at ``/path/to/model`` is updated and the results are written to a new directory ``/path/to/model_out``.
The pipeline with methods which are updated are outlined in the ``wflow_config.yaml`` configuration file and used data sources
in the ``data_catalog.yml`` catalog file.

.. code-block:: console

    hydromt update wflow /path/to/model_root -o /path/to/model_out -i /path/to/wflow_config.yaml -d /path/to/data_catalog.yml -v

The following example updates (overwrites!) the landuse-landcover based staticmaps in a Wflow model with the ``setup_lulcmaps`` method
based on a the different landuse-landcover dataset according to ``setup_lulcmaps.lulc_fn=vito``.


**Overview of options**

To check all options see :ref: `the CLI API`, or do:

.. code-block:: console

    hydromt update --help
