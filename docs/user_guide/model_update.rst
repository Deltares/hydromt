.. _cli_update:

Updating a model
================

The ``update`` method can be used to update one or several model components. The model components are
identical to the headers in the ini-files of each model. Options for a component 
can be set in the ini-file or provides via command line with the `c` and `opt` options if only one component 
is updated. For several, use the configuration file with the `i` option.

After activating the conda environment, the HydroMT ``update`` method can be run from the command line:

.. code-block:: console

    hydromt update

**Example usage**

.. code-block:: console

    Update (overwrite) landuse-landcover maps in a wflow model
    hydromt update wflow /path/to/model_root -c setup_lulcmaps --opt source_name=vito

    Update reservoir maps based on default settings in a wflow model and write to new directory
    hydromt update wflow /path/to/model_root -o /path/to/model_out -c setup_reservoirs

**Further options**

.. include:: ../_generated/cli_update.rst


