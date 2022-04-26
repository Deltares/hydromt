.. _cli_build:

Building a model
================

To build a complete model from scratch using available data the ``build`` command can be used.
The interface is identical for each model, but the configuration file has different 
options (see documentation of the individual models). The mandatory :ref:`region <region>`
argument describes the region of interest. The build method will start by running the component in which
the model grid (if applicable) is defined for the region, usually the `setup_basemaps` method.
The configuration file should listing all the components that the user wants to include during the build. 

.. Tip::
    
    The verbosity of the log messages can be increased with `-v` for info and `-vv` for debug messages.

After activating the HydroMT python environment, the HydroMT ``build`` method can be run from the command line. 
To check its options run:

.. code-block:: console

    hydromt build --help

**Example usage**

.. code-block:: console

    To build a wflow model for a subbasin using and point coordinates snapped to cells with stream order >= 4
    hydromt build wflow /path/to/model_root "{'subbasin': [-7.24, 62.09], 'strord': 4}" -i /path/to/wflow_config.ini


    To build a sfincs model based on a bbox (for Texel)
    hydromt build sfincs /path/to/model_root "{'bbox': [4.6891,52.9750,4.9576,53.1994]}" -i /path/to/sfincs_config.ini

**Further options**

.. include:: ../_generated/cli_build.rst