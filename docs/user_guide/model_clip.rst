.. _cli_clip:

Clipping a model
----------------

.. warning::

    This method is not implemented for all plugins. Please check the documentation of the respective
    :ref:`plugin<plugins>` for more information on whether the clip method is available.

The ``clip`` method allows to clip a subregion from a model, including all static maps,
static geometries and forcing data. The :ref:`region <region>` argument follows the same syntax as
in the :ref:`build method <cli_build>`.

After activating the HydroMT python environment, the HydroMT ``clip`` method can be run from the command line.
To check its options run:

.. code-block:: console

    hydromt clip --help

**Example usage**

.. code-block:: console

    Example usage to clip a wflow model for a subbasin derived from point coordinates
    snapped to cells with stream order >= 4
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'subbasin': [-7.24, 62.09], 'wflow_streamorder': 4}"

    Example usage basin based on ID from model_root basins map
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'basin': 1}"

    Example usage basins whose outlets are inside a geometry
    hydromt clip wflow /path/to/model_root /path/to/model_destination "{'outlet': 'geometry.geojson'}"

**Further options**

.. include:: ../_generated/cli_clip.rst


